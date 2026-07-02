"""
core/fitting.py
---------------
正则化 NNLS 拟合引擎。

模块顶层函数（_fit_with_mu_list_worker、_eval_candidate_for_index）
必须保持在模块顶层，以便 ProcessPoolExecutor 能够序列化（pickle）传入子进程。
"""
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.sparse import diags
from scipy.linalg import block_diag

from .peak_functions import (
    pearson_vii_numba,
    calc_peak_params_numba,
    calc_kalpha2_position,
    precompile_numba_functions,
    SLOPE_M,
    M_REF_MIN,
    D_REF_MAX,
)

# Kα2 / Kα1 强度比（Cu 靶约 0.497，通常取 0.5）
INTENSITY_RATIO: float = 0.5

#: 各靶材的 Kα1 / Kα2 波长 (nm)，与 peak_functions.WAVELENGTHS 保持同步
WAVELENGTHS: dict = {
    "Cu": (0.154059, 0.154442),
    "Co": (0.178897, 0.179285),
    "Fe": (0.193604, 0.193998),
    "Mo": (0.070930, 0.071359),
}


# ---------------------------------------------------------------------------
# 矩阵构建工具
# ---------------------------------------------------------------------------

def build_regularization_matrix(n_d_points: int) -> np.ndarray:
    """
    构造一阶 Tikhonov 正则化矩阵 L（一阶差分，约束分布光滑性）。

    Parameters
    ----------
    n_d_points : 晶粒尺寸网格点数

    Returns
    -------
    L : ndarray, shape (n_d_points-1, n_d_points)
    """
    return diags([-1, 1], [0, 1], shape=(n_d_points - 1, n_d_points)).toarray()


def build_peak_basis(x, mu, D_range, lam1, lam2,
                     intensity_ratio=INTENSITY_RATIO,
                     instrument_fwhm_deg=0.0):
    mu_ka2 = calc_kalpha2_position(mu, lam1, lam2)

    gamma1, m1 = calc_peak_params_numba(
        mu, lam1, D_range, SLOPE_M, M_REF_MIN, D_REF_MAX,
        instrument_fwhm_deg,
    )
    gamma2, m2 = calc_peak_params_numba(
        mu_ka2, lam2, D_range, SLOPE_M, M_REF_MIN, D_REF_MAX,
        instrument_fwhm_deg,
    )

    pk1 = pearson_vii_numba(x, mu, gamma1, m1)
    pk2 = pearson_vii_numba(x, mu_ka2, gamma2, m2) * intensity_ratio
    return pk1, pk2


def build_basis_matrix(x, mu_list, D_range, lam1, lam2,
                        intensity_ratio=INTENSITY_RATIO,
                        instrument_fwhm_deg=0.0):
    """
    为给定的峰位列表构建完整基函数矩阵，同时返回各峰的 k1/k2 子矩阵。

    Parameters
    ----------
    x              : 1-D array — 截取后的 2θ 数据点
    mu_list        : list[float] — 各峰的 Kα1 中心位置 (2θ, °)
    D_range        : 1-D array — 晶粒尺寸网格 (nm)
    lam1, lam2     : float — Kα1 / Kα2 波长 (nm)
    intensity_ratio: float — Kα2/Kα1 强度比

    Returns
    -------
    basis_total    : ndarray, shape (N, P*n_peaks)
    basis_k1_list  : list[ndarray]  — 每个峰的 Kα1 子矩阵
    basis_k2_list  : list[ndarray]  — 每个峰的 Kα2 子矩阵
    """
    basis_k1_list, basis_k2_list = [], []
    n_rows = len(x)
    n_d = len(D_range)
    basis_total = np.empty((n_rows, len(mu_list) * n_d), dtype=float)

    for idx, mu in enumerate(mu_list):
        pk1, pk2 = build_peak_basis(
            x, mu, D_range, lam1, lam2, intensity_ratio,
            instrument_fwhm_deg,
        )

        basis_k1_list.append(pk1)
        basis_k2_list.append(pk2)
        basis_total[:, idx * n_d:(idx + 1) * n_d] = pk1 + pk2

    return basis_total, basis_k1_list, basis_k2_list


def solve_nnls_regularized(basis_total, y_scaled, L_single, n_peaks, alpha):
    """
    构造增广矩阵并执行正则化 NNLS 求解。

    增广系统：
        [  A  ]         [  y  ]
        [ αL  ] · f  =  [  0  ]

    Parameters
    ----------
    basis_total : ndarray, shape (N, P*n_peaks)
    y_scaled    : 1-D array, shape (N,) — 归一化后的信号
    L_single    : ndarray — 单峰的正则化矩阵
    n_peaks     : int — 峰数量
    alpha       : float — 正则化强度

    Returns
    -------
    f_total : 1-D array — NNLS 解（非负晶粒尺寸分布）
    resid   : float     — 残差范数 ‖A·f - y‖
    """
    L_combined = block_diag(*([L_single] * n_peaks))
    P_reg = np.vstack([basis_total, alpha * L_combined])
    y_reg = np.hstack([y_scaled, np.zeros(L_combined.shape[0])])
    f_total, _ = nnls(P_reg, y_reg)
    resid = float(np.linalg.norm(basis_total.dot(f_total) - y_scaled))
    return f_total, resid


def solve_tv_regularized(
    basis_total,
    y_scaled,
    L_single,
    n_peaks,
    alpha,
    *,
    max_iter: int = 300,
    epsilon: float = 1e-6,
):
    """
    平滑 TV 正则化非负求解。

    目标函数:
        0.5 * ||A f - y||² + alpha * sum(sqrt((L f)² + epsilon²))

    与当前 L2/Tikhonov 方法并行存在，用于实验高分辨粒径分布。
    """
    basis_total = np.asarray(basis_total, dtype=float)
    y_scaled = np.asarray(y_scaled, dtype=float)
    L_combined = block_diag(*([L_single] * n_peaks))
    tv_weight = max(float(alpha), 1e-12)
    eps = max(float(epsilon), 1e-12)

    try:
        x0, _ = solve_nnls_regularized(basis_total, y_scaled, L_single, n_peaks, alpha)
    except Exception:
        x0 = np.zeros(basis_total.shape[1], dtype=float)
    if x0.size != basis_total.shape[1] or not np.isfinite(x0).all():
        x0 = np.zeros(basis_total.shape[1], dtype=float)

    def objective_and_grad(f):
        f = np.asarray(f, dtype=float)
        residual = basis_total.dot(f) - y_scaled
        diff = L_combined.dot(f)
        smooth_abs = np.sqrt(diff * diff + eps * eps)
        obj = 0.5 * float(residual.dot(residual)) + tv_weight * float(np.sum(smooth_abs))
        grad = basis_total.T.dot(residual)
        grad += tv_weight * L_combined.T.dot(diff / smooth_abs)
        return obj, grad

    result = minimize(
        objective_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * int(basis_total.shape[1]),
        options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6},
    )
    f_total = np.asarray(result.x, dtype=float)
    f_total = np.clip(f_total, 0.0, None)
    resid = float(np.linalg.norm(basis_total.dot(f_total) - y_scaled))
    return f_total, resid


def solve_hybrid_regularized(
    basis_total,
    y_scaled,
    L_single,
    n_peaks,
    alpha,
    *,
    tv_ratio: float = 0.20,
    max_iter: int = 300,
    epsilon: float = 1e-6,
):
    """
    L2 + TV 混合正则化非负求解。

    目标函数:
        0.5 * ||A f - y||²
        + 0.5 * alpha² * ||L f||²
        + alpha * tv_ratio * sum(sqrt((L f)² + epsilon²))

    L2 项负责整体平滑，TV 项保留局部边缘；相比纯 TV，通常能减轻柱状/阶梯状结果。
    """
    basis_total = np.asarray(basis_total, dtype=float)
    y_scaled = np.asarray(y_scaled, dtype=float)
    L_combined = block_diag(*([L_single] * n_peaks))
    alpha_val = max(float(alpha), 1e-12)
    l2_weight = alpha_val * alpha_val
    tv_weight = alpha_val * max(float(tv_ratio), 0.0)
    eps = max(float(epsilon), 1e-12)

    try:
        x0, _ = solve_nnls_regularized(basis_total, y_scaled, L_single, n_peaks, alpha)
    except Exception:
        x0 = np.zeros(basis_total.shape[1], dtype=float)
    if x0.size != basis_total.shape[1] or not np.isfinite(x0).all():
        x0 = np.zeros(basis_total.shape[1], dtype=float)

    def objective_and_grad(f):
        f = np.asarray(f, dtype=float)
        residual = basis_total.dot(f) - y_scaled
        diff = L_combined.dot(f)
        smooth_abs = np.sqrt(diff * diff + eps * eps)
        obj = 0.5 * float(residual.dot(residual))
        obj += 0.5 * l2_weight * float(diff.dot(diff))
        obj += tv_weight * float(np.sum(smooth_abs))
        grad = basis_total.T.dot(residual)
        grad += l2_weight * L_combined.T.dot(diff)
        if tv_weight > 0.0:
            grad += tv_weight * L_combined.T.dot(diff / smooth_abs)
        return obj, grad

    result = minimize(
        objective_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * int(basis_total.shape[1]),
        options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6},
    )
    f_total = np.asarray(result.x, dtype=float)
    f_total = np.clip(f_total, 0.0, None)
    resid = float(np.linalg.norm(basis_total.dot(f_total) - y_scaled))
    return f_total, resid


def _moving_average_1d(values, radius: int):
    radius = int(radius)
    if radius <= 0 or len(values) <= 2:
        return np.asarray(values, dtype=float)
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(values, dtype=float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _deep_sr_prior(f_total, n_peaks: int, n_d: int, alpha: float):
    """Build an untrained deep-prior style super-resolution target."""
    f_total = np.asarray(f_total, dtype=float)
    if f_total.size != int(n_peaks) * int(n_d):
        return np.clip(f_total, 0.0, None)

    alpha_val = max(float(alpha), 1e-6)
    detail_gain = 0.28 + 0.22 / np.sqrt(alpha_val + 0.25)
    shrink_ratio = 0.004 + 0.010 / np.sqrt(alpha_val + 0.25)
    rows = np.clip(f_total.reshape(int(n_peaks), int(n_d)), 0.0, None)
    prior_rows = []

    for row in rows:
        original_area = float(np.sum(row))
        if original_area <= 1e-14 or not np.isfinite(row).all():
            prior_rows.append(np.zeros_like(row))
            continue

        current = row.copy()
        for layer in range(4):
            fine = _moving_average_1d(current, 1)
            coarse = _moving_average_1d(current, 3 + layer)
            detail = fine - coarse
            local_scale = max(float(np.nanmax(fine)), 1e-12)
            gate = np.tanh(detail / (0.08 * local_scale + 1e-12))
            current = current * (1.0 + detail_gain * np.maximum(gate, 0.0))
            current += 0.18 * detail_gain * np.maximum(detail, 0.0)
            current = np.maximum(current - shrink_ratio * local_scale, 0.0)
            area = float(np.sum(current))
            if area > 1e-14:
                current *= original_area / area

        current = 0.80 * current + 0.20 * _moving_average_1d(current, 1)
        area = float(np.sum(current))
        if area > 1e-14:
            current *= original_area / area
        prior_rows.append(np.clip(current, 0.0, None))

    return np.ravel(np.asarray(prior_rows, dtype=float))


def solve_deep_super_resolution(
    basis_total,
    y_scaled,
    L_single,
    n_peaks,
    alpha,
    *,
    max_iter: int = 220,
):
    """Experimental dependency-free deep-prior super-resolution solver."""
    basis_total = np.asarray(basis_total, dtype=float)
    y_scaled = np.asarray(y_scaled, dtype=float)
    n_peaks = int(n_peaks)
    n_d = int(L_single.shape[1])
    alpha_val = max(float(alpha), 1e-6)

    try:
        x0, _ = solve_hybrid_regularized(
            basis_total,
            y_scaled,
            L_single,
            n_peaks,
            alpha,
            tv_ratio=0.08,
            max_iter=120,
        )
    except Exception:
        x0, _ = solve_nnls_regularized(basis_total, y_scaled, L_single, n_peaks, alpha)
    x0 = np.asarray(x0, dtype=float)
    if x0.size != basis_total.shape[1] or not np.isfinite(x0).all():
        x0 = np.zeros(basis_total.shape[1], dtype=float)

    prior = _deep_sr_prior(x0, n_peaks, n_d, alpha_val)
    if prior.size != x0.size or not np.isfinite(prior).all():
        prior = np.clip(x0, 0.0, None)
    start = np.maximum(0.65 * x0 + 0.35 * prior, 0.0)

    L_combined = block_diag(*([L_single] * n_peaks))
    col_norm = np.sum(basis_total * basis_total, axis=0)
    col_scale = max(float(np.nanmedian(col_norm)), 1e-12)
    smooth_weight = (0.10 * alpha_val) ** 2
    prior_weight = col_scale * (0.045 + 0.025 * np.log10(alpha_val + 1.0))
    positive = start[start > 0]
    sparse_scale = float(np.nanmedian(positive)) if positive.size else 1.0
    sparse_scale = max(sparse_scale, 1e-12)
    sparse_weight = col_scale * sparse_scale * (0.0015 / np.sqrt(alpha_val + 0.25))

    def objective_and_grad(f):
        f = np.asarray(f, dtype=float)
        residual = basis_total.dot(f) - y_scaled
        diff = L_combined.dot(f)
        prior_diff = f - prior
        f_nonneg = np.maximum(f, 0.0)
        obj = 0.5 * float(residual.dot(residual))
        obj += 0.5 * smooth_weight * float(diff.dot(diff))
        obj += 0.5 * prior_weight * float(prior_diff.dot(prior_diff))
        obj += sparse_weight * float(np.sum(np.log1p(f_nonneg / sparse_scale)))

        grad = basis_total.T.dot(residual)
        grad += smooth_weight * L_combined.T.dot(diff)
        grad += prior_weight * prior_diff
        grad += sparse_weight / (sparse_scale + f_nonneg)
        return obj, grad

    result = minimize(
        objective_and_grad,
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * int(basis_total.shape[1]),
        options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6},
    )
    f_total = np.asarray(result.x, dtype=float)
    f_total = np.clip(f_total, 0.0, None)
    resid = float(np.linalg.norm(basis_total.dot(f_total) - y_scaled))
    return f_total, resid


# ---------------------------------------------------------------------------
# 主拟合函数（在主进程中被 app_window.py 调用）
# ---------------------------------------------------------------------------

def solve_regularized_from_basis(
    basis_total,
    y_scaled,
    L_single,
    n_peaks,
    alpha,
    regularization_method: str = "l2",
):
    """Solve with an already-built basis matrix.

    Used by the alpha fast path after peak positions and the data window are fixed.
    """
    method = str(regularization_method or "l2").lower()
    if method in {"hybrid", "mixed", "l2_tv", "l2+tv"}:
        return solve_hybrid_regularized(
            basis_total, y_scaled, L_single, n_peaks, alpha
        )
    if method in {"dl_sr", "deep_sr", "deep_learning", "super_resolution"}:
        return solve_deep_super_resolution(
            basis_total, y_scaled, L_single, n_peaks, alpha
        )
    if method == "tv":
        return solve_tv_regularized(
            basis_total, y_scaled, L_single, n_peaks, alpha
        )
    return solve_nnls_regularized(
        basis_total, y_scaled, L_single, n_peaks, alpha
    )


def fit_with_mu_list(x, y_scaled, mu_list, lam1, lam2, L_single, D_range, alpha,
                     intensity_ratio=INTENSITY_RATIO,
                     instrument_fwhm_deg=0.0,
                     regularization_method: str = "l2"):
    """
    给定峰位列表，完整执行一次正则化 NNLS 拟合。

    Returns
    -------
    resid          : float
    f_total        : ndarray or None（若解全为零则返回 None）
    basis_k1_list  : list[ndarray]
    basis_k2_list  : list[ndarray]
    """
    basis_total, basis_k1_list, basis_k2_list = build_basis_matrix(
        x, mu_list, D_range, lam1, lam2, intensity_ratio,
        instrument_fwhm_deg
    )
    f_total, resid = solve_regularized_from_basis(
        basis_total,
        y_scaled,
        L_single,
        len(mu_list),
        alpha,
        regularization_method,
    )
    if f_total.sum() <= 1e-9 or not np.isfinite(f_total).all():
        return np.inf, None, None, None
    return resid, f_total, basis_k1_list, basis_k2_list


# ---------------------------------------------------------------------------
# 子进程专用函数（必须在模块顶层，ProcessPoolExecutor 才能 pickle）
# ---------------------------------------------------------------------------

def _fit_with_mu_list_worker(x, y_scaled, mu_list, lam1, lam2, intensity_ratio,
                              L_single, D_range, alpha_val,
                              instrument_fwhm_deg=0.0):
    """
    无 self 版本：供子进程使用的 NNLS 拟合入口。

    Returns (loss, f_total) — loss=inf 表示拟合失败。
    """
    precompile_numba_functions()  # 子进程首次调用时触发 JIT 编译

    basis_total, _, _ = build_basis_matrix(
        x, mu_list, D_range, lam1, lam2, intensity_ratio,
        instrument_fwhm_deg
    )
    f_total, resid = solve_nnls_regularized(
        basis_total, y_scaled, L_single, len(mu_list), alpha_val
    )
    if f_total.sum() <= 0 or not np.isfinite(f_total).all():
        return np.inf, None
    return resid, f_total


def _eval_candidate_chunk_for_index(candidates, base_mu, peak_idx,
                                    x, y_scaled, lam1, lam2, intensity_ratio,
                                    L_single, D_range, alpha_val,
                                    instrument_fwhm_deg=0.0):
    """
    Evaluate several candidate positions for one peak.

    The unchanged peak basis blocks are built once per chunk; each candidate only
    rebuilds the basis block for peak_idx. This keeps the scan result equivalent
    to _eval_candidate_for_index while avoiding repeated work.
    """
    precompile_numba_functions()

    base_mu = list(base_mu)
    candidates = [float(mu) for mu in candidates]
    peak_idx = int(peak_idx)
    n_peaks = len(base_mu)
    n_d = int(len(D_range))
    if not candidates or n_peaks <= 0 or n_d <= 0:
        return []

    basis_total = np.empty((len(x), n_peaks * n_d), dtype=float)
    current_slice = slice(peak_idx * n_d, (peak_idx + 1) * n_d)

    for idx, mu in enumerate(base_mu):
        if idx == peak_idx:
            continue
        pk1, pk2 = build_peak_basis(
            x, mu, D_range, lam1, lam2, intensity_ratio,
            instrument_fwhm_deg,
        )
        basis_total[:, idx * n_d:(idx + 1) * n_d] = pk1 + pk2

    results = []
    for mu_val in candidates:
        try:
            pk1, pk2 = build_peak_basis(
                x, mu_val, D_range, lam1, lam2, intensity_ratio,
                instrument_fwhm_deg,
            )
            basis_total[:, current_slice] = pk1 + pk2
            f_total, resid = solve_nnls_regularized(
                basis_total, y_scaled, L_single, n_peaks, alpha_val
            )
            if f_total.sum() <= 0 or not np.isfinite(f_total).all():
                resid = np.inf
        except Exception:
            resid = np.inf
        results.append((float(resid), float(mu_val)))
    return results


def _eval_candidate_for_index(mu_val, base_mu, peak_idx,
                               x, y_scaled, lam1, lam2, intensity_ratio,
                               L_single, D_range, alpha_val,
                               instrument_fwhm_deg=0.0):
    """
    将第 peak_idx 个峰的 μ 替换为 mu_val，评估残差。
    供 ProcessPoolExecutor 并行扫描峰位时使用。

    Returns (loss, mu_val)
    """
    trial = list(base_mu)
    trial[peak_idx] = mu_val
    loss, _ = _fit_with_mu_list_worker(
        x, y_scaled, trial, lam1, lam2, intensity_ratio,
        L_single, D_range, alpha_val, instrument_fwhm_deg
    )
    return loss, mu_val
