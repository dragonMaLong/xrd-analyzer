"""
core/fitting.py
---------------
正则化 NNLS 拟合引擎。

模块顶层函数（_fit_with_mu_list_worker、_eval_candidate_for_index）
必须保持在模块顶层，以便 ProcessPoolExecutor 能够序列化（pickle）传入子进程。
"""
import numpy as np
from scipy.optimize import nnls
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


def build_basis_matrix(x, mu_list, D_range, lam1, lam2,
                        intensity_ratio=INTENSITY_RATIO):
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

    for mu in mu_list:
        mu_ka2 = calc_kalpha2_position(mu, lam1, lam2)

        gamma1, m1 = calc_peak_params_numba(mu,     lam1, D_range, SLOPE_M, M_REF_MIN, D_REF_MAX)
        gamma2, m2 = calc_peak_params_numba(mu_ka2, lam2, D_range, SLOPE_M, M_REF_MIN, D_REF_MAX)

        pk1 = pearson_vii_numba(x, mu,     gamma1, m1)
        pk2 = pearson_vii_numba(x, mu_ka2, gamma2, m2) * intensity_ratio

        basis_k1_list.append(pk1)
        basis_k2_list.append(pk2)

    basis_total = np.hstack([k1 + k2 for k1, k2 in zip(basis_k1_list, basis_k2_list)])
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


# ---------------------------------------------------------------------------
# 主拟合函数（在主进程中被 app_window.py 调用）
# ---------------------------------------------------------------------------

def fit_with_mu_list(x, y_scaled, mu_list, lam1, lam2, L_single, D_range, alpha,
                     intensity_ratio=INTENSITY_RATIO):
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
        x, mu_list, D_range, lam1, lam2, intensity_ratio
    )
    f_total, resid = solve_nnls_regularized(
        basis_total, y_scaled, L_single, len(mu_list), alpha
    )
    if f_total.sum() <= 1e-9 or not np.isfinite(f_total).all():
        return np.inf, None, None, None
    return resid, f_total, basis_k1_list, basis_k2_list


# ---------------------------------------------------------------------------
# 子进程专用函数（必须在模块顶层，ProcessPoolExecutor 才能 pickle）
# ---------------------------------------------------------------------------

def _fit_with_mu_list_worker(x, y_scaled, mu_list, lam1, lam2, intensity_ratio,
                              L_single, D_range, alpha_val):
    """
    无 self 版本：供子进程使用的 NNLS 拟合入口。

    Returns (loss, f_total) — loss=inf 表示拟合失败。
    """
    precompile_numba_functions()  # 子进程首次调用时触发 JIT 编译

    basis_total, _, _ = build_basis_matrix(
        x, mu_list, D_range, lam1, lam2, intensity_ratio
    )
    f_total, resid = solve_nnls_regularized(
        basis_total, y_scaled, L_single, len(mu_list), alpha_val
    )
    if f_total.sum() <= 0 or not np.isfinite(f_total).all():
        return np.inf, None
    return resid, f_total


def _eval_candidate_for_index(mu_val, base_mu, peak_idx,
                               x, y_scaled, lam1, lam2, intensity_ratio,
                               L_single, D_range, alpha_val):
    """
    将第 peak_idx 个峰的 μ 替换为 mu_val，评估残差。
    供 ProcessPoolExecutor 并行扫描峰位时使用。

    Returns (loss, mu_val)
    """
    trial = list(base_mu)
    trial[peak_idx] = mu_val
    loss, _ = _fit_with_mu_list_worker(
        x, y_scaled, trial, lam1, lam2, intensity_ratio,
        L_single, D_range, alpha_val
    )
    return loss, mu_val
