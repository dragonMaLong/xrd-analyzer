"""
core/peak_functions.py
----------------------
NumPy 向量化的核心峰形计算函数，以及 Kα2 峰位计算。

所有函数均为纯函数（无副作用），可在主进程和子进程中安全调用。
"""
import numpy as np


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 各靶材的 Kα1 / Kα2 波长 (nm)
WAVELENGTHS: dict = {
    "Cu": (0.154059, 0.154442),
    "Co": (0.178897, 0.179285),
    "Fe": (0.193604, 0.193998),
    "Mo": (0.070930, 0.071359),
}

#: Scherrer 形状因子（球形颗粒，FWHM 基础上）
SCHERRER_K: float = 0.9

#: Pearson VII m 参数映射常量
#:   D = D_REF_MAX (100 nm) → m = M_REF_MIN (1.0, 可积洛伦兹极限)
#:   D → 0              → m → 5.0   (接近高斯)
M_REF_MIN: float = 1.0
D_REF_MAX: float = 100.0
SLOPE_M: float = (5.0 - M_REF_MIN) / (0.5 - D_REF_MAX)


# ---------------------------------------------------------------------------
# 向量化核函数
# ---------------------------------------------------------------------------

def pearson_vii_numba(xvals, mu, gamma, m):
    """
    计算 Pearson VII 峰形矩阵。

    Parameters
    ----------
    xvals : 1-D array, shape (N,)   — 2θ 取值点
    mu    : float                    — 峰中心 (2θ)
    gamma : 1-D array, shape (P,)   — 各晶粒尺寸对应的半宽参数 γ
    m     : 1-D array, shape (P,)   — 各晶粒尺寸对应的形状参数 m

    Returns
    -------
    out : 2-D array, shape (N, P)
        out[i, j] = (1 + ((xvals[i] - mu) / gamma[j])²)^(-m[j])
    """
    x = np.asarray(xvals, dtype=float)[:, None]
    gamma_arr = np.asarray(gamma, dtype=float)[None, :]
    m_arr = np.asarray(m, dtype=float)[None, :]
    return (1.0 + ((x - mu) / gamma_arr) ** 2.0) ** (-m_arr)


def sphere_interference_profile(xvals, mu, D, wavelength):
    """
    Spherical-crystallite interference kernel.

    This uses the squared sphere form factor, equivalent to the Fourier
    transform of the spherical common-volume function used in WPPM-style
    line-profile models. The returned profile is unit height; callers should
    area-normalize it on their chosen 2theta grid.
    """
    x = np.asarray(xvals, dtype=float)[:, None]
    D_arr = np.asarray(D, dtype=float)[None, :]
    theta = np.deg2rad(x / 2.0)
    theta0 = np.deg2rad(float(mu) / 2.0)
    q = 2.0 * (np.sin(theta) - np.sin(theta0)) / max(float(wavelength), 1e-15)
    z = np.pi * np.maximum(D_arr, 1e-12) * q
    abs_z = np.abs(z)
    amplitude = np.empty_like(z, dtype=float)
    small = abs_z < 1e-5
    amplitude[small] = 1.0 - (z[small] * z[small]) / 10.0
    zz = z[~small]
    amplitude[~small] = 3.0 * (np.sin(zz) - zz * np.cos(zz)) / (zz ** 3.0)
    profile = amplitude * amplitude
    return np.clip(profile, 0.0, None)


def calc_peak_params_numba(mu, wavelength, D_range, slope, M_ref_min, D_ref_max,
                           instrument_fwhm_deg=0.0):
    """
    根据晶粒尺寸数组计算 Pearson VII 的 γ 和 m 参数。

    Parameters
    ----------
    mu         : float            — 峰中心 (2θ, °)
    wavelength : float            — X 射线波长 (nm)
    D_range    : 1-D array (P,)  — 晶粒尺寸网格 (nm)
    slope      : float            — m–D 线性关系斜率
    M_ref_min  : float            — m 最小参考值
    D_ref_max  : float            — D 最大参考值 (nm)

    Returns
    -------
    gamma_vii : 1-D array (P,)   — Pearson VII γ 参数 (°)
    m         : 1-D array (P,)   — Pearson VII m 参数
    """
    theta = np.deg2rad(mu / 2.0)

    # Scherrer 展宽 → 转换为度
    sigma_rad = 0.9 * wavelength / (D_range * np.cos(theta))
    gamma_deg = sigma_rad * 180.0 / np.pi

    # m is clipped to [1.0, 5.0]. m=1 is the integrable Lorentzian limit.
    m = np.clip(M_ref_min + (D_range - D_ref_max) * slope, 1.0, 5.0)

    if instrument_fwhm_deg > 0.0:
        # Approximate convolution of size and instrument FWHM.
        # n=1 behaves like Lorentzian linear addition; n=2 behaves like
        # Gaussian quadratic addition. Pearson m interpolates between them.
        n_mix = np.clip(1.0 + (m - 1.0) / 4.0, 1.0, 2.0)
        gamma_deg = (gamma_deg ** n_mix + float(instrument_fwhm_deg) ** n_mix) ** (1.0 / n_mix)

    # Pearson VII γ 与 FWHM 的换算
    gamma_vii = gamma_deg / (2.0 * np.sqrt(2.0 ** (1.0 / m) - 1.0))
    return gamma_vii, m


# ---------------------------------------------------------------------------
# Python 级辅助函数
# ---------------------------------------------------------------------------

def calc_kalpha2_position(mu_kalpha1: float, lam1: float, lam2: float) -> float:
    """
    由 Kα1 峰位推算同一晶面间距对应的 Kα2 峰位。

    Parameters
    ----------
    mu_kalpha1 : Kα1 峰中心 (2θ, °)
    lam1       : Kα1 波长 (nm)
    lam2       : Kα2 波长 (nm)

    Returns
    -------
    mu_kalpha2 : Kα2 峰中心 (2θ, °)
    """
    d = lam1 / (2.0 * np.sin(np.deg2rad(mu_kalpha1 / 2.0)))
    return 2.0 * np.rad2deg(np.arcsin(lam2 / (2.0 * d)))


def precompile_numba_functions() -> None:
    """
    兼容旧调用路径的预热函数。
    当前实现已改为 NumPy 向量化，不再引入 Numba/llvmlite 打包依赖。
    """
    try:
        dummy_x = np.linspace(58.0, 74.0, 100)
        dummy_D = np.linspace(1.0, 50.0, 10)
        gamma, m = calc_peak_params_numba(
            68.0, 0.154059, dummy_D, SLOPE_M, M_REF_MIN, D_REF_MAX
        )
        _ = pearson_vii_numba(dummy_x, 68.0, gamma, m)
    except Exception as exc:
        print(f"[Peak] 峰形函数预热失败: {exc}")
