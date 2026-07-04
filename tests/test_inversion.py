import numpy as np
from scipy.signal import find_peaks

from xrd_analyzer.core.fitting import (
    WAVELENGTHS,
    build_basis_matrix,
    build_regularization_matrix,
    solve_nnls_regularized,
    solve_regularized_from_basis,
)


def _lognormal_volume_distribution(D, mode_nm=6.0, sigma=0.3):
    """Return a discrete, sum-normalized lognormal volume distribution."""
    D = np.asarray(D, dtype=float)
    mu = np.log(float(mode_nm)) + float(sigma) ** 2
    f = np.exp(-0.5 * ((np.log(D) - mu) / sigma) ** 2)
    f /= D * sigma * np.sqrt(2.0 * np.pi)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(f))
    return f / total


def test_lognormal_volume_distribution_recovery():
    rng = np.random.default_rng(1234)
    D_range = np.linspace(1.0, 30.0, 140)
    true_f = _lognormal_volume_distribution(D_range, mode_nm=6.0, sigma=0.3)

    x = np.linspace(30.0, 55.0, 1000)
    lam1, lam2 = WAVELENGTHS["Cu"]
    basis_total, _, _ = build_basis_matrix(x, [42.0], D_range, lam1, lam2)
    y_clean = basis_total.dot(true_f)
    noise = 0.002 * float(np.max(y_clean)) * rng.normal(size=y_clean.shape)
    y_noisy = np.clip(y_clean + noise, 0.0, None)

    L_single = build_regularization_matrix(len(D_range))
    recovered_f, _ = solve_nnls_regularized(
        basis_total,
        y_noisy,
        L_single,
        n_peaks=1,
        alpha=1.0,
    )

    recovered_sum = float(np.sum(recovered_f))
    assert recovered_sum > 0.0
    assert np.isclose(recovered_sum, 1.0, rtol=0.05, atol=0.05)

    recovered_norm = recovered_f / recovered_sum
    assert np.isclose(np.sum(recovered_norm), 1.0, rtol=1e-6, atol=1e-6)

    true_mean = float(np.sum(D_range * true_f) / np.sum(true_f))
    recovered_mean = float(np.sum(D_range * recovered_norm))
    assert abs(recovered_mean - true_mean) / true_mean < 0.10


def test_elastic_net_recovers_two_sparse_size_modes():
    rng = np.random.default_rng(4321)
    D_range = np.linspace(1.0, 25.0, 120)
    first = _lognormal_volume_distribution(D_range, mode_nm=4.0, sigma=0.13)
    second = _lognormal_volume_distribution(D_range, mode_nm=12.0, sigma=0.14)
    true_f = 0.55 * first + 0.45 * second
    true_f /= float(np.sum(true_f))

    x = np.linspace(30.0, 55.0, 1000)
    lam1, lam2 = WAVELENGTHS["Cu"]
    basis_total, _, _ = build_basis_matrix(x, [42.0], D_range, lam1, lam2)
    y_clean = basis_total.dot(true_f)
    noise = 0.001 * float(np.max(y_clean)) * rng.normal(size=y_clean.shape)
    y_noisy = np.clip(y_clean + noise, 0.0, None)

    L_single = build_regularization_matrix(len(D_range))
    l2_f, _ = solve_nnls_regularized(
        basis_total,
        y_noisy,
        L_single,
        n_peaks=1,
        alpha=20.0,
    )
    elastic_f, _ = solve_regularized_from_basis(
        basis_total,
        y_noisy,
        L_single,
        n_peaks=1,
        alpha=0.05,
        regularization_method="elastic_net",
    )
    legacy_f, _ = solve_regularized_from_basis(
        basis_total,
        y_noisy,
        L_single,
        n_peaks=1,
        alpha=0.05,
        regularization_method="dl_sr",
    )

    min_distance = 10
    l2_norm = l2_f / max(float(np.max(l2_f)), 1e-30)
    elastic_norm = elastic_f / max(float(np.max(elastic_f)), 1e-30)
    legacy_norm = legacy_f / max(float(np.max(legacy_f)), 1e-30)
    l2_peaks, _ = find_peaks(l2_norm, height=0.08, distance=min_distance)
    elastic_peaks, _ = find_peaks(elastic_norm, height=0.08, distance=min_distance)
    legacy_peaks, _ = find_peaks(legacy_norm, height=0.08, distance=min_distance)

    assert len(l2_peaks) == 1
    assert len(elastic_peaks) == 2
    assert len(legacy_peaks) == 2
    assert np.allclose(elastic_f, legacy_f, rtol=1e-10, atol=1e-12)
    assert np.allclose(D_range[elastic_peaks], [4.0, 12.0], atol=0.8)
