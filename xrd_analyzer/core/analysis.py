"""
core/analysis.py
----------------
NNLS 结果的后处理：从晶粒尺寸分布中识别局部峰，
计算各峰的中心、占比等统计量，并构建完整的 all_peak_info 数据结构。
"""
import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# 单峰分析
# ---------------------------------------------------------------------------

def calculate_peak_info(f_dist: np.ndarray, peak_indices: np.ndarray,
                         D_range: np.ndarray) -> tuple:
    """
    将一维分布 f_dist 中的局部峰信息提取出来。

    采用 Voronoi 分割（每个 D 点归属到距离最近的局部峰），
    计算各峰的中心位置、相对占比，以及所属 D 索引列表。

    Parameters
    ----------
    f_dist       : 1-D array — 单个 XRD 峰对应的晶粒尺寸分布
    peak_indices : 1-D array — find_peaks 找到的局部极大值索引
    D_range      : 1-D array — 晶粒尺寸网格 (nm)

    Returns
    -------
    peak_info : list[dict]
        每个 dict 包含 'center'、'percentage'、'indices'
    percentages : list[float]
    """
    if f_dist.sum() == 0 or len(peak_indices) == 0:
        return [], []

    total_weight = f_dist.sum()

    # 将每个 D 点分配给最近的局部峰（完整 Voronoi 覆盖，无死角）
    assignments = np.argmin(
        np.abs(D_range[:, np.newaxis] - D_range[peak_indices]), axis=1
    )

    peak_info = []
    for i, pk_idx in enumerate(peak_indices):
        idx = np.where(assignments == i)[0]
        peak_weight = float(f_dist[idx].sum())
        peak_info.append({
            "center":     float(D_range[pk_idx]),
            "percentage": (peak_weight / total_weight * 100.0) if total_weight > 0 else 0.0,
            "indices":    idx,
        })

    return peak_info, [info["percentage"] for info in peak_info]


# ---------------------------------------------------------------------------
# 全局后处理
# ---------------------------------------------------------------------------

def build_all_peak_info(best_f_total: np.ndarray,
                         active_peak_indices: list,
                         D_range: np.ndarray,
                         peak_colors: list,
                         all_basis_k1: list,
                         all_basis_k2: list) -> tuple:
    """
    将 NNLS 输出的拼接解向量拆分为各峰，并提取每个峰的分布统计。

    Parameters
    ----------
    best_f_total        : 1-D array — NNLS 求解结果（所有峰拼接）
    active_peak_indices : list[int] — 当前激活峰的编号
    D_range             : 1-D array — 晶粒尺寸网格 (nm)
    peak_colors         : list[str] — 各峰对应的颜色
    all_basis_k1        : list[ndarray] — 各峰的 Kα1 基矩阵
    all_basis_k2        : list[ndarray] — 各峰的 Kα2 基矩阵

    Returns
    -------
    all_peak_info      : list[dict]
    global_max_area    : float — 所有局部组分中面积最大值（用于全局归一化）
    """
    all_peak_info = []
    num_peaks = len(active_peak_indices)
    f_segments = np.split(best_f_total, num_peaks)

    for i in range(num_peaks):
        f_peak = f_segments[i]
        if f_peak.sum() == 0:
            continue

        max_val = f_peak.max()
        normalized_dist = f_peak / max_val if max_val > 0 else np.zeros_like(f_peak)

        local_peaks, _ = find_peaks(normalized_dist, height=0.001, distance=5)
        peak_details, _ = calculate_peak_info(f_peak, local_peaks, D_range)

        all_peak_info.append({
            "f_segment":      f_peak,
            "normalized_dist": normalized_dist,
            "peak_details":   peak_details,
            "color":          peak_colors[active_peak_indices[i]],
            "basis_k1":       all_basis_k1[i],
            "basis_k2":       all_basis_k2[i],
        })

    # ── 计算全局面积，供归一化使用 ──────────────────────────────────────────
    component_areas = []
    for pinfo in all_peak_info:
        f_seg = pinfo["f_segment"]
        for det in pinfo["peak_details"]:
            idx = det.get("indices", None)
            if idx is None or len(idx) == 0:
                det["area"]       = 0.0
                det["pct_global"] = 0.0
                continue
            area = float(f_seg[idx].sum())
            det["area"] = area
            component_areas.append(area)

    global_max_area = max(component_areas) if component_areas else 1.0
    global_max_area = max(global_max_area, 1e-12)

    for pinfo in all_peak_info:
        for det in pinfo["peak_details"]:
            det["pct_global"] = 100.0 * det.get("area", 0.0) / global_max_area

    return all_peak_info, global_max_area
