"""
io/file_reader.py
-----------------
XRD 数据文件的读取接口。

目前支持：
  - 两列 TXT 格式（第一行为样品名称，后续为 2θ 与强度）

预留扩展：
  - Bruker .raw (二进制 v3/v4)
  - Rigaku .raw (带头部的 ASCII)
"""
import os
import numpy as np


# ---------------------------------------------------------------------------
# TXT 读取
# ---------------------------------------------------------------------------

def load_txt_file(file_path: str) -> tuple:
    """
    读取两列格式的 TXT 文件。

    文件格式约定
    ------------
    第 1 行 : 样品名称（纯文本，可为空）
    第 2 行起: 以空白字符分隔的两列数值
               列 0 → 2θ (°)
               列 1 → 计数强度

    Parameters
    ----------
    file_path : str — 文件绝对路径

    Returns
    -------
    x_data      : np.ndarray — 2θ 数组 (°)
    y_data      : np.ndarray — 强度数组
    sample_name : str        — 样品名称（取自第一行，若为空则用文件名）

    Raises
    ------
    ValueError  : 文件格式不符（列数不足、数值解析失败等）
    """
    # 先读第一行获取样品名
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline().strip()

    sample_name = first_line if first_line else os.path.splitext(
        os.path.basename(file_path)
    )[0]

    # 跳过第一行读取数值矩阵
    data = np.loadtxt(file_path, skiprows=1)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"文件格式错误：期望至少 2 列数值，实际读取到 shape={data.shape}。\n"
            f"请确认文件第 1 行为样品名称，第 2 行起为 '2θ  强度' 格式。"
        )

    x_data = data[:, 0]
    y_data = data[:, 1]
    return x_data, y_data, sample_name


# ---------------------------------------------------------------------------
# （预留）Bruker / Rigaku RAW 读取
# ---------------------------------------------------------------------------

def load_raw_file(file_path: str) -> tuple:
    """
    读取 Bruker 或 Rigaku .raw 格式文件。（功能尚未实现）

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        ".raw 文件读取尚未实现。\n"
        "Bruker RAW v3/v4 为二进制格式，可参考 xylib 或 fabio 库进行解析。"
    )
