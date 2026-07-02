"""
io/file_reader.py
-----------------
XRD 数据文件统一读取接口。

支持格式
--------
  TXT  : 两列文本格式（第一行样品名称，后续为 2θ 与强度）
  RAW  : Bruker DIFFRAC 二进制格式（v3 RAW1.01 / v4 RAW4.00）
  RAW  : Rigaku ASCII 格式（.raw，以 "*" 开头的注释行）

公开接口
--------
  load_file(file_path)  → (x, y, sample_name, metadata)
    统一入口，自动识别格式，无需用户关心具体格式。

返回的 metadata 字典结构
-------------------------
  {
    "format":           str,          # "TXT" / "Bruker_RAW_v3" / "Bruker_RAW_v4" / "Rigaku_RAW"
    "sample_name":      str,
    "wavelength_Ka1":   float|None,   # nm（已从 Å 换算）
    "wavelength_Ka2":   float|None,
    "date":             str|None,
    "scan_mode":        str|None,
    "anode_material":   str|None,     # 如 "CU" "CO" "MO"
    "ranges": [                       # 各扫描段
        {"start": float, "step": float, "n_steps": int, "count_time": float|None}
    ]
  }
"""
import os
import struct
import numpy as np


# ===========================================================================
# 统一入口
# ===========================================================================

def load_file(file_path: str) -> tuple:
    """
    自动识别格式并读取 XRD 数据。

    Returns (x_data, y_data, sample_name, metadata)

    Raises ValueError 当格式无法识别或解析失败。
    """
    try:
        with open(file_path, "rb") as fh:
            magic = fh.read(8)
    except OSError as e:
        raise ValueError(f"无法打开文件: {e}")

    # Rigaku Ultima IV / RINT 二进制格式
    if magic[:4] == b"FI\x00\x00":
        return load_rigaku_ultima(file_path)

    # Bruker v1 ("RAW " + space, 老式格式)
    if magic[:4] == b"RAW ":
        return load_bruker_raw_v1(file_path)

    # Bruker v3
    if magic[:7] == b"RAW1.01":
        return load_bruker_raw_v3(file_path)

    # Bruker v4
    if magic[:7] == b"RAW4.00":
        return load_bruker_raw_v4(file_path)

    # Rigaku ASCII（以 '*' 开头，或 .raw 扩展名但非二进制）
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".raw" or magic[:1] == b"*":
        try:
            return load_rigaku_raw(file_path)
        except Exception:
            pass

    # 默认: TXT 两列格式
    x, y, name = load_txt_file(file_path)
    meta = _empty_meta("TXT", name)
    return x, y, name, meta


# ===========================================================================
# Rigaku Ultima IV / RINT 二进制格式  (magic = b"FI\x00\x00")
# ===========================================================================
#
# 仪器：Rigaku Ultima IV、D/Max 等系列
# 通过逆向工程自实际数据文件确认的字段位置：
#
#   [0:4]    magic = b"FI\x00\x00"
#   [104:128] char24 — 任务/样品名称（ASCII，可能为空）
#   [228:260] char32 — 仪器型号（ASCII）
#   [2958:2962] float32 LE — θ 起始角（度）；×2 得到 2θ 起始角
#   [2970:2974] float32 LE — 2θ 步长（度）
#   [9162:]  float32 × n — 强度数据
#
#   n_steps = (file_size - 9162) / 4
#   2θ_start = 2 × θ_start
#   2θ_array = 2θ_start + [0, 1, ..., n-1] × step
#
# ===========================================================================

_RIGAKU_HDR           = 9162
_RIGAKU_THETA_START   = 2958   # float32: theta start (°), × 2 = 2theta
_RIGAKU_STEP          = 2970   # float32: 2theta step (°)
_RIGAKU_SAMPLE_NAME   = 104    # char24: job / sample name
_RIGAKU_INSTRUMENT    = 228    # char32: instrument model


def load_rigaku_ultima(file_path: str) -> tuple:
    """
    解析 Rigaku Ultima IV / RINT 二进制格式 (magic = b'FI\x00\x00')。
    字段偏移通过对实测数据文件逆向工程确认。
    """
    import datetime as _dt, re as _re

    with open(file_path, "rb") as fh:
        raw = fh.read()

    total = len(raw)
    if total < _RIGAKU_HDR + 4:
        raise ValueError(f"文件过小（{total} bytes），无法解析为 Rigaku 二进制格式")

    n_steps = (total - _RIGAKU_HDR) // 4
    if n_steps < 2:
        raise ValueError("数据点数不足，文件可能损坏")

    # ── 扫描参数 ────────────────────────────────────────────────────────
    theta_start     = struct.unpack_from("<f", raw, _RIGAKU_THETA_START)[0]
    step            = struct.unpack_from("<f", raw, _RIGAKU_STEP)[0]
    two_theta_start = theta_start * 2.0
    two_theta_end   = round(two_theta_start + (n_steps - 1) * step, 4)

    if not (0.0 <= two_theta_start <= 170.0 and 1e-5 < step <= 5.0):
        raise ValueError(
            f"扫描参数异常（2θ_start={two_theta_start:.3f}°, step={step:.5f}°），"
            "可能不是 Rigaku Ultima 格式或文件损坏。"
        )

    # ── 元数据读取工具 ────────────────────────────────────────────────────
    def _rsc(off, L):
        """读字符串，去掉首个 null 字节及之后内容。"""
        s = _read_str(raw, off, L)
        s = _re.sub(r'\x00.*', '', s).strip()
        return s or None

    # ── 样品与文件信息 ────────────────────────────────────────────────────
    sample_name  = _rsc(104, 24) or _stem(file_path)  # 任务/样品名
    cond_file    = _rsc(204, 24)                        # 测量条件文件 (*.mcd)

    # ── 测量日期（off=36, Unix timestamp uint32）────────────────────────
    meas_date = None
    try:
        ts = struct.unpack_from("<I", raw, 36)[0]
        if 1_000_000_000 < ts < 2_000_000_000:
            meas_date = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    # ── 仪器信息 ──────────────────────────────────────────────────────────
    instrument = _rsc(228, 64)                          # 仪器型号
    try:
        rv = struct.unpack_from("<f", raw, 296)[0]
        instrument_radius = f"{rv:.0f} mm" if 50 < rv < 2000 else None
    except Exception:
        instrument_radius = None

    # 靶材（off=1238 "Cu"/"Co"/"Fe" 等2字节ASCII）
    anode = _rsc(1238, 4) or "Cu"
    # 波长（Cu Kα1=0.15406nm, Co Kα1=0.17890nm, Mo=0.07093nm）
    _ANODE_WAVE = {"Cu":(0.15406,0.15444),"Co":(0.17890,0.17929),
                   "Fe":(0.19360,0.19399),"Mo":(0.07093,0.07136)}
    lam1, lam2 = _ANODE_WAVE.get(anode, (0.15406, 0.15444))

    # ── 光学 / 狭缝 / 探测器 ──────────────────────────────────────────────
    optical_config = _rsc(308, 24)   # 光学配置 "Standard Att."
    slit_div       = _rsc(372, 16)   # 发散狭缝
    slit_incident  = _rsc(436, 16)   # 入射/水平发散狭缝
    slit_scatter   = _rsc(500, 16)   # 散射狭缝
    slit_receive   = _rsc(564, 16)   # 接收狭缝
    kbeta_filter   = _rsc(884, 20)   # Kβ滤片 / 单色器
    detector       = _rsc(1132, 16)  # 探测器
    scan_axis      = _rsc(2206, 8)   # 扫描轴 "1 deg." / "1/2deg"

    # ── DteX PSD 两段存储格式检测 ────────────────────────────────────────
    #
    # Rigaku DteX100/DteX250 系列 PSD 在某些模式下分两块存储数据：
    #   1. "帧数据"：off=3158 处的 n_extra 个 float32（低角度段）
    #   2. "扫描数据"：off=9162（标准区）的 n_main 个 float32
    #
    # 识别条件（三重验证）：
    #   A. detector 含 "DteX" 且不含 "Ultra"
    #   B. off=3154 的 uint32 = n_extra + n_main（合理总点数）
    #   C. PSD 公式算出的峰位 ≥ 25°（物理合理性）
    #
    # 2θ_start 计算规律（逆向工程确认）：
    #   DteX100 / '0.625'   → 2θ_start = off2958
    #   D/teX Ultra / '1 deg.' → 2θ_start = off2958 × 4
    #   DteX250(H) / 其他      → 2θ_start = off2958 × 2 + 1°
    #
    psd_frame_detected = False
    det_str = (detector or "").upper().replace(" ", "").replace("/", "")
    ax_str  = (scan_axis or "")

    if "DTEX" in det_str:
        try:
            n_total_cand = struct.unpack_from("<I", raw, 3154)[0]
            n_extra = n_total_cand - n_steps
            if 100 < n_extra < 5000 and (3158 + n_extra * 4 <= _RIGAKU_HDR):
                extra_c = list(struct.unpack_from(f"<{n_extra}f", raw, 3158))
                main_c  = list(struct.unpack_from(f"<{n_steps}f", raw, _RIGAKU_HDR))
                combined = extra_c + main_c

                # 计算 PSD 起始角
                if "ULTRA" in det_str and "1" in ax_str:
                    start_psd = theta_start * 4.0
                elif "DTEX100" in det_str or "0.625" in ax_str:
                    start_psd = theta_start
                else:
                    start_psd = theta_start * 2.0 + 1.0

                # 物理合理性校验
                peak_idx_psd = combined.index(max(combined))
                if start_psd + peak_idx_psd * step >= 25.0:
                    psd_frame_detected = True
                    n_steps         = len(combined)
                    counts          = combined
                    two_theta_start = start_psd
                    two_theta_end   = round(two_theta_start + (n_steps - 1) * step, 4)
        except Exception:
            pass

    # ── 强度数据（标准路径）─────────────────────────────────────────────
    if not psd_frame_detected:
        counts = list(struct.unpack_from(f"<{n_steps}f", raw, _RIGAKU_HDR))

    meta = {
        # ── 通用字段 ────────────────────────────────────────────────────
        "format":            "Rigaku_Ultima",
        "sample_name":       sample_name,
        "wavelength_Ka1":    lam1,
        "wavelength_Ka2":    lam2,
        "date":              meas_date,
        "scan_mode":         "2θ/θ",
        "anode_material":    anode,
        "ranges": [{
            "start":      round(two_theta_start, 4),
            "step":       round(step, 6),
            "n_steps":    n_steps,
            "count_time": None,
            "end":        two_theta_end,
        }],
        # ── Rigaku Ultima 专有字段 ───────────────────────────────────────
        "instrument":        instrument,
        "instrument_radius": instrument_radius,
        "condition_file":    cond_file,
        "optical_config":    optical_config,
        "slit_div":          slit_div,
        "slit_incident":     slit_incident,
        "slit_scatter":      slit_scatter,
        "slit_receive":      slit_receive,
        "kbeta_filter":      kbeta_filter,
        "detector":          detector,
        "scan_axis":         scan_axis,
        "instrument_id":     None,
        "psd_frame":         psd_frame_detected,   # True = DteX两段拼接格式
    }
    return (
        np.array([two_theta_start + i * step for i in range(n_steps)], dtype=float),
        np.array(counts, dtype=float),
        sample_name,
        meta,
    )


# ===========================================================================
# Bruker RAW v1  ("RAW " + space)
# ===========================================================================
#
# 这是最古老的 Bruker 二进制格式，见于早期 D5000、D8 等仪器。
# 头部仅 156 字节，结构最为简单：
#
#   [0:4]   magic = b"RAW "  (注意末尾是空格，不是"RAW1.01")
#   [4:8]   uint32 LE: n_steps（数据点总数）
#   [8:12]  float32 LE: 仪器最大角度上限（如96°），不是实际终止角！
#   [12:16] float32 LE: step_size（步长，°）
#   [24:28] float32 LE: 2θ 起始角（实测确认）
#   [28:32] float32 LE: θ  起始角（= off[24] / 2）
#   [40:44] char[4]:   仪器序列号（ASCII，非样品名）
#   [72:76] float32 LE: λKα1（Å）
#   [76:80] float32 LE: λKα2（Å）
#   [156:]  float32 × n_steps: 强度数据
#
# 2θ 起始角从 off[24] 读取；终止角推算为 start + (n-1)*step。
# off[8] 是仪器硬件上限（如96°），不代表实际扫描终止角，应忽略。
# 样品名未存储于文件内，使用文件名代替。
# ===========================================================================

_V1_FILE_HDR = 156

def load_bruker_raw_v1(file_path: str) -> tuple:
    """解析 Bruker RAW v1 (b'RAW ') 二进制格式。"""
    with open(file_path, "rb") as fh:
        raw = fh.read()

    if raw[:4] != b"RAW ":
        raise ValueError("魔术字节不符（期望 b'RAW '）")

    total = len(raw)
    if total < _V1_FILE_HDR + 4:
        raise ValueError("文件过小，无法解析（可能已损坏）")

    # ── 核心参数 ────────────────────────────────────────────────────────
    n_steps  = struct.unpack_from("<I", raw, 4)[0]
    end_2th  = struct.unpack_from("<f", raw, 8)[0]
    step     = struct.unpack_from("<f", raw, 12)[0]
    ka1_A    = struct.unpack_from("<f", raw, 72)[0]   # Å
    ka2_A    = struct.unpack_from("<f", raw, 76)[0]

    # 仪器 ID（序列号，4 位 ASCII）
    instrument_id = _read_str(raw, 40, 4)

    # ── 起始角：直接从 off[24] 读取 2θ_start ────────────────────────────
    # off[8] 是仪器硬件上限（如96°），不是实际终止角，忽略。
    # off[24] = 实际 2θ 起始角（SEC33/SEC111/boyuan1 三文件逆向确认）
    start_2th = struct.unpack_from("<f", raw, 24)[0]
    end_2th   = round(start_2th + (n_steps - 1) * step, 4)

    # 参数合理性校验
    if not _valid(start_2th, step, n_steps):
        raise ValueError(
            f"参数异常（start={start_2th:.3f}°, step={step:.5f}°, n={n_steps}），"
            "文件可能损坏或为不受支持的变体。"
        )

    # ── 读取强度数据 ──────────────────────────────────────────────────────
    data_end = _V1_FILE_HDR + n_steps * 4
    if data_end > total:
        raise ValueError(f"数据超出文件边界（期望 {data_end} 字节，实际 {total} 字节）")

    counts = list(struct.unpack_from(f"<{n_steps}f", raw, _V1_FILE_HDR))

    x = [start_2th + i * step for i in range(n_steps)]

    # ── 波长换算 Å → nm ───────────────────────────────────────────────────
    lam1 = _angstrom_to_nm(ka1_A)
    lam2 = _angstrom_to_nm(ka2_A)

    sample_name = _stem(file_path)

    import struct as _struct
    # off24: 当前 theta 起始位置（°），off28 = off24/2，off16 = flags
    theta_pos    = _struct.unpack_from("<f", raw, 24)[0]
    count_time_b = _struct.unpack_from("<f", raw, 28)[0]   # 可能是计数时间
    flags_u32    = _struct.unpack_from("<I", raw, 16)[0]

    meta = {
        "format":            "Bruker_RAW_v1",
        "sample_name":       sample_name,
        "wavelength_Ka1":    lam1,
        "wavelength_Ka2":    lam2,
        "date":              None,
        "scan_mode":         "2θ/θ",
        "anode_material":    "Cu" if (lam1 and 0.150 < lam1 < 0.156) else None,
        "instrument_id":     instrument_id or None,
        "instrument":        None,
        "instrument_radius": None,
        "condition_file":    None,
        "optical_config":    None,
        "slit_div":          None,
        "slit_incident":     None,
        "slit_scatter":      None,
        "slit_receive":      None,
        "kbeta_filter":      None,
        "detector":          None,
        "scan_axis":         None,
        "ranges": [{
            "start":      round(start_2th, 4),
            "step":       round(step, 6),
            "n_steps":    n_steps,
            "count_time": None,
            "end":        round(start_2th + (n_steps - 1) * step, 4),
        }],
    }
    return (
        np.array(x, dtype=float),
        np.array(counts, dtype=float),
        sample_name,
        meta,
    )


# ===========================================================================
# TXT
# ===========================================================================

def load_txt_file(file_path: str) -> tuple:
    """
    读取两列 TXT 格式。
    第 1 行为样品名称，第 2 行起为 '2θ  强度' 数值对。
    Returns (x_data, y_data, sample_name)
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline().strip()

    sample_name = first_line if first_line else _stem(file_path)
    data = np.loadtxt(file_path, skiprows=1)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"文件格式错误：期望至少 2 列数值，实际 shape={data.shape}。\n"
            "请确认第 1 行为样品名称，第 2 行起为 '2θ  强度' 格式。"
        )
    return data[:, 0], data[:, 1], sample_name


# ===========================================================================
# Bruker RAW v3  (RAW1.01)
# ===========================================================================
#
# 文件结构（小端序）:
#   [0:712]   文件头（712 bytes）
#     [0:7]   magic b'RAW1.01'
#     [7]     保留
#     [156:160] uint32 — 扫描段数（某些版本）
#     [326:386] char60 — 样品名称
#     [608:612] char4  — 阳极元素 (e.g. b'CU  ')
#     [612:620] float64 — λKα1 (Å)
#     [620:628] float64 — λKα2 (Å)
#     [390:410] char20 — 测量日期
#   每段 = 段头(304 bytes) + n_steps×float32
#   段头:
#     [0:4]   uint32  — n_steps
#     [4:8]   float32 — count_time (s/step)
#     [8:16]  float64 — start_2theta (°)
#     [16:24] float64 — step_size (°)
# ===========================================================================

_V3_FILE_HDR  = 712
_V3_RANGE_HDR = 304


def load_bruker_raw_v3(file_path: str) -> tuple:
    """
    解析 Bruker RAW v3 (RAW1.01) 格式，兼容 PowDLL 导出的 SAG-free 变体。

    标准 Bruker v3 段头字段
    -----------------------
    seg[0:4]   uint32  — 段头大小 (=304, 不是 n_steps!)
    seg[4:8]   uint32  — n_steps
    seg[8:16]  float64 — start_2theta (°)
    seg[16:24] float64 — step (°)

    PowDLL SAG-free 变体
    --------------------
    同上偏移，但：
    seg[8:16]  float64 — theta_start (°) → 需 ×2 得到 2θ_start
    seg[16:24] float64 — step 单位为 millidegrees → 需 ÷1000 得到度
    """
    import datetime as _dt

    with open(file_path, "rb") as fh:
        raw = fh.read()

    if raw[:7] != b"RAW1.01":
        raise ValueError("魔术字节不符（期望 RAW1.01）")

    total = len(raw)

    # ── 文件头元数据 ─────────────────────────────────────────────────────
    # 样品名（标准位置，PowDLL文件通常为空）
    sample_name = _try_str(raw, [326, 368, 72], 60) or _stem(file_path)

    # 日期 + 时间（PowDLL文件头特有字段）
    date_str = _read_str(raw, 16, 10)   # "03/28/25" 或 "MM/DD/YY"
    time_str = _read_str(raw, 26, 10)   # "02:04:18"
    operator = _read_str(raw, 36, 28)   # 操作员/用户名

    # 整合日期时间
    meas_date = None
    if date_str:
        # 格式 "MM/DD/YY" → 转成 YYYY-MM-DD HH:MM:SS
        try:
            dt_obj = _dt.datetime.strptime(date_str.strip(), "%m/%d/%y")
            meas_date = dt_obj.strftime("%Y-%m-%d")
            if time_str:
                meas_date += f" {time_str.strip()}"
        except ValueError:
            meas_date = f"{date_str} {time_str}".strip()
    # 标准 v3 的日期字段
    if not meas_date:
        meas_date = _read_str(raw, 390, 20) or None

    # 靶材 + 波长（标准 v3 位置）
    anode = _read_str(raw, 608, 4) or None
    lam1  = _angstrom_to_nm(_read_f64(raw, 624))   # 实测位置 624/632
    lam2  = _angstrom_to_nm(_read_f64(raw, 632))
    if lam1 is None:                                 # 备用标准位置
        lam1 = _angstrom_to_nm(_read_f64(raw, 612))
        lam2 = _angstrom_to_nm(_read_f64(raw, 620))

    # ── 段数量 ──────────────────────────────────────────────────────────
    n_ranges = _v3_n_ranges(raw, total)

    # ── 解析各段 ────────────────────────────────────────────────────────
    offset = _V3_FILE_HDR
    all_x, all_y, ranges_meta = [], [], []
    raw_variants = set()
    supplemental_points_total = 0
    last_data_end = offset

    for i in range(n_ranges):
        if offset + _V3_RANGE_HDR > total:
            break

        # seg[0:4] = 段头大小(304)，seg[4:8] = n_steps（注意：不是 seg[0]）
        range_header_size = struct.unpack_from("<I", raw, offset)[0]
        if not (24 <= range_header_size <= 4096 and offset + range_header_size <= total):
            range_header_size = _V3_RANGE_HDR
        n_steps    = struct.unpack_from("<I", raw, offset + 4)[0]
        count_time = struct.unpack_from("<f", raw, offset + 4 + 4)[0]  # offset+8 as f32
        start      = struct.unpack_from("<d", raw, offset + 8)[0]
        step       = struct.unpack_from("<d", raw, offset + 16)[0]
        raw_start  = start
        raw_step   = step
        raw_variant = "standard_2theta_degree"

        # ── 自动检测 PowDLL SAG-free 变体 ──────────────────────────────
        # 特征：step > 1°（实际是 millidegrees），start < 30°（实际是 theta）
        if step > 1.0 and 0.001 < step / 1000.0 < 1.0:
            step  = step / 1000.0      # millidegrees → degrees
            start = start * 2.0        # theta → 2θ
            raw_variant = "theta_mdeg"

        # 参数合理性校验；失败时尝试备用偏移
        if not _valid(start, step, n_steps):
            start, step, n_steps, count_time = _v3_alt_offsets(raw, offset)
            raw_start = start
            raw_step = step
            raw_variant = "alternate_offsets"
            if not _valid(start, step, n_steps):
                raise ValueError(
                    f"第 {i+1} 段参数异常 "
                    f"(start={start:.3f}°, step={step:.5f}°, n={n_steps})。\n"
                    "文件可能损坏，或为不受支持的仪器子版本。"
                )

        data_start = offset + range_header_size
        data_end   = data_start + n_steps * 4
        if data_end > total:
            raise ValueError(f"第 {i+1} 段数据超出文件边界（文件可能被截断）")

        counts = list(struct.unpack_from(f"<{n_steps}f", raw, data_start))
        supplemental_unit_size = struct.unpack_from("<I", raw, offset + 252)[0]
        supplemental_bytes = struct.unpack_from("<I", raw, offset + 256)[0]
        supplemental_counts = []
        if (
            supplemental_unit_size == 4
            and supplemental_bytes > 0
            and supplemental_bytes % 4 == 0
            and supplemental_bytes <= 4096
            and data_end + supplemental_bytes <= total
        ):
            extra_n = supplemental_bytes // 4
            candidate = list(struct.unpack_from(f"<{extra_n}f", raw, data_end))
            if all(np.isfinite(v) and v >= 0 for v in candidate):
                supplemental_counts = candidate

        if supplemental_counts:
            counts.extend(supplemental_counts)
            data_end += len(supplemental_counts) * 4
            supplemental_points_total += len(supplemental_counts)

        total_steps = len(counts)
        all_x.extend(start + j * step for j in range(total_steps))
        all_y.extend(counts)
        range_meta = _range_dict(start, step, total_steps, count_time)
        range_meta.update({
            "declared_n_steps": int(n_steps),
            "supplemental_points": int(len(supplemental_counts)),
            "supplemental_unit_size": int(supplemental_unit_size),
            "supplemental_bytes": int(supplemental_bytes if supplemental_counts else 0),
            "raw_variant": raw_variant,
            "range_header_size": int(range_header_size),
            "raw_start_field": float(raw_start),
            "raw_step_field": float(raw_step),
            "data_offset": int(data_start),
        })
        ranges_meta.append(range_meta)
        raw_variants.add(raw_variant)
        last_data_end = data_end
        offset = data_end

    if not all_x:
        raise ValueError("文件中未找到有效数据点")

    meta = {
        "format":            "Bruker_RAW_v3",
        "sample_name":       sample_name,
        "wavelength_Ka1":    lam1,
        "wavelength_Ka2":    lam2,
        "date":              meas_date,
        "scan_mode":         "2θ/θ",
        "anode_material":    anode,
        "operator":          operator or None,
        "instrument":        None,
        "instrument_radius": None,
        "condition_file":    None,
        "optical_config":    None,
        "slit_div":          None,
        "slit_incident":     None,
        "slit_scatter":      None,
        "slit_receive":      None,
        "kbeta_filter":      None,
        "detector":          None,
        "scan_axis":         None,
        "instrument_id":     None,
        "raw_variant":       "+".join(sorted(raw_variants)) if raw_variants else None,
        "supplemental_points": int(supplemental_points_total),
        "trailing_bytes":    max(0, total - last_data_end),
        "ranges":            ranges_meta,
    }
    return np.array(all_x), np.array(all_y), sample_name, meta


def _v3_n_ranges(raw: bytes, total: int) -> int:
    for off, size in [(156, 4), (8, 4), (7, 1)]:
        if off + size <= _V3_FILE_HDR:
            val = raw[off] if size == 1 else struct.unpack_from("<I", raw, off)[0]
            if 1 <= val <= 50:
                return int(val)
    # 按文件大小估算
    return min(max(1, (total - _V3_FILE_HDR) // (_V3_RANGE_HDR + 4000)), 10)


def _v3_alt_offsets(raw: bytes, base: int) -> tuple:
    """备用字段偏移（少数仪器子版本）。"""
    try:
        n = struct.unpack_from("<I", raw, base)[0]
        s = struct.unpack_from("<d", raw, base + 12)[0]
        w = struct.unpack_from("<d", raw, base + 24)[0]
        ct = struct.unpack_from("<f", raw, base + 4)[0]
        return s, w, n, ct
    except Exception:
        return 0.0, 0.0, 0, 0.0


# ===========================================================================
# Bruker RAW v4  (RAW4.00)
# ===========================================================================
#
# 块结构: [4 bytes type][4 bytes length][content]
# 关键块: b'FILE' — 样品信息，b'HEAD' — 仪器信息，b'MEAS' — 测量数据
# ===========================================================================

def load_bruker_raw_v4(file_path: str) -> tuple:
    with open(file_path, "rb") as fh:
        raw = fh.read()

    if raw[:7] != b"RAW4.00":
        raise ValueError("魔术字节不符（期望 RAW4.00）")

    sample_name = _stem(file_path)
    date = anode = lam1 = lam2 = None
    all_x, all_y, ranges_meta = [], [], []

    offset = 8
    total  = len(raw)

    while offset + 8 <= total:
        btype = raw[offset:offset + 4]
        blen  = struct.unpack_from("<I", raw, offset + 4)[0]
        cs    = offset + 8
        ce    = cs + blen
        if ce > total:
            break
        c = raw[cs:ce]

        if btype == b"FILE":
            sample_name = _read_str(c, 0, 72) or sample_name
            date        = _read_str(c, 72, 20) or None

        elif btype == b"HEAD":
            anode  = _read_str(c, 0, 4) or None
            lam1   = _angstrom_to_nm(_read_f64(c, 4))
            lam2   = _angstrom_to_nm(_read_f64(c, 12))

        elif btype == b"MEAS":
            try:
                n   = struct.unpack_from("<I", c, 0)[0]
                ct  = struct.unpack_from("<f", c, 4)[0]
                s   = struct.unpack_from("<d", c, 8)[0]
                w   = struct.unpack_from("<d", c, 16)[0]
                if _valid(s, w, n):
                    counts = list(struct.unpack_from(f"<{n}f", c, 24))
                    all_x.extend(s + j * w for j in range(n))
                    all_y.extend(counts)
                    ranges_meta.append(_range_dict(s, w, n, ct))
            except struct.error:
                pass

        offset = ce

    if not all_x:
        raise ValueError("Bruker RAW v4 文件中未找到有效测量数据（MEAS 块）")

    meta = {
        "format":         "Bruker_RAW_v4",
        "sample_name":    sample_name,
        "wavelength_Ka1": lam1,
        "wavelength_Ka2": lam2,
        "date":           date,
        "scan_mode":      None,
        "anode_material": anode,
        "ranges":         ranges_meta,
    }
    return np.array(all_x), np.array(all_y), sample_name, meta


# ===========================================================================
# Rigaku RAW  (ASCII)
# ===========================================================================

def load_rigaku_raw(file_path: str) -> tuple:
    """解析 Rigaku ASCII .raw 格式（注释行以 '*' 开头）。"""
    sample_name = _stem(file_path)
    date = anode = lam1 = None
    start_angle = step_size = None
    x_vals, y_vals = [], []

    with open(file_path, "r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("*"):
                content = line[1:].strip()
                for sep in ("=", " "):
                    if sep in content:
                        k, _, v = content.partition(sep)
                        k, v = k.strip().upper(), v.strip()
                        if k in ("SAMPLE", "SAMPLENAME", "NAME"):
                            sample_name = v or sample_name
                        elif k in ("DATE", "MEAS_DATE"):
                            date = v
                        elif k in ("TARGET", "ANODE", "TUBE"):
                            anode = v
                        elif k in ("WAVE1", "LAMBDA", "WAVELENGTH"):
                            try:
                                f = float(v)
                                lam1 = f / 10.0 if f > 0.1 else f  # Å→nm
                            except ValueError:
                                pass
                        elif k in ("2THETASTART", "START"):
                            try: start_angle = float(v)
                            except ValueError: pass
                        elif k in ("STEP", "STEPSIZE", "SCAN_STEP"):
                            try: step_size = float(v)
                            except ValueError: pass
                        break
                continue

            if not line.strip():
                continue

            parts = line.split()
            try:
                if len(parts) >= 2:
                    x_vals.append(float(parts[0]))
                    y_vals.append(float(parts[1]))
                elif len(parts) == 1 and step_size and start_angle is not None:
                    x_vals.append(start_angle + len(y_vals) * step_size)
                    y_vals.append(float(parts[0]))
            except ValueError:
                continue

    if len(x_vals) < 5:
        raise ValueError("Rigaku RAW 文件有效数据点不足（<5），请检查格式。")

    ranges_meta = [{
        "start":      round(x_vals[0], 4),
        "step":       round(x_vals[1] - x_vals[0], 6) if len(x_vals) > 1 else None,
        "n_steps":    len(x_vals),
        "count_time": None,
    }]
    meta = {
        "format":         "Rigaku_RAW",
        "sample_name":    sample_name,
        "wavelength_Ka1": lam1,
        "wavelength_Ka2": None,
        "date":           date,
        "scan_mode":      None,
        "anode_material": anode,
        "ranges":         ranges_meta,
    }
    return np.array(x_vals), np.array(y_vals), sample_name, meta


# ===========================================================================
# 内部工具
# ===========================================================================

def _stem(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def _read_str(data: bytes, offset: int, length: int) -> str:
    try:
        return data[offset:offset + length].decode("latin-1").rstrip("\x00").strip()
    except Exception:
        return ""


def _try_str(data: bytes, offsets: list, length: int) -> str:
    """按偏移列表依次尝试，返回第一个非空、可打印的字符串。"""
    for off in offsets:
        s = _read_str(data, off, length)
        if s and len(s) >= 2 and s.isprintable():
            return s
    return ""


def _read_f64(data: bytes, offset: int):
    try:
        v = struct.unpack_from("<d", data, offset)[0]
        return v if (v == v) and abs(v) < 1e15 else None  # NaN/Inf 过滤
    except Exception:
        return None


def _angstrom_to_nm(val):
    """将 Å 转换为 nm；非物理值返回 None。"""
    if val is None:
        return None
    return val / 10.0 if 0.05 < val < 3.0 else None


def _valid(start: float, step: float, n: int) -> bool:
    return (0.0 <= start <= 175.0) and (1e-5 < step <= 5.0) and (1 <= n <= 200_000)


def _range_dict(start, step, n_steps, count_time) -> dict:
    return {
        "start":      round(float(start), 4),
        "step":       round(float(step), 6),
        "n_steps":    int(n_steps),
        "count_time": round(float(count_time), 3) if count_time else None,
    }


def _empty_meta(fmt: str, sample_name: str) -> dict:
    return {
        "format":         fmt,
        "sample_name":    sample_name,
        "wavelength_Ka1": None,
        "wavelength_Ka2": None,
        "date":           None,
        "scan_mode":      None,
        "anode_material": None,
        "ranges":         [],
    }
