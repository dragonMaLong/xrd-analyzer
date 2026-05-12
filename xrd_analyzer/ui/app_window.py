"""
ui/app_window.py
-----------------
XRDApp 主类：
  - 继承三个 Mixin，统一协调左侧面板、右侧图表和 L-Curve 功能
  - 管理数据加载、计算线程、结果后处理和 CSV 导出
"""
import os
import csv
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from ..core.peak_functions import precompile_numba_functions
from ..core.fitting import (
    WAVELENGTHS,
    INTENSITY_RATIO,
    fit_with_mu_list,
    build_regularization_matrix,
    _eval_candidate_for_index,      # 子进程函数，必须在顶层可导入
)
from ..core.analysis import build_all_peak_info
from ..io.file_reader import load_txt_file
from ..utils import resource_path

from .control_panel_mixin import ControlPanelMixin
from .plot_panel_mixin import PlotPanelMixin
from .l_curve_mixin import LCurveMixin


class XRDApp(ControlPanelMixin, PlotPanelMixin, LCurveMixin):
    """
    XRD 多峰拟合分析工具主窗口。

    继承关系
    --------
    ControlPanelMixin — 左侧控制面板
    PlotPanelMixin    — 右侧三联图
    LCurveMixin       — L-Curve 正则化参数分析
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("XRD多峰拟合分析工具")

        # ── 状态变量 ─────────────────────────────────────────────────
        self.data_loaded      = False
        self.results_ready    = False
        self.stop_flag        = threading.Event()
        self.dragging_slider  = None
        self.dragging_peak_index = None
        self.D_STEP           = 0.1

        # ── 交互 Artist 引用 ─────────────────────────────────────────
        self.line_min = None
        self.line_max = None

        # ── 多峰状态 ──────────────────────────────────────────────────
        self.max_peaks    = 5
        self.peak_colors  = ["#FF00FF", "#0077FF", "#00C853", "#FFAB00", "#00E5FF"]
        self.active_peak_indices   = []
        self.peak_mu_sliders       = []
        self.peak_check_vars       = []
        self.peak_mu_rects_preview = []
        self.peak_mu_rects_axes0   = []

        # ── 数据占位符 ────────────────────────────────────────────────
        self.source_var = None       # 在 _build_scrollable_controls 中创建

        try:
            root.iconbitmap(resource_path("logo.ico"))
        except Exception:
            pass

        self._setup_ui()
        precompile_numba_functions()

    # ------------------------------------------------------------------
    # UI 构建（串联各 Mixin）
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """构建完整的窗口布局：左侧面板 + 右侧图表。"""
        self.root.minsize(800, 600)
        self.root.geometry("2000x1400")
        self.root.update_idletasks()

        # ── 左侧控制面板 ──────────────────────────────────────────────
        self.LEFT_COL_W = 240
        self.left_frame = tk.Frame(self.root, bg="#F0F0F0", width=self.LEFT_COL_W)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)
        self.left_frame.pack_propagate(False)

        # 滚动条样式
        sbar_style = ttk.Style()
        sbar_style.theme_use("clam")
        sbar_style.configure(
            "LeftPane.Vertical.TScrollbar",
            width=20, background="#C0C0C0", troughcolor="#E0E0E0",
            bordercolor="#F2FEFE", arrowcolor="#666666",
            darkcolor="#A0A0A0", lightcolor="#F0F0F0",
        )

        button_style = {
            "font": ("微软雅黑", 11, "bold"),
            "fg": "white", "relief": "flat", "pady": 6,
        }

        # 按钮区（不可滚动）
        self.btn_zone = tk.Frame(self.left_frame, width=self.LEFT_COL_W, bg="#F0F0F0")
        self.btn_zone.pack(anchor="n", pady=(0, 6))

        self.btn_import = tk.Button(
            self.btn_zone, text="📂 导入TXT文件", bg="#8B8B8B",
            command=self.load_txt, **button_style,
        )
        self.btn_import.pack(fill=tk.X, pady=8)

        self.btn_fast = tk.Button(
            self.btn_zone, text="⚡ 极速计算", bg="#5100FF",
            command=lambda: self.compute_thread(mode="fast"), **button_style,
        )
        self.btn_fast.pack(fill=tk.X, pady=8)

        self.btn_fine = tk.Button(
            self.btn_zone, text="🔍 精细计算", bg="#FF9932",
            command=lambda: self.compute_thread(mode="fine"), **button_style,
        )
        self.btn_fine.pack(fill=tk.X, pady=8)

        self.btn_save = tk.Button(
            self.btn_zone, text="💾 保存结果", bg="#69BB66",
            command=self.save_results, **button_style,
        )
        self.btn_save.pack(fill=tk.X, pady=8)

        self.btn_stop = tk.Button(
            self.btn_zone, text="⏹ 停止计算", bg="#CA4F4F",
            command=self.stop_compute, **button_style,
        )
        self.btn_stop.pack(fill=tk.X, pady=8)

        self.btn_lcurve = tk.Button(
            self.btn_zone, text="📈 L-Curve 分析", bg="#00838F",
            command=self.run_l_curve_thread, **button_style,
        )
        self.btn_lcurve.pack(fill=tk.X, pady=8)

        ttk.Separator(self.btn_zone, orient="horizontal").pack(fill="x", pady=10)

        # 可滚动控件区
        self.scroll_zone = tk.Frame(self.left_frame, width=self.LEFT_COL_W, bg="#F0F0F0")
        self.scroll_zone.pack(fill=tk.BOTH, expand=True)
        self.scroll_zone.pack_propagate(False)
        self._build_scrollable_controls()   # ← ControlPanelMixin

        # ── 右侧图表区 ────────────────────────────────────────────────
        self.setup_plots()                  # ← PlotPanelMixin
        self.bind_events()                  # ← PlotPanelMixin

    # ------------------------------------------------------------------
    # 线程安全 UI 调度帮助函数
    # ------------------------------------------------------------------

    def ui(self, fn, *args, **kwargs):
        """在主线程中安全地调度 Tkinter 操作。"""
        self.root.after(0, lambda: fn(*args, **kwargs))

    def ui_set(self, var, value):
        """在主线程中安全地设置 StringVar / IntVar 等。"""
        self.root.after(0, lambda: var.set(value))

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_txt(self):
        """打开文件对话框，读取两列 TXT 格式 XRD 数据。"""
        file_path = filedialog.askopenfilename(filetypes=[("TXT files", "*.txt")])
        if not file_path:
            return
        try:
            x, y, name = load_txt_file(file_path)
            self.x_data            = x
            self.y_data            = y
            self.data_loaded       = True
            self.current_file_name = name
            self.update_preview(None)
        except Exception as exc:
            messagebox.showwarning("文件提取错误", f"无法加载文件: {exc}")

    # ------------------------------------------------------------------
    # 计算线程
    # ------------------------------------------------------------------

    def compute_thread(self, mode: str = "fine"):
        """在后台线程中启动拟合计算，避免阻塞 UI。"""
        self.stop_flag.clear()

        # 显示进度条
        self.progress_label.place(relx=0.62, rely=0.01, relwidth=0.10)
        self.progress_bar.place(relx=0.73, rely=0.01, relwidth=0.24, height=20)
        self.ui_set(self.progress_var, "计算中...")
        self.progress_bar["maximum"] = 100
        self.progress_bar["value"]   = 0

        # 禁用按钮防止重入
        for btn in (self.btn_fast, self.btn_fine):
            try:
                btn.config(state=tk.DISABLED)
            except Exception:
                pass

        threading.Thread(
            target=self.compute_fit, args=(mode,), daemon=True
        ).start()

    def compute_fit(self, mode: str = "fine"):
        """执行多峰拟合（子线程）。"""
        try:
            # ── 1. 读取参数 ──────────────────────────────────────────
            source      = self.source_var.get()
            lam1, lam2  = WAVELENGTHS.get(source, WAVELENGTHS["Cu"])
            mu_centers  = [s.get() for s in self.peak_mu_sliders]

            # ── 2. 截取并预处理数据 ──────────────────────────────────
            mask  = (
                (self.x_data >= self.slider_min.get())
                & (self.x_data <= self.slider_max.get())
            )
            x     = self.x_data[mask]
            y_raw = self.y_data[mask]

            bg_indices = np.where(
                (x < self.slider_min.get() + 0.5)
                | (x > self.slider_max.get() - 0.5)
            )[0]
            if len(bg_indices) < 2:
                bg_indices = np.array([0, len(x) - 1])
            slope_bg, intercept_bg = np.polyfit(x[bg_indices], y_raw[bg_indices], 1)
            background = slope_bg * x + intercept_bg
            y = y_raw - background
            y[y < 0] = 0

            if y.max() <= 0:
                self.ui_set(self.progress_var, "错误：无有效信号")
                return
            y_scaled = y / y.max()

            # ── 3. 粒径网格 + 正则化矩阵 ────────────────────────────
            d_min, d_max = self.slider_d_min.get(), self.slider_d_max.get()
            raw_pts  = int((d_max - d_min) / 0.1)
            num_pts  = min(800, max(200, raw_pts))
            self.D_range = np.linspace(d_min, d_max, num_pts)
            L_single = build_regularization_matrix(len(self.D_range))

            alpha_val = float(self.slider_alpha.get())

            # ── 4. 峰位扫描（逐峰并行）──────────────────────────────
            if mode == "fast":
                halfwidth, steps = 0.0, 1
            else:
                halfwidth, steps = 0.1, 11

            total = max(1, len(mu_centers) * steps)
            done  = 0

            best_mu = list(mu_centers)
            for i in range(len(best_mu)):
                if self.stop_flag.is_set():
                    self.ui_set(self.progress_var, "已停止")
                    return

                center = best_mu[i]
                low    = max(center - halfwidth, self.slider_min.get())
                high   = min(center + halfwidth, self.slider_max.get())
                if high <= low:
                    low, high = center - 1e-4, center + 1e-4

                candidates  = np.linspace(low, high, steps)
                best_loss   = None
                best_val    = center
                max_workers = min(4, os.cpu_count() or 1)

                args_common = (
                    tuple(best_mu), i, x, y_scaled, lam1, lam2,
                    INTENSITY_RATIO, L_single, self.D_range, alpha_val,
                )

                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = [
                        ex.submit(_eval_candidate_for_index, mu, *args_common)
                        for mu in candidates
                    ]
                    for fut in as_completed(futs):
                        if self.stop_flag.is_set():
                            self.ui_set(self.progress_var, "已停止")
                            break
                        try:
                            loss, mu_val = fut.result()
                        except Exception:
                            loss, mu_val = np.inf, None

                        done += 1
                        pct  = int(done * 100 / total)
                        # ← 通过 after() 在主线程更新进度条（线程安全）
                        self.ui(lambda p=pct: setattr(self.progress_bar, "value", p)
                                or self.progress_bar.config(value=p))
                        self.ui_set(
                            self.progress_var,
                            f"扫描峰 {i + 1}/{len(best_mu)}… {pct}%",
                        )

                        if mu_val is not None and (
                            best_loss is None or loss < best_loss
                        ):
                            best_loss, best_val = loss, mu_val

                best_mu[i] = best_val

            if self.stop_flag.is_set():
                self.ui_set(self.progress_var, "已停止")
                return

            # ── 5. 最终一次完整拟合 ──────────────────────────────────
            resid, f_total, basis_k1_list, basis_k2_list = fit_with_mu_list(
                x, y_scaled, best_mu, lam1, lam2, L_single,
                self.D_range, alpha_val,
            )
            if f_total is None:
                self.ui_set(self.progress_var, "拟合失败：解全为零")
                return

            self.best_f_total    = f_total
            self.all_basis_k1    = basis_k1_list
            self.all_basis_k2    = basis_k2_list

            # 同步峰位滑块到最优值
            for s, mu in zip(self.peak_mu_sliders, best_mu):
                self.ui(s.set, mu)

            self._hide_axes0_overlays()

            # ── 6. 保存中间数据并更新图表 ────────────────────────────
            self.x_segment      = x
            self.y_segment_raw  = y_raw
            self.y_segment      = y
            self.background     = background

            self.process_multi_peak_results()
            self.ui_set(self.progress_var, "拟合成功！")

        except Exception as exc:
            self.ui(messagebox.showwarning, "提示", f"计算过程中发生错误: {exc}")
            self.ui_set(self.progress_var, "计算失败")
        finally:
            for btn in (self.btn_fast, self.btn_fine):
                try:
                    self.ui(btn.config, state=tk.NORMAL)
                except Exception:
                    pass

    def stop_compute(self):
        """中断正在运行的计算。"""
        self.stop_flag.set()
        self.ui_set(self.progress_var, "正在停止...")

    # ------------------------------------------------------------------
    # 结果后处理
    # ------------------------------------------------------------------

    def process_multi_peak_results(self):
        """调用 core/analysis 后处理 NNLS 结果，然后更新图表。"""
        self.all_peak_info, self.global_max_component_area = build_all_peak_info(
            self.best_f_total,
            self.active_peak_indices,
            self.D_range,
            self.peak_colors,
            self.all_basis_k1,
            self.all_basis_k2,
        )
        self.results_ready = True
        self.update_multi_peak_plots()          # ← PlotPanelMixin
        self.ui_set(self.progress_var, "拟合成功！")

    # ------------------------------------------------------------------
    # CSV 导出
    # ------------------------------------------------------------------

    def save_results(self):
        """将粒径分布数据和 XRD 拟合曲线导出为 CSV。"""
        if not self.results_ready:
            messagebox.showwarning("提示", "请先完成计算再保存。")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")],
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)

                # ── 0. 总面积（归一化基准）───────────────────────────
                A_total_sum = max(
                    sum(
                        float(det.get("area", 0.0))
                        for info in self.all_peak_info
                        for det in info.get("peak_details", [])
                    ),
                    1e-12,
                )

                # ── 1. 全局总分布 ─────────────────────────────────────
                global_f_sum = sum(
                    np.asarray(info["f_segment"], dtype=float)
                    for info in self.all_peak_info
                )
                global_total_Y = global_f_sum / A_total_sum

                # ── 2. 各分峰分布（体积密度）─────────────────────────
                scaled_curves = [
                    (self.active_peak_indices[i] + 1,
                     np.asarray(info["f_segment"], dtype=float) / A_total_sum)
                    for i, info in enumerate(self.all_peak_info)
                ]

                # ── 3. 表头 ───────────────────────────────────────────
                dist_header = ["Global_Total_D(nm)", "Global_Total_Y(PDF)", ""]
                for peak_id, _ in scaled_curves:
                    dist_header += [f"Peak{peak_id}_D(nm)", f"Peak{peak_id}_Y(PDF)", ""]

                # ── 4. XRD 拟合数据 ───────────────────────────────────
                x      = self.x_segment
                bg     = self.background
                y_corr = getattr(self, "y_corr", None)
                y      = self.y_segment if y_corr is None else y_corr
                y_raw  = getattr(self, "y_segment_raw", y + bg)

                total_fit = np.zeros_like(x)
                peak_fits = []
                comp_fits_by_peak = []

                for info in self.all_peak_info:
                    f_seg      = info["f_segment"]
                    basis_k1   = info["basis_k1"]
                    basis_k2   = info["basis_k2"]
                    fit_peak   = (basis_k1.dot(f_seg) + basis_k2.dot(f_seg)) * y.max()
                    peak_fits.append(fit_peak)
                    total_fit += fit_peak

                    comps = []
                    for det in info["peak_details"]:
                        idx = det.get("indices", None)
                        if idx is None or len(idx) == 0:
                            comps.append(np.full_like(x, np.nan, dtype=float))
                            continue
                        f_comp = np.zeros_like(f_seg)
                        f_comp[idx] = f_seg[idx]
                        comps.append(
                            (basis_k1[:, idx].dot(f_comp[idx])
                             + basis_k2[:, idx].dot(f_comp[idx])) * y.max()
                        )
                    comp_fits_by_peak.append(comps)

                total_fit_out  = total_fit + bg
                peak_fits_out  = [pf + bg for pf in peak_fits]
                comp_fits_out  = []
                comp_headers   = []
                for i, comps in enumerate(comp_fits_by_peak):
                    peak_id = self.active_peak_indices[i] + 1
                    for j, det in enumerate(self.all_peak_info[i]["peak_details"]):
                        comp_headers.append(f"P{peak_id}_Comp{j+1}@{det['center']:.2f}nm")
                        c = comps[j]
                        comp_fits_out.append(
                            np.full_like(x, np.nan, dtype=float)
                            if np.isnan(c).all() else c + bg
                        )

                left_header = ["2θ (deg)", "Raw Data", "Background", "Total Fit"]
                left_header += [
                    f"Peak_{self.active_peak_indices[i]+1}_Contribution"
                    for i in range(len(self.all_peak_info))
                ]
                left_header += comp_headers

                # ── 5. 写入 CSV ───────────────────────────────────────
                writer.writerow(dist_header + [""] + left_header)

                n_dist = len(self.D_range)
                n_left = len(x)
                n_rows = max(n_dist, n_left)

                # 分布列数据
                dist_cols = [
                    ([f"{d:.4f}" for d in self.D_range],
                     [f"{v:.6f}" for v in global_total_Y])
                ]
                for _, curve in scaled_curves:
                    dist_cols.append((
                        [f"{d:.4f}" for d in self.D_range],
                        [f"{v:.6f}" for v in curve],
                    ))

                # XRD 列数据
                left_cols = [
                    [f"{v:.4f}" for v in x],
                    [f"{v:.2f}"  for v in y_raw],
                    [f"{v:.2f}"  for v in bg],
                    [f"{v:.2f}"  for v in total_fit_out],
                ]
                for pf in peak_fits_out:
                    left_cols.append([f"{v:.2f}" for v in pf])
                for cf in comp_fits_out:
                    left_cols.append(
                        ["" if np.isnan(v) else f"{v:.2f}" for v in cf]
                    )

                for r in range(n_rows):
                    row = []
                    for D_col, Y_col in dist_cols:
                        row.append(D_col[r] if r < n_dist else "")
                        row.append(Y_col[r] if r < n_dist else "")
                        row.append("")
                    row.append("")
                    for col in left_cols:
                        row.append(col[r] if r < n_left else "")
                    writer.writerow(row)

            messagebox.showinfo("成功", "结果已成功保存！")

        except Exception as exc:
            messagebox.showwarning("保存失败", f"保存文件时出错: {exc}")
