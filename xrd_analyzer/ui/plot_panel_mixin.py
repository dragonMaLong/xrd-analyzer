"""
ui/plot_panel_mixin.py
-----------------------
右侧图表区 Mixin：
  - 三联图布局（预览图 / XRD 拟合图 / 粒径分布图）
  - 鼠标拖拽交互（移动范围线、峰位标记）
  - 图例点击显示/隐藏
  - Artist 清理与安全重绘工具
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import find_peaks


class PlotPanelMixin:
    """右侧图表面板的所有 UI/绘图方法。"""

    # ------------------------------------------------------------------
    # 初始化图表区
    # ------------------------------------------------------------------

    def setup_plots(self):
        """创建三联图（顶部预览 / 左下拟合 / 右下分布）并嵌入 Tkinter。"""
        import tkinter as tk
        from tkinter import ttk

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(15, 5.8), dpi=100)
        self.preview_ax = self.fig.add_axes([0.05, 0.75, 0.90, 0.20])
        self.axes0      = self.fig.add_axes([0.05, 0.08, 0.42, 0.60])
        self.axes1      = self.fig.add_axes([0.53, 0.08, 0.42, 0.60])

        # 工具栏放顶部独立容器
        topbar = tk.Frame(self.right_frame)
        topbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, topbar, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, anchor="nw", padx=4, pady=2)

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.right_frame.update_idletasks()
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.94)
        self.canvas.draw()

        # 进度标签 + 进度条
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            self.right_frame, textvariable=self.progress_var, anchor="w"
        )
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "green.Horizontal.TProgressbar",
            background="#32CD32", troughcolor="lightgray",
            borderwidth=1, lightcolor="#32CD32", darkcolor="#32CD32",
        )
        self.progress_bar = ttk.Progressbar(
            self.right_frame, orient="horizontal",
            length=100, mode="determinate",
            style="green.Horizontal.TProgressbar",
        )
        self.progress_label.place_forget()
        self.progress_bar.place_forget()

        # 同步 Figure 尺寸到画布实际像素
        def _sync(event=None, once=False):
            w = self.canvas_widget.winfo_width()
            h = self.canvas_widget.winfo_height()
            if w <= 1 or h <= 1:
                if not once:
                    self.root.after(50, _sync)
                return
            dpi = self.fig.get_dpi() or 100
            self.fig.set_size_inches(w / dpi, h / dpi, forward=True)
            self._safe_draw_idle()

        self.canvas_widget.bind("<Map>", lambda e: _sync(once=True))
        self.root.after_idle(_sync)

        # 窗口 resize 去抖重绘
        def _redraw_after_resize(_evt=None):
            if hasattr(self, "_resize_after_id"):
                self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = self.root.after(50, self._safe_draw_idle)

        self.right_frame.bind("<Configure>", _redraw_after_resize)

    # ------------------------------------------------------------------
    # 安全重绘工具
    # ------------------------------------------------------------------

    def _sanitize_axes_texts(self, ax):
        """清理 Axes 上已失效（figure=None）的 Text，避免 idle_draw 崩溃。"""
        try:
            stale = [
                t for t in list(getattr(ax, "texts", []))
                if getattr(t, "figure", None) is None
            ]
            for t in stale:
                try:
                    ax.texts.remove(t)
                except Exception:
                    pass
        except Exception:
            pass

    def _sanitize_axes_artists(self, ax):
        """清理 Axes 中已失效的 artist，避免 draw 阶段空引用报错。"""
        try:
            children = list(getattr(ax, "_children", []))
            stale = [
                a for a in children
                if getattr(a, "figure", None) is None
                or getattr(a, "axes",   None) is not ax
            ]
            for a in stale:
                try:
                    a.remove()
                except Exception:
                    pass
            if stale and hasattr(ax, "_children"):
                ax._children = [
                    a for a in ax._children
                    if getattr(a, "figure", None) is not None
                    and getattr(a, "axes",   None) is ax
                ]
        except Exception:
            pass

    def _safe_draw_idle(self):
        """重绘前清理失效 artist，避免 Tk idle_draw 阶段崩溃。"""
        try:
            for ax in (self.preview_ax, self.axes0, self.axes1):
                self._sanitize_axes_artists(ax)
                self._sanitize_axes_texts(ax)
        except Exception:
            pass
        self.canvas.draw_idle()

    def _safe_hide(self, artist):
        """安全隐藏 artist，跳过已失效对象。"""
        if artist is None:
            return
        if getattr(artist, "figure", None) is None:
            return
        try:
            artist.set_visible(False)
        except Exception as e:
            print("隐藏对象失败:", e)

    def _enforce_clipping(self, ax):
        """强制所有可见元素裁剪到 Axes 边界，防止溢出到其他子图。"""
        try:
            patch = ax.patch
            if patch is None:
                return
            patch.set_clip_on(True)
            for collection in (
                ax.lines, ax.collections, ax.texts, ax.patches, ax.images
            ):
                for obj in collection:
                    obj.set_clip_on(True)
                    obj.set_clip_path(patch)
            leg = ax.get_legend()
            if leg is not None:
                leg.set_clip_on(True)
                leg.set_clip_path(patch)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 计算完成后的覆盖层隐藏
    # ------------------------------------------------------------------

    def _hide_axes0_overlays(self):
        """隐藏 axes0 上的蓝色范围线、红框、峰标注，不影响预览图。"""
        for name in ("line_min", "line_max", "axes0_line_min", "axes0_line_max"):
            ln = getattr(self, name, None)
            if (ln is not None
                    and getattr(ln, "figure", None) is not None
                    and getattr(ln, "axes", None) is self.axes0):
                try:
                    ln.set_visible(False)
                except Exception:
                    pass

        rects = []
        if getattr(self, "rect_axes0", None) is not None:
            rects.append(self.rect_axes0)
        if isinstance(getattr(self, "peak_rects", None), (list, tuple)):
            rects.extend([r for r in self.peak_rects if r is not None])
        for r in rects:
            if (getattr(r, "figure", None) is not None
                    and getattr(r, "axes", None) is self.axes0):
                try:
                    r.set_visible(False)
                except Exception:
                    pass

        if hasattr(self, "peak_artists"):
            keep = []
            for a in self.peak_artists:
                if (getattr(a, "figure", None) is not None
                        and getattr(a, "axes", None) is self.axes0):
                    try:
                        a.set_visible(False)
                    except Exception:
                        pass
                    keep.append(a)
            self.peak_artists = keep

        self.ui(self._safe_draw_idle)

    # ------------------------------------------------------------------
    # 预览图更新
    # ------------------------------------------------------------------

    def update_preview(self, val=None):
        if not self.data_loaded:
            return

        angle_min = self.slider_min.get()
        angle_max = self.slider_max.get()
        self._refresh_peak_slider_bounds()

        # ── 顶部全览图 ──────────────────────────────────────────────
        self.preview_ax.cla()
        self.peak_mu_rects_preview = []
        self.preview_ax.plot(
            self.x_data, self.y_data,
            ".", color="gray", alpha=0.5, label=self.current_file_name,
        )
        self.line_min = self.preview_ax.axvline(
            angle_min, color="blue", linestyle="--", picker=True, pickradius=5
        )
        self.line_max = self.preview_ax.axvline(
            angle_max, color="blue", linestyle="--", picker=True, pickradius=5
        )
        for i, slider in enumerate(self.peak_mu_sliders):
            mu    = slider.get()
            color = self.peak_colors[self.active_peak_indices[i]]
            rect  = self.preview_ax.axvspan(mu - 0.1, mu + 0.1, color=color, alpha=0.4, picker=True)
            self.peak_mu_rects_preview.append(rect)
        self.preview_ax.set_title("完整数据预览", fontsize=10)
        self.preview_ax.set_xlim(self.x_data.min(), self.x_data.max())
        self.preview_ax.legend(loc="upper right")

        # ── 左下拟合预览图 ──────────────────────────────────────────
        self.axes0.cla()
        self.peak_mu_rects_axes0 = []
        mask = (self.x_data >= angle_min) & (self.x_data <= angle_max)
        if np.any(mask):
            self.axes0.plot(
                self.x_data[mask], self.y_data[mask],
                ".", color="gray", alpha=0.6,
            )
            for i, slider in enumerate(self.peak_mu_sliders):
                mu    = slider.get()
                color = self.peak_colors[self.active_peak_indices[i]]
                rect  = self.axes0.axvspan(mu - 0.02, mu + 0.02, color=color, alpha=0.4)
                self.peak_mu_rects_axes0.append(rect)
        self.axes0.set_xlim(angle_min, angle_max)
        self.axes0.set_title("拟合范围预览")
        self.axes0.set_xlabel("2θ (°)")
        self.axes0.set_ylabel("强度")

        # ── 右下分布图（计算前清空）──────────────────────────────────
        if not self.results_ready:
            self.axes1.cla()
            self.axes1.set_title("粒径分布 (计算后显示)")

        self._enforce_clipping(self.preview_ax)
        self._enforce_clipping(self.axes0)
        self._enforce_clipping(self.axes1)
        self.ui(self._safe_draw_idle)

    # ------------------------------------------------------------------
    # 结果图更新
    # ------------------------------------------------------------------

    def update_multi_peak_plots(self):
        """计算完成后，更新左下 XRD 拟合图和右下粒径分布图。"""
        # ── 左下：XRD 拟合分解图 ────────────────────────────────────
        self.axes0.cla()
        x, y_raw, y, bg = (
            self.x_segment, self.y_segment_raw,
            self.y_segment, self.background,
        )
        self.axes0.plot(x, y_raw, ".", c="gray", alpha=0.5, label="原始数据")
        self.axes0.plot(x, bg,    "k--", alpha=0.7, label="背景")

        total_fit_curve = np.zeros_like(x)
        cmap = plt.get_cmap("tab10")

        for i, info in enumerate(self.all_peak_info):
            f_segment  = info["f_segment"]
            basis_k1   = info["basis_k1"]
            basis_k2   = info["basis_k2"]
            peak_color = info["color"]

            peak_fit = (basis_k1.dot(f_segment) + basis_k2.dot(f_segment)) * y.max() + bg
            total_fit_curve += peak_fit - bg

            # ── 局部组分分解 ─────────────────────────────────────────
            peak_components = []
            for j, detail in enumerate(info["peak_details"]):
                idx = detail.get("indices", None)
                if idx is None or len(idx) == 0:
                    continue
                f_component = np.zeros_like(f_segment)
                f_component[idx] = f_segment[idx]
                comp_fit = (
                    basis_k1[:, idx].dot(f_component[idx])
                    + basis_k2[:, idx].dot(f_component[idx])
                ) * y.max() + bg
                peak_components.append({
                    "fit":       comp_fit,
                    "color":     cmap(j % 10),
                    "d_center":  detail["center"],
                    "percentage": detail["percentage"],
                })

            for comp in peak_components:
                self.axes0.plot(x, comp["fit"], "-", lw=1.5, color=comp["color"])
                self.axes0.fill_between(
                    x, bg, comp["fit"],
                    where=(comp["fit"] >= bg), color=comp["color"], alpha=0.3,
                )
                pk_idx = np.argmax(comp["fit"] - bg)
                self.axes0.text(
                    x[pk_idx], comp["fit"][pk_idx] * 1.05,
                    f'{comp["d_center"]:.2f}nm ({comp["percentage"]:.0f}%)',
                    ha="center", va="bottom",
                    color=comp["color"], fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

            pk_idx = np.argmax(peak_fit - bg)
            self.axes0.text(
                x[pk_idx], peak_fit[pk_idx],
                f"峰 {self.active_peak_indices[i] + 1}",
                ha="center", va="bottom",
                color=peak_color, fontsize=9, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

        self.axes0.plot(x, total_fit_curve + bg, "k-", lw=2.5, alpha=0.7, label="总拟合")
        self.axes0.set_xlabel("2θ (°)")
        self.axes0.set_ylabel("强度")
        self.axes0.set_title("XRD多峰拟合及粒径分解")
        self.axes0.set_xlim(x.min(), x.max())

        # ── 右下：粒径分布图 ────────────────────────────────────────
        self.axes1.cla()
        lines, labels = [], []
        self.legend_handles   = {}
        self.actual_components = {}
        self.dist_texts        = {}

        eps = 1e-12
        D_range = self.D_range

        # 计算总面积（用于归一化）
        A_total_sum = 0.0
        for info in self.all_peak_info:
            A_total_sum += sum(
                float(det.get("area", 0.0)) for det in info.get("peak_details", [])
            )
        A_total_sum = max(A_total_sum, eps)

        # 全局总分布
        global_f_sum = sum(
            np.asarray(info["f_segment"], dtype=float) for info in self.all_peak_info
        )
        global_y_pdf = global_f_sum / A_total_sum

        gl_line, = self.axes1.plot(
            D_range, global_y_pdf, color="black", lw=2.5, linestyle="--", zorder=1
        )
        gl_fill  = self.axes1.fill_between(
            D_range, global_y_pdf, 0, color="gray", alpha=0.2, zorder=1
        )
        gl_dummy, = self.axes1.plot([], [], color="black", lw=2.5, label="总分布")
        lines.append(gl_dummy);  labels.append("总分布")
        self.legend_handles["global"]    = gl_dummy
        self.actual_components["global"] = {"line": gl_line, "fills": [gl_fill]}
        self.dist_texts["global"]        = []

        Hc_max = float(global_y_pdf.max())

        # 各分峰分布
        for i, info in enumerate(self.all_peak_info):
            peak_id   = self.active_peak_indices[i]
            color     = info["color"]
            f_total   = np.asarray(info["f_segment"], dtype=float)
            line_y_pdf = f_total / A_total_sum

            actual_line, = self.axes1.plot(
                D_range, line_y_pdf, color=color, lw=2, linestyle="-", zorder=2
            )
            fill = self.axes1.fill_between(
                D_range, line_y_pdf, 0, color=color, alpha=0.15, zorder=2
            )

            label    = f"峰 {peak_id + 1} 粒径分布"
            dummy, = self.axes1.plot([], [], color=color, lw=2, linestyle="-", label=label)
            lines.append(dummy);  labels.append(label)
            self.legend_handles[peak_id]    = dummy
            self.actual_components[peak_id] = {"line": actual_line, "fills": [fill]}

            texts = []
            for det in info.get("peak_details", []):
                idx = det.get("indices", [])
                if len(idx) == 0:
                    continue
                local_max_idx = np.argmax(line_y_pdf[idx])
                real_idx = idx[local_max_idx]
                cx = float(D_range[real_idx])
                cy = float(line_y_pdf[real_idx])
                Hc_max = max(Hc_max, cy)
                txt = self.axes1.text(
                    cx, cy * 1.05, f"{det['center']:.1f}nm",
                    ha="center", va="bottom",
                    color=color, fontsize=8, fontweight="bold", zorder=3,
                )
                texts.append(txt)
            self.dist_texts[peak_id] = texts

        self.axes1.set_xlabel("晶粒尺寸 (nm)")
        self.axes1.set_ylabel("体积密度 (Volume Density)")
        self.axes1.set_title("晶粒尺寸分布 (总分布 vs 分峰)")
        self.axes1.set_xlim(D_range.min(), D_range.max())
        self.axes1.set_ylim(0, Hc_max * 1.2)

        if lines:
            leg = self.axes1.legend(lines, labels, loc="upper right")
            for idx2, legline in enumerate(leg.get_lines()):
                key = "global" if idx2 == 0 else self.active_peak_indices[idx2 - 1]
                legline.set_picker(True)
                legline.set_pickradius(5)
                self.legend_handles[key] = legline

        self.fig.canvas.mpl_connect("pick_event", self.on_pick_legend)
        self._enforce_clipping(self.axes1)
        self.ui(self._safe_draw_idle)

    def on_pick_legend(self, event):
        """点击图例切换对应分布曲线的可见性。"""
        clicked_key = None
        for key, handle in self.legend_handles.items():
            if handle is event.artist:
                clicked_key = key
                break
        if clicked_key is None:
            return

        comps = self.actual_components.get(clicked_key, {})
        line  = comps.get("line", None)
        fills = comps.get("fills", [])

        if line is not None:
            cur_vis = line.get_visible()
        elif fills:
            cur_vis = fills[0].get_visible()
        else:
            return

        new_visible = not cur_vis
        if line is not None:
            line.set_visible(new_visible)
        for f in fills:
            try:
                f.set_visible(new_visible)
            except Exception:
                pass
        for t in self.dist_texts.get(clicked_key, []):
            if getattr(t, "figure", None) is not None:
                t.set_visible(new_visible)

        event.artist.set_alpha(1.0 if new_visible else 0.2)
        self.ui(self._safe_draw_idle)

    # ------------------------------------------------------------------
    # 鼠标交互（拖拽范围线 / 峰位标记）
    # ------------------------------------------------------------------

    def bind_events(self):
        """绑定鼠标按下 / 移动 / 释放事件到 Matplotlib Canvas。"""
        self.canvas.mpl_connect("button_press_event",   self.on_press)
        self.canvas.mpl_connect("motion_notify_event",  self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def on_press(self, event):
        if not self.data_loaded:
            return
        tol = 0.5
        x   = event.xdata
        self.dragging_slider     = None
        self.dragging_peak_index = None

        if event.inaxes in (self.preview_ax, self.axes0) and x is not None:
            if abs(x - self.slider_min.get()) < tol:
                self.dragging_slider = self.slider_min
            elif abs(x - self.slider_max.get()) < tol:
                self.dragging_slider = self.slider_max
            else:
                for i, slider in enumerate(self.peak_mu_sliders):
                    mu = slider.get()
                    if mu - 0.1 <= x <= mu + 0.1:
                        self.dragging_slider     = slider
                        self.dragging_peak_index = i
                        break

        if self.dragging_slider is not None:
            try:
                self.toolbar.release_zoom(event)
                self.toolbar.release_pan(event)
            except Exception:
                pass

    def on_motion(self, event):
        if event.inaxes in (self.preview_ax, self.axes0) and event.xdata:
            tol = 0.5
            x   = event.xdata
            cursor_set = False
            if (abs(x - self.slider_min.get()) < tol
                    or abs(x - self.slider_max.get()) < tol):
                self.canvas.get_tk_widget().config(cursor="sb_h_double_arrow")
                cursor_set = True
            else:
                for slider in self.peak_mu_sliders:
                    if slider.get() - 0.1 <= x <= slider.get() + 0.1:
                        self.canvas.get_tk_widget().config(cursor="sb_h_double_arrow")
                        cursor_set = True
                        break
            if not cursor_set:
                self.canvas.get_tk_widget().config(cursor="arrow")

        if self.dragging_slider and event.xdata:
            self.dragging_slider.set(event.xdata)

            if self.dragging_slider in (self.slider_min, self.slider_max):
                self.line_min.set_xdata([self.slider_min.get()] * 2)
                self.line_max.set_xdata([self.slider_max.get()] * 2)
                self._refresh_peak_slider_bounds()
            elif self.dragging_peak_index is not None:
                mu        = self.dragging_slider.get()
                new_left  = mu - 0.1
                new_right = mu + 0.1

                rect_p = self.peak_mu_rects_preview[self.dragging_peak_index]
                y0p, y1p = self.preview_ax.get_ylim()
                if hasattr(rect_p, "set_verts"):
                    rect_p.set_verts([[
                        (new_left, y0p), (new_left, y1p),
                        (new_right, y1p), (new_right, y0p),
                    ]])

                rect_a = self.peak_mu_rects_axes0[self.dragging_peak_index]
                y0a, y1a = self.axes0.get_ylim()
                if hasattr(rect_a, "set_verts"):
                    rect_a.set_verts([[
                        (new_left, y0a), (new_left, y1a),
                        (new_right, y1a), (new_right, y0a),
                    ]])

            self.ui(self._safe_draw_idle)

    def on_release(self, event):
        self.dragging_slider     = None
        self.dragging_peak_index = None
        self.canvas.get_tk_widget().config(cursor="arrow")
        self._refresh_peak_slider_bounds()
