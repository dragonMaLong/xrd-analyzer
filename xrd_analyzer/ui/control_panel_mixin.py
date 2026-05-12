"""
ui/control_panel_mixin.py
--------------------------
左侧控制面板 Mixin：
  - 可滚动控件区（角度范围、峰选择、粒径范围、平滑因子、靶材选择）
  - 峰位滑块动态创建/销毁
  - 鼠标滚轮绑定
"""
import platform
import tkinter as tk
from tkinter import ttk
import webbrowser

from ..utils import resource_path


class ControlPanelMixin:
    """所有左侧面板相关的 UI 方法，通过 Mixin 方式混入 XRDApp。"""

    # ------------------------------------------------------------------
    # 可滚动控件区构建
    # ------------------------------------------------------------------

    def _build_scrollable_controls(self):
        """把滑块、复选框、Combobox 等控件全部放入可滚动的 Canvas 中。"""

        # ── 外层容器（固定在 scroll_zone 内）──────────────────────────
        self.scroll_container = tk.Frame(self.scroll_zone, bg="#F0F0F0")
        self.scroll_container.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(
            self.scroll_container, bd=0, highlightthickness=0, bg="#F0F0F0"
        )
        self.vscroll = ttk.Scrollbar(
            self.scroll_container,
            orient="vertical",
            command=self.scroll_canvas.yview,
            style="LeftPane.Vertical.TScrollbar",
        )
        self.scroll_canvas.configure(yscrollcommand=self.vscroll.set)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ── 真正承载控件的 Frame ────────────────────────────────────
        self.scroll_frame = tk.Frame(self.scroll_canvas, bg="#F0F0F0")
        self.scroll_window = self.scroll_canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
        )

        # 内容变化时更新滚动区域
        def _on_frame_cfg(event):
            self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            )

        self.scroll_frame.bind("<Configure>", _on_frame_cfg)

        # 跟随容器宽高
        def _sync_canvas(e):
            sbw = 16
            cw  = max(50, e.width - sbw - 2)
            self.scroll_canvas.configure(height=e.height, width=e.width - sbw)
            self.scroll_canvas.itemconfig(self.scroll_window, width=cw)
            self.root.after(
                1,
                lambda: self.scroll_canvas.configure(
                    scrollregion=self.scroll_canvas.bbox("all")
                ),
            )

        self.scroll_zone.bind("<Configure>", _sync_canvas)

        # 鼠标滚轮（仅在鼠标进入时生效）
        self._bind_scroll_wheel(self.scroll_canvas)

        # ── 以下控件全部放入 scroll_frame ──────────────────────────

        # 角度范围滑块
        self.slider_min = tk.Scale(
            self.scroll_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            label="起始角 (2θ)", resolution=0.01, command=self.update_preview,
        )
        self.slider_min.set(30)
        self.slider_min.pack(fill=tk.X, padx=5, pady=5)

        self.slider_max = tk.Scale(
            self.scroll_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            label="结束角 (2θ)", resolution=0.01, command=self.update_preview,
        )
        self.slider_max.set(70)
        self.slider_max.pack(fill=tk.X, padx=5, pady=5)

        ttk.Separator(self.scroll_frame, orient="horizontal").pack(fill="x", pady=10)

        # ── 峰选择复选框 + 动态峰位滑块 ─────────────────────────────
        tk.Label(
            self.scroll_frame, text="晶面/峰 选择",
            bg="#F0F0F0", font=("微软雅黑", 10, "bold"),
        ).pack(pady=(5, 0))

        self.peaks_frame = tk.Frame(self.scroll_frame, bg="#F0F0F0")
        self.peaks_frame.pack(fill=tk.X, padx=5, pady=5)

        # 只在第一次构建时创建复选框变量；rebuild 时复用
        if not self.peak_check_vars:
            for i in range(self.max_peaks):
                var = tk.IntVar(value=1 if i == 0 else 0)
                self.peak_check_vars.append(var)

        for i, var in enumerate(self.peak_check_vars):
            tk.Checkbutton(
                self.peaks_frame,
                text=f"峰 {i + 1} (Peak {i + 1})",
                variable=var,
                command=self.update_ui_for_peaks,
                bg="#F0F0F0", activebackground="#E0E0E0",
                font=("微软雅黑", 9),
            ).pack(anchor="w", pady=2)

        # 峰位滑块容器
        self.sliders_frame = tk.Frame(self.scroll_frame, bg="#F0F0F0")
        self.sliders_frame.pack(fill=tk.X, pady=5)

        ttk.Separator(self.scroll_frame, orient="horizontal").pack(fill="x", pady=10)

        # ── 粒径范围 + 平滑因子 ──────────────────────────────────────
        self.slider_d_min = tk.Scale(
            self.scroll_frame, from_=0.1, to=1, orient=tk.HORIZONTAL,
            label="最小粒径 (nm)", resolution=0.1, length=200,
        )
        self.slider_d_min.set(0.5)
        self.slider_d_min.pack(fill=tk.X, padx=5, pady=5)

        self.slider_d_max = tk.Scale(
            self.scroll_frame, from_=50, to=300, orient=tk.HORIZONTAL,
            label="最大粒径 (nm)", resolution=0.1, length=200,
        )
        self.slider_d_max.set(100)
        self.slider_d_max.pack(fill=tk.X, padx=5, pady=5)

        self.slider_alpha = tk.Scale(
            self.scroll_frame, from_=0.01, to=100, orient=tk.HORIZONTAL,
            label="平滑因子 (α)", resolution=0.01, length=200,
        )
        self.slider_alpha.set(1)
        self.slider_alpha.pack(fill=tk.X, padx=5, pady=5)

        ttk.Separator(self.scroll_frame, orient="horizontal").pack(fill="x", pady=10)

        # ── X 射线源选择 ─────────────────────────────────────────────
        tk.Label(
            self.scroll_frame, text="X射线源 (X-Ray Source)", bg="#F0F0F0"
        ).pack(pady=(10, 5))

        # 避免重复构建时覆盖已有变量
        if not hasattr(self, "source_var") or self.source_var is None:
            self.source_var = tk.StringVar(value="Cu")
        self.source_menu = ttk.Combobox(
            self.scroll_frame,
            textvariable=self.source_var,
            values=["Cu", "Co", "Fe", "Mo"],
            state="readonly", width=10, justify="center",
        )
        self.source_menu.pack(pady=(0, 10))

        # ── 网站 Banner ──────────────────────────────────────────────
        self._build_site_banner()

        # 初始化峰位滑块
        self.update_ui_for_peaks()
        self.update_preview()

    def _build_site_banner(self):
        """构建底部的网站图片 + 链接 Banner。"""
        site_banner = tk.Frame(self.scroll_frame, bg="#F0F0F0")
        site_banner.pack(fill=tk.X, padx=5, pady=(0, 10))

        try:
            self.site_png = tk.PhotoImage(file=resource_path("long.png"))
            maxw = 200
            if self.site_png.width() > maxw:
                r = max(1, self.site_png.width() // maxw)
                self.site_png = self.site_png.subsample(r, r)
            lbl_img = tk.Label(
                site_banner, image=self.site_png, bg="#F0F0F0", cursor="hand2"
            )
            lbl_img.pack()
            lbl_img.bind("<Button-1>", self._open_site)
        except Exception:
            pass  # 没有图片文件时静默跳过

        lbl_link = tk.Label(
            site_banner,
            text="www.dragonscience.top",
            fg="#1a73e8", bg="#F0F0F0",
            cursor="hand2",
            font=("微软雅黑", 7, "underline"),
        )
        lbl_link.pack(pady=(2, 0))
        lbl_link.bind("<Button-1>", self._open_site)

    def _open_site(self, *_):
        webbrowser.open_new("https://www.dragonscience.top")

    # ------------------------------------------------------------------
    # 动态峰位滑块管理
    # ------------------------------------------------------------------

    def update_ui_for_peaks(self):
        """根据复选框状态动态创建或销毁峰中心滑块，保持已有峰位不变。"""
        # 保存当前各峰的位置
        current_positions = {
            self.active_peak_indices[i]: slider.get()
            for i, slider in enumerate(self.peak_mu_sliders)
        }

        # 销毁旧滑块
        for slider in self.peak_mu_sliders:
            slider.destroy()
        self.peak_mu_sliders.clear()

        # 重新确定激活峰列表
        self.active_peak_indices = [
            i for i, var in enumerate(self.peak_check_vars) if var.get() == 1
        ]
        angle_min  = self.slider_min.get()
        angle_max  = self.slider_max.get()
        num_active = len(self.active_peak_indices)

        for i, peak_idx in enumerate(self.active_peak_indices):
            label_text = f"峰 {peak_idx + 1} 中心 (2θ)"
            slider = tk.Scale(
                self.sliders_frame,
                from_=angle_min, to=angle_max,
                orient=tk.HORIZONTAL,
                label=label_text, resolution=0.01, length=200,
                fg=self.peak_colors[peak_idx],
                bg="#F0F0F0", highlightbackground="#F0F0F0",
            )
            # 恢复已有峰的位置，新峰均匀分布
            if peak_idx in current_positions:
                pos = current_positions[peak_idx]
            else:
                pos = angle_min + (angle_max - angle_min) * (i + 1) / (num_active + 1)
            slider.set(pos)
            slider.pack(fill=tk.X, padx=5)
            slider.config(command=self.update_preview)
            self.peak_mu_sliders.append(slider)

        self.update_preview()

    def _refresh_peak_slider_bounds(self):
        """将峰位滑块的取值范围同步到当前蓝线范围，并钳制当前值。"""
        angle_min = self.slider_min.get()
        angle_max = self.slider_max.get()
        for s in self.peak_mu_sliders:
            s.config(from_=angle_min, to=angle_max)
            v = s.get()
            if v < angle_min:
                s.set(angle_min + 1e-6)
            elif v > angle_max:
                s.set(angle_max - 1e-6)

    # ------------------------------------------------------------------
    # 鼠标滚轮绑定
    # ------------------------------------------------------------------

    def _bind_scroll_wheel(self, widget):
        """让鼠标滚轮仅在鼠标悬停于 widget 上时作用于左侧滚动区。"""
        system = platform.system().lower()

        def _on_mousewheel(e):
            if "darwin" in system:
                delta = -1 if e.delta > 0 else 1
            else:
                delta = -int(e.delta / 120)
            self.scroll_canvas.yview_scroll(delta, "units")

        def _on_enter(_):
            widget.bind_all("<MouseWheel>", _on_mousewheel)
            widget.bind_all("<Button-4>",
                            lambda e: self.scroll_canvas.yview_scroll(-1, "units"))
            widget.bind_all("<Button-5>",
                            lambda e: self.scroll_canvas.yview_scroll(1, "units"))

        def _on_leave(_):
            widget.unbind_all("<MouseWheel>")
            widget.unbind_all("<Button-4>")
            widget.unbind_all("<Button-5>")

        widget.bind("<Enter>", _on_enter)
        widget.bind("<Leave>", _on_leave)
