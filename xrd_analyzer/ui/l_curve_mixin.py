"""
ui/l_curve_mixin.py
--------------------
L-Curve 正则化参数自动分析 Mixin：
  - 扫描一系列 α 值，计算残差范数与解的粗糙度范数
  - 最大曲率法（点到端点直线的最远距离）定位拐点
  - 弹出独立窗口展示 L-Curve，并提供一键应用推荐 α
"""
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import messagebox
import tkinter as tk

from ..core.peak_functions import (
    calc_kalpha2_position,
    precompile_numba_functions,
    SLOPE_M, M_REF_MIN, D_REF_MAX,
)
from ..core.fitting import (
    WAVELENGTHS,
    build_basis_matrix,
    build_regularization_matrix,
    solve_nnls_regularized,
)


class LCurveMixin:
    """L-Curve 分析的所有方法。"""

    # ------------------------------------------------------------------
    # 线程入口
    # ------------------------------------------------------------------

    def run_l_curve_thread(self):
        """启动 L-Curve 计算线程，防止卡死 UI。"""
        if not self.data_loaded:
            messagebox.showwarning("提示", "请先导入数据！")
            return
        self.btn_lcurve.config(state=tk.DISABLED)
        self.ui_set(self.progress_var, "正在进行 L-Curve 扫描...")
        threading.Thread(target=self.compute_l_curve, daemon=True).start()

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------

    def compute_l_curve(self):
        """执行 α 扫描，寻找 L-Curve 拐点（最大曲率法）。"""
        try:
            # ── 1. 准备参数 ─────────────────────────────────────────
            source   = self.source_var.get()
            lam1, lam2 = WAVELENGTHS.get(source, WAVELENGTHS["Cu"])
            mu_centers = [s.get() for s in self.peak_mu_sliders]

            # 截取 + 扣背底
            mask = (
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
            y = y_raw - (slope_bg * x + intercept_bg)
            y[y < 0] = 0
            if y.max() <= 0:
                return
            y_scaled = y / y.max()

            # ── 2. 构建粒径网格与基矩阵 ─────────────────────────────
            d_min, d_max = self.slider_d_min.get(), self.slider_d_max.get()
            raw_pts  = int((d_max - d_min) / 0.1)
            num_pts  = min(800, max(200, raw_pts))
            D_range  = np.linspace(d_min, d_max, num_pts)

            L_single    = build_regularization_matrix(len(D_range))
            basis_total, _, _ = build_basis_matrix(x, mu_centers, D_range, lam1, lam2)

            # ── 3. 扫描 α（对数空间，50 点）──────────────────────────
            alpha_values   = np.logspace(-2, 4, 50)
            residual_norms = []
            solution_norms = []
            n_peaks = len(mu_centers)

            for idx, alpha in enumerate(alpha_values):
                self.ui_set(self.progress_var, f"L-Curve 扫描: {idx + 1}/{len(alpha_values)}")

                f_total, _ = solve_nnls_regularized(
                    basis_total, y_scaled, L_single, n_peaks, alpha
                )
                from scipy.linalg import block_diag
                L_combined = block_diag(*([L_single] * n_peaks))

                res_norm = float(np.linalg.norm(basis_total.dot(f_total) - y_scaled))
                sol_norm = float(np.linalg.norm(L_combined.dot(f_total)))
                residual_norms.append(res_norm)
                solution_norms.append(sol_norm)

            # ── 4. 最大曲率法定位拐点 ───────────────────────────────
            x_log = np.log10(residual_norms)
            y_log = np.log10(solution_norms)

            x_min, x_max = x_log.min(), x_log.max()
            y_min, y_max = y_log.min(), y_log.max()
            x_norm = (x_log - x_min) / (x_max - x_min + 1e-15)
            y_norm = (y_log - y_min) / (y_max - y_min + 1e-15)

            # 端点连线方程 Ax + By + C = 0
            x1, y1 = x_norm[0],  y_norm[0]
            x2, y2 = x_norm[-1], y_norm[-1]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            denom = np.sqrt(A ** 2 + B ** 2) + 1e-15
            distances = np.abs(A * x_norm + B * y_norm + C) / denom

            best_idx   = int(np.argmax(distances))
            best_alpha = float(alpha_values[best_idx])

            # ── 5. 弹出结果窗口 ──────────────────────────────────────
            self.ui(
                self.show_l_curve_popup,
                residual_norms, solution_norms, alpha_values, best_idx,
            )
            self.ui_set(self.progress_var, f"推荐 Alpha: {best_alpha:.2f}")

        except Exception as exc:
            print(f"L-Curve Error: {exc}")
            self.ui(messagebox.showerror, "错误", f"L-Curve 计算失败: {exc}")
        finally:
            self.ui(lambda: self.btn_lcurve.config(state=tk.NORMAL))

    # ------------------------------------------------------------------
    # 结果弹窗
    # ------------------------------------------------------------------

    def show_l_curve_popup(self, x_data, y_data, alphas, best_idx):
        """在独立窗口中显示 L-Curve，并提供一键应用推荐 α 的按钮。"""
        top = tk.Toplevel(self.root)
        top.title("L-Curve 分析结果")
        top.geometry("700x550")

        fig_lc = plt.figure(figsize=(7, 5), dpi=100)
        ax = fig_lc.add_subplot(111)
        ax.loglog(x_data, y_data, "b.-", markersize=8, label="L-Curve")

        best_x     = x_data[best_idx]
        best_y     = y_data[best_idx]
        best_alpha = alphas[best_idx]

        ax.loglog(
            best_x, best_y, "ro", markersize=12,
            label=f"Optimal α = {best_alpha:.2f}",
        )
        for i in range(0, len(alphas), 3):
            ax.text(x_data[i], y_data[i], f"{alphas[i]:.1e}", fontsize=8)

        ax.set_xlabel("Residual Norm ‖Af − y‖  (Fitting Error)")
        ax.set_ylabel("Solution Norm ‖Lf‖  (Roughness)")
        ax.set_title("L-Curve Parameter Selection")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig_lc, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def apply_alpha():
            self.slider_alpha.set(best_alpha)
            top.destroy()
            messagebox.showinfo(
                "提示",
                f"已应用推荐参数 Alpha = {best_alpha:.2f}\n请点击「精细计算」重新拟合。",
            )

        tk.Button(
            top,
            text=f"应用推荐值 (α = {best_alpha:.2f})",
            command=apply_alpha,
            bg="#FF9800", fg="white",
            font=("Arial", 12, "bold"),
        ).pack(pady=10)
