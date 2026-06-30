"""
ui/l_curve_mixin.py
--------------------
PyQt5 L-Curve 正则化参数自动分析 Mixin。
"""
import threading

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout

from ..core.fitting import (
    WAVELENGTHS,
    build_basis_matrix,
    build_regularization_matrix,
    solve_nnls_regularized,
)
from .qt_controls import MessageBoxAdapter as messagebox


class LCurveMixin:
    """L-Curve 分析的所有方法。"""

    def _collect_l_curve_params(self):
        angle_min = self.slider_min.get()
        angle_max = self.slider_max.get()
        fit_peak_indices = self._selected_peak_indices_in_fit_range(angle_min, angle_max)
        return {
            "source": self.source_var.get(),
            "mu_centers": [self.peak_mu_sliders[i].get() for i in fit_peak_indices],
            "angle_min": angle_min,
            "angle_max": angle_max,
            "d_min": float(getattr(self, "particle_size_min", 0.1)),
            "d_max": float(getattr(self, "particle_size_max", 100.0)),
            "d_step": float(getattr(self, "particle_size_step", 0.1)),
            "instrument_fwhm": float(getattr(self, "instrument_fwhm", 0.0)),
            "baseline_state": self._current_manual_baseline_state(),
        }

    def run_l_curve_thread(self):
        """启动 L-Curve 计算线程，防止卡死 UI。"""
        if not self.data_loaded:
            messagebox.showwarning("提示", "请先导入数据！")
            return
        if not self.active_peak_indices:
            messagebox.showwarning("提示", "请至少选择一个峰。")
            return
        params = self._collect_l_curve_params()
        if not params["mu_centers"]:
            messagebox.showwarning("提示", "当前蓝色拟合范围内没有峰，请先在范围内添加或移动峰。")
            return
        self.btn_lcurve.setEnabled(False)
        self.ui_set(self.progress_var, "正在进行 L-Curve 扫描...")
        self.progress_label.show()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        threading.Thread(target=self.compute_l_curve, args=(params,), daemon=True).start()

    def compute_l_curve(self, params):
        """执行 α 扫描，寻找 L-Curve 拐点（最大曲率法）。"""
        try:
            source = params["source"]
            lam1, lam2 = WAVELENGTHS.get(source, WAVELENGTHS["Cu"])
            mu_centers = list(params["mu_centers"])
            inst_fwhm = float(params["instrument_fwhm"])
            angle_min = params["angle_min"]
            angle_max = params["angle_max"]

            mask = (self.x_data >= angle_min) & (self.x_data <= angle_max)
            x = self.x_data[mask]
            y_raw = self.y_data[mask]
            if len(x) < 2:
                self.ui(messagebox.showwarning, "提示", "当前角度范围内没有足够的数据点。")
                return

            background = self._compute_background_for_segment(
                x,
                y_raw,
                angle_min,
                angle_max,
                params.get("baseline_state"),
            )
            y = y_raw - background
            y[y < 0] = 0
            if y.max() <= 0:
                self.ui_set(self.progress_var, "错误：无有效信号")
                return
            y_scaled = y / y.max()

            d_min, d_max = params["d_min"], params["d_max"]
            d_step = float(params.get("d_step", getattr(self, "particle_size_step", 0.1)) or 0.1)
            if hasattr(self, "_build_particle_size_grid"):
                D_range = self._build_particle_size_grid(d_min, d_max, d_step)
            else:
                D_range = np.arange(float(d_min), float(d_max) + d_step * 0.5, d_step)

            L_single = build_regularization_matrix(len(D_range))
            basis_total, _, _ = build_basis_matrix(
                x,
                mu_centers,
                D_range,
                lam1,
                lam2,
                instrument_fwhm_deg=inst_fwhm,
            )

            alpha_values = np.logspace(-2, 4, 50)
            residual_norms = []
            solution_norms = []
            n_peaks = len(mu_centers)

            from scipy.linalg import block_diag

            L_combined = block_diag(*([L_single] * n_peaks))
            for idx, alpha in enumerate(alpha_values):
                if self.stop_flag.is_set():
                    self.ui_set(self.progress_var, "已停止")
                    return
                pct = int((idx + 1) * 100 / len(alpha_values))
                self.ui(self.progress_bar.setValue, pct)
                self.ui_set(self.progress_var, f"L-Curve 扫描: {idx + 1}/{len(alpha_values)}")

                f_total, _ = solve_nnls_regularized(
                    basis_total, y_scaled, L_single, n_peaks, alpha
                )
                res_norm = float(np.linalg.norm(basis_total.dot(f_total) - y_scaled))
                sol_norm = float(np.linalg.norm(L_combined.dot(f_total)))
                residual_norms.append(res_norm)
                solution_norms.append(sol_norm)

            x_log = np.log10(residual_norms)
            y_log = np.log10(solution_norms)

            x_min, x_max = x_log.min(), x_log.max()
            y_min, y_max = y_log.min(), y_log.max()
            x_norm = (x_log - x_min) / (x_max - x_min + 1e-15)
            y_norm = (y_log - y_min) / (y_max - y_min + 1e-15)

            x1, y1 = x_norm[0], y_norm[0]
            x2, y2 = x_norm[-1], y_norm[-1]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            denom = np.sqrt(A**2 + B**2) + 1e-15
            distances = np.abs(A * x_norm + B * y_norm + C) / denom

            best_idx = int(np.argmax(distances))
            best_alpha = float(alpha_values[best_idx])

            self.ui(
                self.show_l_curve_popup,
                residual_norms,
                solution_norms,
                alpha_values,
                best_idx,
            )
            self.ui_set(self.progress_var, f"推荐 Alpha: {best_alpha:.2f}")

        except Exception as exc:
            print(f"L-Curve Error: {exc}")
            self.ui(messagebox.showerror, "错误", f"L-Curve 计算失败: {exc}")
        finally:
            self.ui(self.btn_lcurve.setEnabled, True)

    def show_l_curve_popup(self, x_data, y_data, alphas, best_idx):
        """在独立窗口中显示 L-Curve，并提供一键应用推荐 α 的按钮。"""
        top = QDialog(self)
        top.setWindowTitle("L-Curve 分析结果")
        top.resize(700, 550)
        layout = QVBoxLayout(top)

        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.setTitle("L-Curve Parameter Selection", color="#111827", size="10pt")
        plot.setLabel("bottom", "Residual Norm ||Af - y||  (Fitting Error)")
        plot.setLabel("left", "Solution Norm ||Lf||  (Roughness)")
        plot.showGrid(x=True, y=True, alpha=0.32)
        plot.getPlotItem().setLogMode(x=True, y=True)
        legend = plot.addLegend(offset=(10, 10))
        legend.setBrush(pg.mkBrush(255, 255, 255, 220))
        legend.setPen(pg.mkPen("#d1d5db"))

        x_arr = np.asarray(x_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)
        curve = pg.PlotDataItem(
            x_arr,
            y_arr,
            pen=pg.mkPen("#2563eb", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#2563eb"),
            symbolPen=pg.mkPen("#ffffff", width=1),
            name="L-Curve",
        )
        plot.addItem(curve)

        best_x = x_data[best_idx]
        best_y = y_data[best_idx]
        best_alpha = alphas[best_idx]
        best_item = pg.ScatterPlotItem(
            [best_x],
            [best_y],
            size=12,
            brush=pg.mkBrush("#ef4444"),
            pen=pg.mkPen("#991b1b", width=1.5),
            name=f"Optimal alpha = {best_alpha:.2f}",
        )
        plot.addItem(best_item)
        for i in range(0, len(alphas), 3):
            label = pg.TextItem(f"{alphas[i]:.1e}", color="#374151", anchor=(0, 1))
            font = label.textItem.font()
            font.setPointSize(7)
            label.textItem.setFont(font)
            label.setPos(float(x_data[i]), float(y_data[i]))
            plot.addItem(label)

        layout.addWidget(plot, 1)

        def apply_alpha():
            self.slider_alpha.set(float(best_alpha))
            top.accept()
            messagebox.showinfo(
                "提示",
                f"已应用推荐参数 Alpha = {best_alpha:.2f}\n请点击「精细计算」重新拟合。",
            )

        btn = QPushButton(f"应用推荐值 (α = {best_alpha:.2f})")
        btn.setStyleSheet(
            "background: #f8f9fa; color: #2c3135; font: 9pt 'Microsoft YaHei';"
            "border: 1px solid #c9ced3; border-radius: 3px; padding: 4px 8px;"
        )
        btn.clicked.connect(apply_alpha)
        layout.addWidget(btn)

        top.exec_()
