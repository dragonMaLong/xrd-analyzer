"""
ui/app_window.py
----------------
PyQt5 XRDApp 主类：
  - 继承三个 Mixin，统一协调左侧面板、右侧图表和 L-Curve 功能
  - 管理数据加载、计算线程、结果后处理和 CSV 导出
"""
import csv
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QSettings, QStandardPaths, Qt, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QDesktopServices, QFont, QIcon, QColor, QPainter, QPainterPath, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
    QWidget,
    QTableWidgetItem,
)

from ..core.analysis import build_all_peak_info
from ..core.fitting import (
    INTENSITY_RATIO,
    WAVELENGTHS,
    _eval_candidate_for_index,
    build_regularization_matrix,
    fit_with_mu_list,
)
from ..io.file_reader import load_file as load_xrd_file
from ..update_checker import DEFAULT_UPDATE_REPOSITORY, UpdateInfo, check_for_update
from ..updater import UpdateDownloadError, download_update, launch_update_and_exit
from ..utils import resource_path
from ..version import __version__
from .control_panel_mixin import ControlPanelMixin
from .import_dialog import XRDFileImportDialog
from .l_curve_mixin import LCurveMixin
from .plot_panel_mixin import PlotPanelMixin
from .qt_controls import FileDialogAdapter as filedialog
from .qt_controls import MessageBoxAdapter as messagebox


DEFAULT_ANGLE_MIN = 60.0
DEFAULT_ANGLE_MAX = 74.6
APP_VERSION = __version__
UPDATE_REPOSITORY = DEFAULT_UPDATE_REPOSITORY
AUTO_UPDATE_CHECK_DELAY_MS = 1500


@dataclass
class XRDSample:
    path: str
    x_data: np.ndarray
    y_data: np.ndarray
    name: str
    metadata: dict
    status: str = "pending"
    compare_visible: bool = True
    peak_states: list[dict] = field(default_factory=list)
    analysis_state: dict = field(default_factory=dict)
    baseline_state: dict = field(default_factory=dict)
    marker_label_state: dict = field(default_factory=dict)
    plot_view_state: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


class _UiDispatcher(QObject):
    call_requested = pyqtSignal(object, tuple, dict)

    def __init__(self):
        super().__init__()
        self.call_requested.connect(self._run)

    @pyqtSlot(object, tuple, dict)
    def _run(self, fn, args, kwargs):
        fn(*args, **kwargs)


class UpdateCheckWorker(QObject):
    finished = pyqtSignal(object, bool)
    failed = pyqtSignal(str, bool)

    def __init__(self, current_version: str, repository: str, manual: bool) -> None:
        super().__init__()
        self.current_version = current_version
        self.repository = repository
        self.manual = manual

    def run(self) -> None:
        try:
            info = check_for_update(self.current_version, repository=self.repository, timeout=4.0)
        except Exception as exc:
            self.failed.emit(str(exc), self.manual)
            return
        self.finished.emit(info, self.manual)


class UpdateDownloadWorker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, info: UpdateInfo) -> None:
        super().__init__()
        self.info = info

    def run(self) -> None:
        try:
            path = download_update(
                self.info,
                progress_callback=lambda downloaded, total: self.progress.emit(downloaded, total),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(self.info, str(path))


class XRDApp(QMainWindow, ControlPanelMixin, PlotPanelMixin, LCurveMixin):
    """XRD 多峰拟合分析工具主窗口。"""

    def _stop_process_pool(self, executor, futures=()):
        """尽快终止正在执行的子进程任务。"""
        for fut in futures:
            fut.cancel()

        terminate_workers = getattr(executor, "terminate_workers", None)
        if terminate_workers is not None:
            terminate_workers()
            return

        for proc in getattr(executor, "_processes", {}).values():
            try:
                proc.terminate()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

    def __init__(self):
        super().__init__()
        self.root = self
        self.setWindowTitle(f"XRD晶粒尺寸分布分析-DragonScience V{APP_VERSION}")
        self._ui_dispatcher = _UiDispatcher()

        self.data_loaded = False
        self.results_ready = False
        self.stop_flag = threading.Event()
        self.dragging_slider = None
        self.dragging_peak_index = None
        self.D_STEP = 0.1

        self.line_min = None
        self.line_max = None

        self.max_peaks = 5
        self.peak_colors = ["#FF00FF", "#0077FF", "#00C853", "#FFAB00", "#00E5FF"]
        self.active_peak_indices = []
        self.result_active_peak_indices = []
        self.peak_mu_sliders = []
        self.peak_check_vars = []
        self.peak_rows = []
        self.peak_color_buttons = []
        self.peak_visible_buttons = []
        self._building_peak_controls = False
        self.manual_baseline_enabled = False
        self.manual_baseline_edited = False
        self.manual_baseline_user_points = []
        self.manual_baseline_endpoint_y = {"left": None, "right": None}
        self.manual_baseline_endpoint_deleted = set()
        self._manual_baseline_next_anchor_id = 1
        self._manual_baseline_curve_item = None
        self._manual_baseline_anchor_items = []
        self._syncing_manual_baseline_anchor = False
        self.particle_size_min = 0.5
        self.particle_size_max = 100.0
        self.instrument_fwhm = 0.0
        self.marker_label_state = {}
        self.plot_view_state = {}
        self.peak_mu_rects_preview = []
        self.peak_mu_rects_axes0 = []
        self.peak_mu_lines_axes0 = []
        self._plot_drag = None
        self.samples: list[XRDSample] = []
        self.active_sample_index = -1
        self._updating_compare_checks = False
        self._hovered_sample_row = -1
        self.settings = QSettings("XRDAnalyzer", "XRDAnalyzerPyQt5")
        self.import_directory = self._read_import_directory()
        self._import_available_sort = (0, Qt.AscendingOrder)
        self._checking_for_updates = False
        self._update_thread: QThread | None = None
        self._update_worker: UpdateCheckWorker | None = None
        self._update_download_thread: QThread | None = None
        self._update_download_worker: UpdateDownloadWorker | None = None
        self._update_progress_dialog: QProgressDialog | None = None
        self._available_update_info = None

        self.source_var = None

        icon_path = resource_path("logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._setup_ui()
        QTimer.singleShot(AUTO_UPDATE_CHECK_DELAY_MS, self._auto_check_for_updates)

    def _setup_ui(self):
        """构建完整窗口布局：左侧面板 + 右侧图表。"""
        self.setMinimumSize(800, 600)
        self.resize(1600, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.main_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStyleSheet(
            """
            QSplitter::handle:horizontal {
                background: #e5e7eb;
                margin: 0 2px;
            }
            QSplitter::handle:horizontal:hover {
                background: #93c5fd;
            }
            """
        )
        main_layout.addWidget(self.main_splitter, 1)

        self.LEFT_COL_W = 330
        self.left_frame = QFrame()
        self.left_frame.setObjectName("leftFrame")
        self.left_frame.setMinimumWidth(260)
        self.left_frame.setStyleSheet(
            "#leftFrame { background: #f3f4f5; border: 0; }"
        )
        left_layout = QVBoxLayout(self.left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)
        self.main_splitter.addWidget(self.left_frame)
        self._build_left_sidebar(left_layout)

        self.right_frame = QFrame()
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(4)
        self.main_splitter.addWidget(self.right_frame)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([self.LEFT_COL_W, 1270])

        self.setup_plots()
        self.bind_events()
        self.statusBar().showMessage("打开或拖入 TXT、RAW 文件")

    def ui(self, fn, *args, **kwargs):
        """线程安全地调度 UI 操作到 Qt 主线程。"""
        self._ui_dispatcher.call_requested.emit(fn, args, kwargs)

    def ui_set(self, var, value):
        self.ui(var.set, value)

    def _auto_check_for_updates(self) -> None:
        self.check_for_updates(manual=False)

    def check_for_updates(self, _checked: bool = False, *, manual: bool = True) -> None:
        if self._checking_for_updates:
            if manual:
                self.statusBar().showMessage("正在检查软件更新...", 3000)
            return

        self._checking_for_updates = True
        update_button = getattr(self, "update_button", None)
        if update_button is not None:
            update_button.setEnabled(False)
        if manual:
            self.statusBar().showMessage("正在连接更新源检查软件更新...", 3000)

        thread = QThread(self)
        worker = UpdateCheckWorker(APP_VERSION, UPDATE_REPOSITORY, manual)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_update_check_finished)
        worker.failed.connect(self._on_update_check_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_update_check_worker)
        self._update_thread = thread
        self._update_worker = worker
        thread.start()

    def _on_update_check_finished(self, info: UpdateInfo, manual: bool) -> None:
        self._finish_update_check()
        if not info.update_available:
            self._set_update_available_indicator(None)
            if manual:
                self.statusBar().showMessage(f"当前已是最新版本 v{info.current_version}", 5000)
                QMessageBox.information(self, "软件更新", f"当前已是最新版本 v{info.current_version}")
            return

        self._set_update_available_indicator(info)
        self.statusBar().showMessage(
            f"发现新版本 v{info.latest_version}，点击软件更新左侧蓝色云朵即可更新。",
            8000,
        )
        if not manual:
            return

        self._show_update_available_dialog(info)

    def _show_pending_update_dialog(self) -> None:
        info = self._available_update_info
        if info is None:
            self.check_for_updates(manual=True)
            return
        self.statusBar().showMessage(f"发现新版本 v{info.latest_version}，准备更新。", 5000)
        self._show_update_available_dialog(info)

    def _show_update_available_dialog(self, info: UpdateInfo) -> None:
        title = f"发现新版本 v{info.latest_version}"
        download_hint = f"安装包: {info.asset_name}" if info.asset_name else "安装包: 自动选择"
        release_notes = str(info.release_notes or "").strip()
        notes_hint = f"\n\n本次更新内容:\n{release_notes}" if release_notes else "\n\n本次更新内容:\n暂无更新说明。"
        message = (
            f"当前版本: v{info.current_version}\n"
            f"最新版本: v{info.latest_version}\n\n"
            f"来源: {info.source_name or 'DragonScience'}\n"
            f"{download_hint}"
            f"{notes_hint}\n\n"
            "是否现在下载并重启到新版本？"
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("软件更新")
        box.setText(title)
        box.setInformativeText(message)
        update_button = box.addButton("更新", QMessageBox.AcceptRole)
        box.addButton("稍后", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() == update_button:
            self.statusBar().showMessage(f"正在下载 v{info.latest_version}...", 3000)
            self._download_and_install_update(info)

    def _set_update_available_indicator(self, info: UpdateInfo | None, *, enabled: bool = True) -> None:
        self._available_update_info = info
        button = getattr(self, "update_available_button", None)
        if button is None:
            return
        has_update = info is not None and bool(info.update_available)
        button.setVisible(has_update)
        button.setEnabled(bool(enabled))
        if has_update:
            button.setToolTip(f"发现新版本 v{info.latest_version}，点击更新")
        else:
            button.setToolTip("发现新版本，点击更新")

    def _on_update_check_failed(self, message: str, manual: bool) -> None:
        self._finish_update_check()
        if manual:
            self.statusBar().showMessage(f"软件更新检查失败: {message}", 5000)
            QMessageBox.warning(self, "软件更新检查失败", message)
        else:
            self.statusBar().showMessage(f"自动检查软件更新失败: {message}", 5000)

    def _finish_update_check(self) -> None:
        self._checking_for_updates = False
        update_button = getattr(self, "update_button", None)
        if update_button is not None:
            update_button.setEnabled(True)

    def _clear_update_check_worker(self) -> None:
        self._update_thread = None
        self._update_worker = None

    def _download_and_install_update(self, info: UpdateInfo) -> None:
        if self._update_download_thread is not None:
            self.statusBar().showMessage("正在下载软件更新...", 3000)
            return
        if not info.download_url:
            self._open_update_page(info)
            return

        self._show_update_progress(info)
        update_button = getattr(self, "update_button", None)
        if update_button is not None:
            update_button.setEnabled(False)
        self._set_update_available_indicator(info, enabled=False)

        thread = QThread(self)
        worker = UpdateDownloadWorker(info)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_update_download_progress)
        worker.finished.connect(self._on_update_download_finished)
        worker.failed.connect(self._on_update_download_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_update_download_worker)
        self._update_download_thread = thread
        self._update_download_worker = worker
        thread.start()

    def _show_update_progress(self, info: UpdateInfo) -> None:
        dialog = QProgressDialog(f"正在下载 v{info.latest_version}...", None, 0, 100, self)
        dialog.setWindowTitle("软件更新")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        dialog.setCancelButton(None)
        dialog.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #9ca3af;
                border-radius: 4px;
                text-align: center;
                background: #f3f4f6;
            }
            QProgressBar::chunk {
                background: #22c55e;
                border-radius: 3px;
            }
            """
        )
        self._update_progress_dialog = dialog
        dialog.show()

    def _on_update_download_progress(self, downloaded: int, total: int) -> None:
        dialog = self._update_progress_dialog
        if dialog is None:
            return
        if total > 0:
            dialog.setRange(0, 100)
            value = max(0, min(100, int(downloaded * 100 / total)))
            dialog.setValue(value)
            dialog.setLabelText(
                f"正在下载软件更新... {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
            )
        else:
            dialog.setRange(0, 0)
            dialog.setLabelText(f"正在下载软件更新... {downloaded / 1024 / 1024:.1f} MB")

    def _on_update_download_finished(self, info: UpdateInfo, path: str) -> None:
        dialog = self._update_progress_dialog
        if dialog is not None:
            dialog.setRange(0, 100)
            dialog.setValue(100)
            dialog.setLabelText("下载完成，正在安装更新...")
        self.statusBar().showMessage(
            f"已下载 v{info.latest_version}，正在安装更新，请稍后重新打开软件。",
            3000,
        )
        QTimer.singleShot(800, lambda: self._launch_downloaded_update(path))

    def _on_update_download_failed(self, message: str) -> None:
        dialog = self._update_progress_dialog
        if dialog is not None:
            dialog.close()
            self._update_progress_dialog = None
        update_button = getattr(self, "update_button", None)
        if update_button is not None:
            update_button.setEnabled(True)
        if self._available_update_info is not None:
            self._set_update_available_indicator(self._available_update_info, enabled=True)
        QMessageBox.warning(self, "软件更新失败", message)

    def _clear_update_download_worker(self) -> None:
        self._update_download_thread = None
        self._update_download_worker = None

    def _launch_downloaded_update(self, path: str) -> None:
        try:
            launch_update_and_exit(Path(path))
        except UpdateDownloadError as exc:
            if self._update_progress_dialog is not None:
                self._update_progress_dialog.close()
                self._update_progress_dialog = None
            QMessageBox.warning(self, "软件更新失败", str(exc))
            update_button = getattr(self, "update_button", None)
            if update_button is not None:
                update_button.setEnabled(True)
            if self._available_update_info is not None:
                self._set_update_available_indicator(self._available_update_info, enabled=True)
            return
        if self._update_progress_dialog is not None:
            self._update_progress_dialog.close()
            self._update_progress_dialog = None
        killer = threading.Timer(5.0, lambda: os._exit(0))
        killer.daemon = True
        killer.start()
        for widget in QApplication.topLevelWidgets():
            widget.close()
        QApplication.exit(0)

    def _open_update_page(self, info: UpdateInfo) -> None:
        url = info.download_url or info.release_url
        if not url:
            QMessageBox.warning(self, "软件更新", "没有可打开的下载链接。")
            return
        QDesktopServices.openUrl(QUrl(url))

    @staticmethod
    def _default_import_directory() -> str:
        desktop = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        if desktop and Path(desktop).is_dir():
            return desktop
        desktop_path = Path.home() / "Desktop"
        if desktop_path.is_dir():
            return str(desktop_path)
        return str(Path.home())

    def _read_import_directory(self) -> str:
        saved = self.settings.value("import_directory", "", type=str)
        if saved and Path(saved).is_dir():
            return str(Path(saved))
        return self._default_import_directory()

    def _write_import_directory_setting(self, path: str | Path) -> None:
        directory = Path(path)
        if directory.is_file():
            directory = directory.parent
        if directory.is_dir():
            self.settings.setValue("import_directory", str(directory))

    def _collect_fit_params(self):
        angle_min = self.slider_min.get()
        angle_max = self.slider_max.get()
        fit_peak_indices = self._selected_peak_indices_in_fit_range(angle_min, angle_max)
        return {
            "source": self.source_var.get(),
            "mu_centers": [self.peak_mu_sliders[i].get() for i in fit_peak_indices],
            "angle_min": angle_min,
            "angle_max": angle_max,
            "d_min": float(getattr(self, "particle_size_min", 0.5)),
            "d_max": float(getattr(self, "particle_size_max", 100.0)),
            "alpha": float(self.slider_alpha.get()),
            "instrument_fwhm": float(getattr(self, "instrument_fwhm", 0.0)),
            "active_peak_indices": list(fit_peak_indices),
            "baseline_state": self._current_manual_baseline_state(),
        }

    def _set_default_import_range_and_peak(self):
        """Set import defaults and place peak 1 at the strongest point in range."""
        if not self.data_loaded:
            return

        angle_min = DEFAULT_ANGLE_MIN
        angle_max = DEFAULT_ANGLE_MAX
        self.slider_min.set(angle_min)
        self.slider_max.set(angle_max)
        self._refresh_peak_slider_bounds()
        self._save_current_analysis_state()

        x = np.asarray(self.x_data, dtype=float)
        y = np.asarray(self.y_data, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x >= angle_min) & (x <= angle_max)
        if not np.any(mask):
            return

        local_x = x[mask]
        local_y = y[mask]
        peak_center = float(local_x[int(np.argmax(local_y))])

        for peak_idx in self.active_peak_indices:
            if peak_idx == 0 and peak_idx < len(self.peak_mu_sliders):
                self.peak_mu_sliders[peak_idx].set(peak_center)
                break
        self._save_current_peak_states()
        self._save_current_analysis_state()

    def load_file(self):
        """Open BET-style import dialog and load selected XRD files."""
        existing = [sample.path for sample in self.samples]
        dialog = XRDFileImportDialog(
            self,
            initial_dir=self.import_directory,
            existing_paths=existing,
            available_sort=self._import_available_sort,
        )
        if dialog.exec_() != XRDFileImportDialog.Accepted:
            self._import_available_sort = dialog.available_sort()
            return
        paths = dialog.selected_paths()
        if not paths:
            self._import_available_sort = dialog.available_sort()
            return
        self.import_directory = str(dialog.current_directory)
        self._import_available_sort = dialog.available_sort()
        self._write_import_directory_setting(self.import_directory)
        self.sync_files(paths)

    @staticmethod
    def _path_key(path: str | Path) -> str:
        try:
            return str(Path(path).resolve()).lower()
        except OSError:
            return str(path).lower()

    def _load_sample_from_path(self, path: str | Path) -> XRDSample:
        x, y, name, meta = load_xrd_file(str(path))
        meta["file_name"] = os.path.splitext(os.path.basename(str(path)))[0]
        return XRDSample(
            path=str(path),
            x_data=x,
            y_data=y,
            name=name,
            metadata=meta,
            peak_states=self._default_peak_states(),
        )

    def sync_files(self, paths: list[str]) -> None:
        """Make the sample list match the import dialog's selected files."""
        self._save_current_peak_states()
        self._save_current_analysis_state()
        self._save_current_manual_baseline_state()
        self._save_current_marker_label_state()
        self._save_current_plot_view_state()

        existing_by_key = {self._path_key(sample.path): sample for sample in self.samples}
        active_sample = self.samples[self.active_sample_index] if 0 <= self.active_sample_index < len(self.samples) else None
        new_samples: list[XRDSample] = []
        seen_keys: set[str] = set()
        errors = []

        for path in paths:
            key = self._path_key(path)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            existing = existing_by_key.get(key)
            if existing is not None:
                new_samples.append(existing)
                continue
            try:
                new_samples.append(self._load_sample_from_path(path))
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")

        if errors:
            messagebox.showwarning("文件读取错误", "部分文件无法加载：\n" + "\n".join(errors[:8]))
        if not new_samples:
            return

        target_index = new_samples.index(active_sample) if active_sample in new_samples else 0
        self.samples = new_samples
        self.active_sample_index = -1
        self.data_loaded = False
        self.results_ready = False
        self.refresh_sample_table()
        self.select_sample(target_index)
        self.statusBar().showMessage(f"样品列表已同步：{len(new_samples)} 个 XRD 文件", 4000)

    def load_files(self, paths: list[str]):
        """Append a batch of files into the sample table, used by drag-and-drop."""
        existing = {str(Path(sample.path).resolve()).lower() for sample in self.samples}
        loaded = 0
        errors = []
        for path in paths:
            key = str(Path(path).resolve()).lower()
            if key in existing:
                continue
            try:
                self.samples.append(self._load_sample_from_path(path))
                existing.add(key)
                loaded += 1
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")

        self.refresh_sample_table()
        if self.samples and self.active_sample_index < 0:
            self.select_sample(0)
        elif loaded:
            self.select_sample(len(self.samples) - loaded)
        if loaded:
            self.statusBar().showMessage(f"已导入 {loaded} 个 XRD 文件", 4000)

        if errors:
            messagebox.showwarning("文件读取错误", "部分文件无法加载：\n" + "\n".join(errors[:8]))

    def refresh_sample_table(self):
        compare_col = getattr(self, "sample_compare_col", 0)
        file_col = getattr(self, "sample_file_col", 1)
        status_col = getattr(self, "sample_status_col", 2)
        self.sample_table.blockSignals(True)
        self.sample_table.setRowCount(len(self.samples))
        for row, sample in enumerate(self.samples):
            compare_item = QTableWidgetItem()
            compare_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            compare_item.setCheckState(Qt.Checked if getattr(sample, "compare_visible", True) else Qt.Unchecked)
            compare_item.setTextAlignment(Qt.AlignCenter)
            compare_item.setToolTip("在对比分析中显示" if getattr(sample, "compare_visible", True) else "在对比分析中隐藏")

            file_item = QTableWidgetItem(Path(sample.path).stem)
            file_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            file_item.setForeground(QBrush(QColor("#111827")))
            file_item.setToolTip(sample.path)
            status_item = QTableWidgetItem("")
            status_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            status_item.setIcon(self._status_icon(sample.status))
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setToolTip("计算完成" if sample.status == "complete" else "待计算")
            self.sample_table.setItem(row, compare_col, compare_item)
            self.sample_table.setItem(row, file_col, file_item)
            old_widget = self.sample_table.cellWidget(row, status_col)
            if old_widget is not None:
                self.sample_table.removeCellWidget(row, status_col)
                old_widget.deleteLater()
            self.sample_table.setItem(row, status_col, status_item)
        self.sample_table.blockSignals(False)
        if 0 <= self.active_sample_index < len(self.samples):
            self.sample_table.selectRow(self.active_sample_index)
        self._sync_compare_select_all_state()
        QTimer.singleShot(0, self._position_compare_select_all_check)
        self._update_comparison_plots_if_available()

    def _on_sample_table_item_changed(self, item: QTableWidgetItem):
        compare_col = getattr(self, "sample_compare_col", 0)
        if item is None or item.column() != compare_col:
            return
        row = item.row()
        if not (0 <= row < len(self.samples)):
            return
        self.samples[row].compare_visible = item.checkState() == Qt.Checked
        item.setToolTip("在对比分析中显示" if self.samples[row].compare_visible else "在对比分析中隐藏")
        self._sync_compare_select_all_state()
        self._update_comparison_plots_if_available()

    def _on_compare_select_all_changed(self, state: int):
        if getattr(self, "_updating_compare_checks", False):
            return
        if state == Qt.PartiallyChecked:
            return
        checked = state == Qt.Checked
        for sample in self.samples:
            sample.compare_visible = checked
        compare_col = getattr(self, "sample_compare_col", 0)
        self.sample_table.blockSignals(True)
        for row in range(self.sample_table.rowCount()):
            item = self.sample_table.item(row, compare_col)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                item.setToolTip("在对比分析中显示" if checked else "在对比分析中隐藏")
        self.sample_table.blockSignals(False)
        self._sync_compare_select_all_state()
        self._update_comparison_plots_if_available()

    def _sync_compare_select_all_state(self) -> None:
        checkbox = getattr(self, "select_all_compare_check", None)
        if checkbox is None:
            return
        if not self.samples:
            state = Qt.Unchecked
        elif all(getattr(sample, "compare_visible", True) for sample in self.samples):
            state = Qt.Checked
        elif any(getattr(sample, "compare_visible", True) for sample in self.samples):
            state = Qt.PartiallyChecked
        else:
            state = Qt.Unchecked
        self._updating_compare_checks = True
        checkbox.setEnabled(bool(self.samples))
        checkbox.setCheckState(state)
        self._updating_compare_checks = False

    def _position_compare_select_all_check(self, *args) -> None:
        checkbox = getattr(self, "select_all_compare_check", None)
        table = getattr(self, "sample_table", None)
        if checkbox is None or table is None:
            return
        header = table.horizontalHeader()
        if header is None or not header.isVisible():
            checkbox.hide()
            return
        col = getattr(self, "sample_compare_col", 0)
        size = checkbox.size()
        if not size.isValid() or size.isEmpty():
            size = checkbox.sizeHint()
        x = header.sectionViewportPosition(col) + (header.sectionSize(col) - size.width()) // 2
        y = (header.height() - size.height()) // 2
        checkbox.setVisible(x + size.width() > 0 and x < header.width())
        checkbox.setGeometry(x, y, size.width(), size.height())

    def _update_comparison_plots_if_available(self) -> None:
        updater = getattr(self, "update_comparison_plots", None)
        if callable(updater):
            updater()

    def _sample_hover_plots(self) -> tuple[object, ...]:
        return (
            getattr(self, "compare_preview_plot", None),
            getattr(self, "compare_size_plot", None),
        )

    def _on_sample_table_row_hovered(self, row: int) -> None:
        sample_index = int(row) if 0 <= int(row) < len(self.samples) else None
        self._set_hovered_sample_row(sample_index)
        setter = getattr(self, "set_sample_curve_hover_plots", None)
        if callable(setter):
            setter(sample_index, *self._sample_hover_plots())

    def _set_hovered_sample_row(self, row: int | None) -> None:
        target = int(row) if row is not None and 0 <= int(row) < len(self.samples) else -1
        if target == getattr(self, "_hovered_sample_row", -1):
            return
        previous = getattr(self, "_hovered_sample_row", -1)
        self._hovered_sample_row = target
        self._apply_sample_row_hover(previous, False)
        self._apply_sample_row_hover(target, True)

    def _apply_sample_row_hover(self, row: int, hovered: bool) -> None:
        if row < 0 or row >= self.sample_table.rowCount():
            return
        self.sample_table.blockSignals(True)
        try:
            for column in range(self.sample_table.columnCount()):
                if column == getattr(self, "sample_compare_col", 0):
                    continue
                item = self.sample_table.item(row, column)
                if item is None:
                    continue
                base_font = item.data(Qt.UserRole + 301)
                base_foreground = item.data(Qt.UserRole + 302)
                if hovered:
                    if not isinstance(base_font, QFont):
                        item.setData(Qt.UserRole + 301, QFont(item.font()))
                    if not isinstance(base_foreground, QBrush):
                        item.setData(Qt.UserRole + 302, QBrush(item.foreground()))
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setForeground(QBrush(QColor("#111827")))
                else:
                    if isinstance(base_font, QFont):
                        item.setFont(QFont(base_font))
                        item.setData(Qt.UserRole + 301, None)
                    if isinstance(base_foreground, QBrush):
                        item.setForeground(QBrush(base_foreground))
                        item.setData(Qt.UserRole + 302, None)
        finally:
            self.sample_table.blockSignals(False)
        self.sample_table.viewport().update()

    def _select_sample_from_curve(self, row: int) -> None:
        if not (0 <= int(row) < len(self.samples)):
            return
        row = int(row)
        current_column = self.sample_table.currentColumn()
        if current_column < 0:
            current_column = getattr(self, "sample_file_col", 1)
        self.sample_table.setCurrentCell(row, current_column)
        self.sample_table.selectRow(row)
        try:
            self.sample_table.scrollToItem(
                self.sample_table.item(row, getattr(self, "sample_file_col", 1)),
                QtWidgets.QAbstractItemView.PositionAtCenter,
            )
        except Exception:
            pass
        self.statusBar().showMessage(f"已切换到样品：{self._sample_display_name(self.samples[row])}", 2200)

    def _status_icon(self, status: str) -> QIcon:
        return self._complete_icon() if status == "complete" else self._pending_icon()

    def _status_cell_widget(self, status: str) -> QWidget:
        widget = QWidget(self.sample_table)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label = QLabel(widget)
        label.setPixmap(self._status_icon(status).pixmap(18, 18))
        label.setAlignment(Qt.AlignCenter)
        label.setToolTip("计算完成" if status == "complete" else "待计算")
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def _pending_icon(self) -> QIcon:
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f5c542"))
        painter.drawEllipse(2, 2, 16, 16)
        painter.setBrush(QColor("#ffffff"))
        for x in (7, 10, 13):
            painter.drawEllipse(x - 1, 10 - 1, 2, 2)
        painter.end()
        return QIcon(pixmap)

    def _complete_icon(self) -> QIcon:
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#2fb344"))
        painter.drawEllipse(2, 2, 16, 16)
        painter.setPen(QColor("#ffffff"))
        pen = painter.pen()
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(6, 10)
        path.lineTo(9, 13)
        path.lineTo(14, 7)
        painter.drawPath(path)
        painter.end()
        return QIcon(pixmap)

    def _on_sample_table_current_changed(self, current_row, _current_col, _previous_row, _previous_col):
        if 0 <= current_row < len(self.samples):
            self.select_sample(current_row)

    def select_sample(self, index: int):
        if not (0 <= index < len(self.samples)):
            return
        if self.active_sample_index == index and self.data_loaded:
            return
        self._save_current_peak_states()
        self._save_current_analysis_state()
        self._save_current_manual_baseline_state()
        self._save_current_marker_label_state()
        self._save_current_plot_view_state()
        self.active_sample_index = index
        sample = self.samples[index]
        self.x_data = sample.x_data
        self.y_data = sample.y_data
        self.data_loaded = True
        self.current_file_name = sample.name
        self.current_metadata = sample.metadata
        if not sample.peak_states:
            sample.peak_states = self._default_peak_states()
        if sample.analysis_state:
            self._apply_analysis_state(sample.analysis_state)
        elif sample.results and "x_segment" in sample.results:
            x_segment = np.asarray(sample.results["x_segment"], dtype=float)
            if x_segment.size:
                self._apply_analysis_state(
                    {"angle_min": float(np.nanmin(x_segment)), "angle_max": float(np.nanmax(x_segment))}
                )
        else:
            self._apply_analysis_state({"angle_min": DEFAULT_ANGLE_MIN, "angle_max": DEFAULT_ANGLE_MAX})
        self._apply_peak_states(sample.peak_states)
        self._apply_manual_baseline_state(sample.baseline_state)
        self._apply_marker_label_state(sample.marker_label_state)
        self._apply_plot_view_state(sample.plot_view_state)
        self.update_info_panel(sample.metadata)
        if sample.results:
            self._restore_sample_results(sample)
            self.update_preview(None)
            self.update_multi_peak_plots()
            self._restore_sample_plot_view_state()
            self.update_result_table()
        else:
            self.results_ready = False
            self.clear_result_table()
            if not sample.analysis_state:
                self._set_default_import_range_and_peak()
            self.update_preview(None)
        self._save_current_analysis_state()

    def _store_current_sample_results(self):
        if not (0 <= self.active_sample_index < len(self.samples)):
            return
        sample = self.samples[self.active_sample_index]
        sample.status = "complete"
        sample.analysis_state = self._current_analysis_state()
        sample.baseline_state = self._current_manual_baseline_state()
        sample.marker_label_state = self._current_marker_label_state()
        sample.plot_view_state = self._current_plot_view_state()
        sample.results = {
            "best_f_total": self.best_f_total,
            "all_basis_k1": self.all_basis_k1,
            "all_basis_k2": self.all_basis_k2,
            "D_range": self.D_range,
            "all_peak_info": self.all_peak_info,
            "global_max_component_area": self.global_max_component_area,
            "result_active_peak_indices": self.result_active_peak_indices,
            "x_segment": self.x_segment,
            "y_segment_raw": self.y_segment_raw,
            "y_segment": self.y_segment,
            "background": self.background,
        }
        self.ui(self.refresh_sample_table)
        self.ui(self._update_comparison_plots_if_available)

    def _restore_sample_results(self, sample: XRDSample):
        for key, value in sample.results.items():
            setattr(self, key, value)
        self._apply_marker_label_state(sample.marker_label_state)
        self._apply_plot_view_state(sample.plot_view_state)
        self.results_ready = True

    def compute_thread(self, mode: str = "fine"):
        """在后台线程中启动拟合计算，避免阻塞 UI。"""
        if not self.data_loaded:
            messagebox.showwarning("提示", "请先导入数据。")
            return
        if not self.active_peak_indices:
            messagebox.showwarning("提示", "请至少选择一个峰。")
            return

        params = self._collect_fit_params()
        if not params["mu_centers"]:
            messagebox.showwarning("提示", "当前蓝色拟合范围内没有峰，请先在范围内添加或移动峰。")
            return

        self.stop_flag.clear()
        self.progress_label.show()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.ui_set(self.progress_var, "计算中...")

        for btn in (self.btn_fast, self.btn_fine):
            btn.setEnabled(False)

        threading.Thread(target=self.compute_fit, args=(mode, params), daemon=True).start()

    def compute_fit(self, mode: str = "fine", params: dict | None = None):
        """执行多峰拟合（子线程）。"""
        try:
            params = params or {}
            source = params["source"]
            lam1, lam2 = WAVELENGTHS.get(source, WAVELENGTHS["Cu"])
            mu_centers = list(params["mu_centers"])
            if not mu_centers:
                self.ui(messagebox.showwarning, "提示", "当前蓝色拟合范围内没有峰，请先在范围内添加或移动峰。")
                self.ui_set(self.progress_var, "计算失败")
                return

            angle_min = params["angle_min"]
            angle_max = params["angle_max"]
            mask = (self.x_data >= angle_min) & (self.x_data <= angle_max)
            x = self.x_data[mask]
            y_raw = self.y_data[mask]
            if len(x) < 2:
                self.ui(messagebox.showwarning, "提示", "当前角度范围内没有足够的数据点。")
                self.ui_set(self.progress_var, "计算失败")
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
            raw_pts = int((d_max - d_min) / 0.1)
            num_pts = min(800, max(200, raw_pts))
            self.D_range = np.linspace(d_min, d_max, num_pts)
            L_single = build_regularization_matrix(len(self.D_range))

            alpha_val = float(params["alpha"])
            inst_fwhm = float(params["instrument_fwhm"])

            if mode == "fast":
                halfwidth, steps = 0.0, 1
            else:
                halfwidth, steps = 0.1, 11

            total = max(1, len(mu_centers) * steps)
            done = 0
            best_mu = list(mu_centers)

            for i in range(len(best_mu)):
                if self.stop_flag.is_set():
                    self.ui_set(self.progress_var, "已停止")
                    return

                center = best_mu[i]
                low = max(center - halfwidth, angle_min)
                high = min(center + halfwidth, angle_max)
                if high <= low:
                    low, high = center - 1e-4, center + 1e-4

                candidates = np.linspace(low, high, steps)
                best_loss = None
                best_val = center
                max_workers = min(4, os.cpu_count() or 1)

                args_common = (
                    tuple(best_mu),
                    i,
                    x,
                    y_scaled,
                    lam1,
                    lam2,
                    INTENSITY_RATIO,
                    L_single,
                    self.D_range,
                    alpha_val,
                    inst_fwhm,
                )

                ex = ProcessPoolExecutor(max_workers=max_workers)
                futs = [ex.submit(_eval_candidate_for_index, mu, *args_common) for mu in candidates]
                pending = set(futs)
                stopped = False
                try:
                    while pending:
                        if self.stop_flag.is_set():
                            stopped = True
                            self._stop_process_pool(ex, pending)
                            self.ui_set(self.progress_var, "已停止")
                            return

                        finished, pending = wait(
                            pending,
                            timeout=0.05,
                            return_when=FIRST_COMPLETED,
                        )
                        for fut in finished:
                            try:
                                loss, mu_val = fut.result()
                            except Exception:
                                loss, mu_val = np.inf, None

                            done += 1
                            pct = int(done * 100 / total)
                            self.ui(self.progress_bar.setValue, pct)
                            self.ui_set(
                                self.progress_var,
                                f"扫描峰 {i + 1}/{len(best_mu)}：{pct}%",
                            )

                            if mu_val is not None and (best_loss is None or loss < best_loss):
                                best_loss, best_val = loss, mu_val
                finally:
                    if not stopped:
                        ex.shutdown(wait=False, cancel_futures=True)

                best_mu[i] = best_val

            if self.stop_flag.is_set():
                self.ui_set(self.progress_var, "已停止")
                return

            ex = ProcessPoolExecutor(max_workers=1)
            fut = ex.submit(
                fit_with_mu_list,
                x,
                y_scaled,
                best_mu,
                lam1,
                lam2,
                L_single,
                self.D_range,
                alpha_val,
                instrument_fwhm_deg=inst_fwhm,
            )
            stopped = False
            try:
                while not fut.done():
                    if self.stop_flag.is_set():
                        stopped = True
                        self._stop_process_pool(ex, (fut,))
                        self.ui_set(self.progress_var, "已停止")
                        return
                    wait((fut,), timeout=0.05)
                resid, f_total, basis_k1_list, basis_k2_list = fut.result()
            finally:
                if not stopped:
                    ex.shutdown(wait=False, cancel_futures=True)

            if f_total is None:
                self.ui_set(self.progress_var, "拟合失败：解全为零")
                return

            self.best_f_total = f_total
            self.all_basis_k1 = basis_k1_list
            self.all_basis_k2 = basis_k2_list
            self.result_active_peak_indices = list(params["active_peak_indices"])

            for peak_idx, mu in zip(params["active_peak_indices"], best_mu):
                if peak_idx < len(self.peak_mu_sliders):
                    self.ui(self.peak_mu_sliders[peak_idx].set, mu)

            self.ui(self._hide_axes0_overlays)

            self.x_segment = x
            self.y_segment_raw = y_raw
            self.y_segment = y
            self.background = background

            self.process_multi_peak_results(self.result_active_peak_indices)
            self.ui_set(self.progress_var, "拟合成功！")

        except Exception as exc:
            self.ui(messagebox.showwarning, "提示", f"计算过程中发生错误: {exc}")
            self.ui_set(self.progress_var, "计算失败")
        finally:
            for btn in (self.btn_fast, self.btn_fine):
                self.ui(btn.setEnabled, True)

    def stop_compute(self):
        """中断正在运行的计算。"""
        self.stop_flag.set()
        self.ui_set(self.progress_var, "正在停止...")

    def process_multi_peak_results(self, active_peak_indices=None):
        """调用 core/analysis 后处理 NNLS 结果，然后更新图表。"""
        active_peak_indices = list(active_peak_indices or self.active_peak_indices)
        self.all_peak_info, self.global_max_component_area = build_all_peak_info(
            self.best_f_total,
            active_peak_indices,
            self.D_range,
            self.peak_colors,
            self.all_basis_k1,
            self.all_basis_k2,
        )
        self.result_active_peak_indices = active_peak_indices
        self.results_ready = True
        self._store_current_sample_results()
        self.ui(self.update_multi_peak_plots)
        self.ui(self.update_result_table)
        self.ui_set(self.progress_var, "拟合成功！")

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
            active_indices = getattr(self, "result_active_peak_indices", self.active_peak_indices)
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)

                A_total_sum = max(
                    sum(
                        float(det.get("area", 0.0))
                        for info in self.all_peak_info
                        for det in info.get("peak_details", [])
                    ),
                    1e-12,
                )

                global_f_sum = sum(
                    np.asarray(info["f_segment"], dtype=float) for info in self.all_peak_info
                )
                global_total_Y = global_f_sum / A_total_sum

                scaled_curves = [
                    (
                        active_indices[i] + 1,
                        np.asarray(info["f_segment"], dtype=float) / A_total_sum,
                    )
                    for i, info in enumerate(self.all_peak_info)
                ]

                dist_header = ["Global_Total_D(nm)", "Global_Total_Y(PDF)", ""]
                for peak_id, _ in scaled_curves:
                    dist_header += [f"Peak{peak_id}_D(nm)", f"Peak{peak_id}_Y(PDF)", ""]

                x = self.x_segment
                bg = self.background
                y_corr = getattr(self, "y_corr", None)
                y = self.y_segment if y_corr is None else y_corr
                y_raw = getattr(self, "y_segment_raw", y + bg)

                total_fit = np.zeros_like(x)
                peak_fits = []
                comp_fits_by_peak = []

                for info in self.all_peak_info:
                    f_seg = info["f_segment"]
                    basis_k1 = info["basis_k1"]
                    basis_k2 = info["basis_k2"]
                    fit_peak = (basis_k1.dot(f_seg) + basis_k2.dot(f_seg)) * y.max()
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
                            (
                                basis_k1[:, idx].dot(f_comp[idx])
                                + basis_k2[:, idx].dot(f_comp[idx])
                            )
                            * y.max()
                        )
                    comp_fits_by_peak.append(comps)

                total_fit_out = total_fit + bg
                peak_fits_out = [pf + bg for pf in peak_fits]
                comp_fits_out = []
                comp_headers = []
                for i, comps in enumerate(comp_fits_by_peak):
                    peak_id = active_indices[i] + 1
                    for j, det in enumerate(self.all_peak_info[i]["peak_details"]):
                        comp_headers.append(f"P{peak_id}_Comp{j+1}@{det['center']:.2f}nm")
                        c = comps[j]
                        comp_fits_out.append(
                            np.full_like(x, np.nan, dtype=float)
                            if np.isnan(c).all()
                            else c + bg
                        )

                left_header = ["2θ (deg)", "Raw Data", "Background", "Total Fit"]
                left_header += [
                    f"Peak_{active_indices[i]+1}_Contribution"
                    for i in range(len(self.all_peak_info))
                ]
                left_header += comp_headers

                writer.writerow(dist_header + [""] + left_header)

                n_dist = len(self.D_range)
                n_left = len(x)
                n_rows = max(n_dist, n_left)

                dist_cols = [
                    (
                        [f"{d:.4f}" for d in self.D_range],
                        [f"{v:.6f}" for v in global_total_Y],
                    )
                ]
                for _, curve in scaled_curves:
                    dist_cols.append(
                        (
                            [f"{d:.4f}" for d in self.D_range],
                            [f"{v:.6f}" for v in curve],
                        )
                    )

                left_cols = [
                    [f"{v:.4f}" for v in x],
                    [f"{v:.2f}" for v in y_raw],
                    [f"{v:.2f}" for v in bg],
                    [f"{v:.2f}" for v in total_fit_out],
                ]
                for pf in peak_fits_out:
                    left_cols.append([f"{v:.2f}" for v in pf])
                for cf in comp_fits_out:
                    left_cols.append(["" if np.isnan(v) else f"{v:.2f}" for v in cf])

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

    def closeEvent(self, event):
        self.stop_flag.set()
        if self._update_thread is not None and self._update_thread.isRunning():
            self._update_thread.quit()
            self._update_thread.wait(5500)
        if self._update_download_thread is not None and self._update_download_thread.isRunning():
            self._update_download_thread.quit()
            if not self._update_download_thread.wait(2500):
                self._update_download_thread.terminate()
                self._update_download_thread.wait(1200)
        if self._update_progress_dialog is not None:
            self._update_progress_dialog.close()
            self._update_progress_dialog = None
        event.accept()
