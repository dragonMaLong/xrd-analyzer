"""
ui/app_window.py
----------------
PyQt5 XRDApp 主类：
  - 继承三个 Mixin，统一协调左侧面板、右侧图表和 L-Curve 功能
  - 管理数据加载、计算线程、结果后处理和 CSV 导出
"""
import csv
import os
import tempfile
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
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

from ..core.analysis import build_all_peak_info, calculate_rfit_percent
from ..core.fitting import (
    INTENSITY_RATIO,
    WAVELENGTHS,
    _eval_candidate_chunk_for_index,
    _eval_candidate_for_index,
    build_regularization_matrix,
    fit_with_mu_list,
    solve_regularized_from_basis,
)
from ..io.file_reader import load_file as load_xrd_file
from ..io.project_file import (
    ALGORITHM_VERSION,
    PROJECT_EXTENSION,
    ProjectFormatError,
    data_sha256,
    file_sha256,
    load_project,
    materialize_project_snapshot,
    save_project,
    stable_state_sha256,
)
from ..update_checker import DEFAULT_UPDATE_REPOSITORY, UpdateInfo, check_for_update
from ..updater import UpdateDownloadError, download_update, launch_update_and_exit
from ..utils import resource_path
from ..version import __version__
from .control_panel_mixin import ControlPanelMixin, SAMPLE_STATUS_PROGRESS_ROLE
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
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_fingerprint: str = ""
    file_fingerprint: str = ""
    status: str = "pending"
    compare_visible: bool = True
    parameter_state: dict = field(default_factory=dict)
    peak_states: list[dict] = field(default_factory=list)
    analysis_state: dict = field(default_factory=dict)
    baseline_state: dict = field(default_factory=dict)
    marker_label_state: dict = field(default_factory=dict)
    plot_view_state: dict = field(default_factory=dict)
    size_visibility_state: dict = field(default_factory=dict)
    size_total_inclusion_state: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    runtime_plot_cache: dict = field(default_factory=dict, repr=False)
    result_signature: str = ""
    result_is_current: bool = False
    project_path: str = ""
    project_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_dirty: bool = True
    project_revision: int = 0


class _UiDispatcher(QObject):
    call_requested = pyqtSignal(object, tuple, dict)

    def __init__(self):
        super().__init__()
        self.call_requested.connect(self._run)

    @pyqtSlot(object, tuple, dict)
    def _run(self, fn, args, kwargs):
        fn(*args, **kwargs)


class _ImmediateFuture:
    def __init__(self, result=None, exception: Exception | None = None):
        self._result = result
        self._exception = exception

    def done(self) -> bool:
        return True

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result

    def cancel(self) -> bool:
        return False


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs) -> _ImmediateFuture:
        try:
            return _ImmediateFuture(result=fn(*args, **kwargs))
        except Exception as exc:
            return _ImmediateFuture(exception=exc)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        return None


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
        QApplication.setOrganizationName("DragonScience")
        QApplication.setApplicationName("XRDAnalyzer")
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

        self.max_peaks = 1
        self.peak_colors = ["#FF0000", "#0077FF", "#00C853", "#FFAB00", "#00E5FF"]
        self.active_peak_indices = []
        self.result_active_peak_indices = []
        self.peak_mu_sliders = []
        self.peak_check_vars = []
        self.peak_rows = []
        self.peak_color_buttons = []
        self.peak_visible_buttons = []
        self._building_peak_controls = False
        # The baseline is always part of the imported sample view.  Editing is
        # a separate transient tool mode controlled by the Baseline button.
        self.manual_baseline_enabled = True
        self.manual_baseline_editing = True
        self.manual_baseline_edited = False
        self.manual_baseline_user_points = []
        self.manual_baseline_endpoint_y = {"left": None, "right": None}
        self.manual_baseline_endpoint_deleted = set()
        self._manual_baseline_next_anchor_id = 1
        self._manual_baseline_curve_item = None
        self._manual_baseline_anchor_items = []
        self._syncing_manual_baseline_anchor = False
        self._manual_baseline_drag_anchor = None
        self._manual_baseline_drag_anchor_id = None
        self.particle_size_min = 0.1
        self.particle_size_max = 100.0
        self.particle_size_step = 0.1
        self.instrument_fwhm = 0.0
        self.regularization_method = "l2"
        self.peak_kernel = "pearson7"
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
        self._suspend_project_dirty = False
        self._project_save_jobs: dict[str, str] = {}
        self._file_load_jobs: dict[str, str] = {}
        self._pending_file_loads: dict[str, dict] = {}
        self._sample_status_tasks: dict[str, dict] = {}
        self._project_io_progress_generation = 0
        self._project_io_lock = threading.Lock()
        self._project_io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xrd-project-io")
        self._prepared_project_snapshot_lock = threading.Lock()
        self._prepared_project_snapshot_generation = 0
        self._prepared_project_snapshot_jobs: dict[str, int] = {}
        self._prepared_project_snapshots: dict[str, tuple[int, str, str]] = {}
        self._prepared_project_snapshot_closing = False
        snapshot_cache_root = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        if not snapshot_cache_root:
            snapshot_cache_root = tempfile.gettempdir()
        self._prepared_project_snapshot_dir = (
            Path(snapshot_cache_root) / "prepared-projects" / str(uuid.uuid4())
        )
        self._suspend_plot_updates = False
        self._restoring_sample_state = False
        self._transient_runtime_plot_cache: dict = {}
        self._sample_render_timer = QTimer(self)
        self._sample_render_timer.setSingleShot(True)
        self._sample_render_timer.timeout.connect(self._render_selected_sample)
        self._fit_cache = None
        self.fit_quality_history = []
        self.current_rfit_percent = None
        self._fit_worker_running = False
        self._fit_task_sample_id: str | None = None
        self._fit_task_token: str | None = None
        self._fit_task_operation = ""
        self._alpha_fast_running = False
        self._alpha_fast_pending = False
        self._alpha_fast_revision = 0
        self._alpha_fast_timer = QTimer(self)
        self._alpha_fast_timer.setSingleShot(True)
        self._alpha_fast_timer.setInterval(120)
        self._alpha_fast_timer.timeout.connect(self._start_alpha_fast_recompute)

        self.source_var = None

        icon_path = resource_path("logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._setup_ui()
        self._update_window_title()
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

    def _update_window_title(self) -> None:
        title = f"XRD晶粒尺寸分布分析-DragonScience V{APP_VERSION}"
        if 0 <= self.active_sample_index < len(self.samples):
            sample = self.samples[self.active_sample_index]
            dirty = " *" if sample.project_dirty else ""
            title += f" — {sample.name}{dirty}"
        self.setWindowTitle(title)

    def _mark_project_dirty(self, sample: XRDSample | None = None) -> None:
        if getattr(self, "_suspend_project_dirty", False):
            return
        if sample is None and 0 <= self.active_sample_index < len(self.samples):
            sample = self.samples[self.active_sample_index]
        if sample is None:
            return
        sample.project_dirty = True
        sample.project_revision += 1
        if hasattr(self, "_ui_dispatcher"):
            self.ui(self._update_window_title)

    def _set_project_clean(self, sample: XRDSample | None = None) -> None:
        if sample is None and 0 <= self.active_sample_index < len(self.samples):
            sample = self.samples[self.active_sample_index]
        if sample is not None:
            sample.project_dirty = False
        self._update_window_title()

    def _current_parameter_state(self) -> dict:
        source = self.source_var.get() if getattr(self, "source_var", None) is not None else "Cu"
        alpha = self.slider_alpha.get() if hasattr(self, "slider_alpha") else 1.0
        return {
            "source": str(source),
            "alpha": float(alpha),
            "particle_size_min": float(getattr(self, "particle_size_min", 0.1)),
            "particle_size_max": float(getattr(self, "particle_size_max", 100.0)),
            "particle_size_step": float(getattr(self, "particle_size_step", 0.1)),
            "instrument_fwhm": float(getattr(self, "instrument_fwhm", 0.0)),
            "regularization_method": str(getattr(self, "regularization_method", "l2") or "l2"),
            "peak_kernel": str(getattr(self, "peak_kernel", "pearson7") or "pearson7"),
            "size_distribution_mode": str(getattr(self, "size_distribution_mode", "volume") or "volume"),
        }

    def _apply_parameter_state(self, state: dict | None) -> None:
        state = dict(state or {})
        self.particle_size_min = float(state.get("particle_size_min", 0.1))
        self.particle_size_max = float(state.get("particle_size_max", 100.0))
        self.particle_size_step = float(state.get("particle_size_step", 0.1))
        self.instrument_fwhm = float(state.get("instrument_fwhm", 0.0))
        self.regularization_method = str(state.get("regularization_method", "l2") or "l2")
        self.peak_kernel = str(state.get("peak_kernel", "pearson7") or "pearson7")
        self.size_distribution_mode = (
            "number" if str(state.get("size_distribution_mode", "volume")).lower() == "number" else "volume"
        )
        if getattr(self, "source_var", None) is not None:
            combo = getattr(self, "source_menu", None)
            if combo is not None:
                combo.blockSignals(True)
            self.source_var.set(state.get("source", "Cu"))
            if combo is not None:
                combo.blockSignals(False)
        if hasattr(self, "slider_alpha"):
            self.slider_alpha.set(float(state.get("alpha", 1.0)), emit=False)

    def _save_current_parameter_state(self) -> None:
        if not (0 <= self.active_sample_index < len(self.samples)):
            return
        sample = self.samples[self.active_sample_index]
        state = self._current_parameter_state()
        if sample.parameter_state != state:
            computation_keys = {
                "source",
                "alpha",
                "particle_size_min",
                "particle_size_max",
                "particle_size_step",
                "instrument_fwhm",
                "regularization_method",
                "peak_kernel",
            }
            calculation_changed = any(
                sample.parameter_state.get(key) != state.get(key) for key in computation_keys
            )
            sample.parameter_state = state
            if calculation_changed:
                sample.result_is_current = False
            self._mark_project_dirty()

    @staticmethod
    def _calculation_peak_states(states: list[dict]) -> list[dict]:
        return [
            {
                "checked": bool(item.get("checked", True)),
                "value": round(float(item.get("value", 0.0)), 8),
            }
            for item in states
        ]

    def _result_signature_for_sample(self, sample: XRDSample) -> str:
        calculation_parameters = {
            key: value
            for key, value in sample.parameter_state.items()
            if key != "size_distribution_mode"
        }
        return stable_state_sha256(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "data_sha256": sample.data_fingerprint or data_sha256(sample.x_data, sample.y_data),
                "parameters": calculation_parameters,
                "analysis": sample.analysis_state,
                "peaks": self._calculation_peak_states(sample.peak_states),
                "baseline": sample.baseline_state,
            }
        )

    @staticmethod
    def _compact_fit_curve_snapshot_from_data(curve_data: dict | None) -> dict | None:
        if not isinstance(curve_data, dict):
            return None
        peaks = []
        for peak_spec in curve_data.get("peak_specs", []) or []:
            try:
                peak_id = int(peak_spec["peak_id"])
                peak_signal = np.asarray(peak_spec["signal"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            components = []
            for component in peak_spec.get("components", []) or []:
                try:
                    components.append(
                        {
                            "detail_index": int(component["detail_index"]),
                            "signal": np.asarray(component["signal"], dtype=float),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            peaks.append(
                {
                    "peak_id": peak_id,
                    "signal": peak_signal,
                    "components": components,
                }
            )
        if not peaks:
            return None
        return {"version": 1, "peaks": peaks}

    @classmethod
    def _compact_fit_curve_snapshot_from_results(cls, results: dict) -> dict | None:
        existing = results.get("fit_curve_snapshot")
        if isinstance(existing, dict) and existing.get("peaks"):
            return existing
        try:
            x = np.asarray(results["x_segment"], dtype=float)
            y = np.asarray(results["y_segment"], dtype=float)
            peak_infos = list(results.get("all_peak_info") or [])
            active_indices = list(results.get("result_active_peak_indices") or [])
            all_basis_k1 = list(results.get("all_basis_k1") or [])
            all_basis_k2 = list(results.get("all_basis_k2") or [])
        except (KeyError, TypeError, ValueError):
            return None
        if x.size == 0 or not peak_infos:
            return None
        y_scale = float(np.nanmax(y)) if y.size else 1.0
        curve_specs = []
        for i, info in enumerate(peak_infos):
            try:
                peak_id = int(info.get("peak_id", active_indices[i] if i < len(active_indices) else i))
                f_segment = np.asarray(info["f_segment"], dtype=float)
                basis_k1 = np.asarray(
                    info.get("basis_k1", all_basis_k1[i] if i < len(all_basis_k1) else None),
                    dtype=float,
                )
                basis_k2 = np.asarray(
                    info.get("basis_k2", all_basis_k2[i] if i < len(all_basis_k2) else None),
                    dtype=float,
                )
                if basis_k1.ndim != 2 or basis_k2.ndim != 2:
                    return None
                peak_signal = (basis_k1.dot(f_segment) + basis_k2.dot(f_segment)) * y_scale
            except (KeyError, TypeError, ValueError, IndexError):
                return None
            components = []
            for detail_index, detail in enumerate(info.get("peak_details", []) or []):
                indices = np.asarray(detail.get("indices", []), dtype=int)
                indices = indices[
                    (indices >= 0)
                    & (indices < f_segment.size)
                    & (indices < basis_k1.shape[1])
                    & (indices < basis_k2.shape[1])
                ]
                if indices.size == 0:
                    continue
                weights = f_segment[indices]
                component_signal = (
                    basis_k1[:, indices].dot(weights)
                    + basis_k2[:, indices].dot(weights)
                ) * y_scale
                components.append(
                    {
                        "detail_index": int(detail_index),
                        "signal": component_signal,
                    }
                )
            curve_specs.append(
                {
                    "peak_id": peak_id,
                    "signal": peak_signal,
                    "components": components,
                }
            )
        return {"version": 1, "peaks": curve_specs}

    def _sample_to_project_record(self, sample: XRDSample) -> dict:
        # Full Kα1/Kα2 basis matrices scale as peaks × angles × particle-size
        # bins and can reach hundreds of MB. Projects only need the derived
        # one-dimensional display curves; bases are rebuilt by the next fit.
        source_results = dict(sample.results or {})
        runtime_curve_data = (sample.runtime_plot_cache or {}).get("fit_curve_data")
        curve_snapshot = self._compact_fit_curve_snapshot_from_data(runtime_curve_data)
        if curve_snapshot is None:
            curve_snapshot = self._compact_fit_curve_snapshot_from_results(source_results)

        persistent_results = {}
        for key, value in source_results.items():
            if key in {"_fit_cache", "all_basis_k1", "all_basis_k2", "fit_curve_snapshot"}:
                continue
            if key == "all_peak_info":
                persistent_results[key] = [
                    {
                        info_key: info_value
                        for info_key, info_value in dict(info).items()
                        if info_key not in {"basis_k1", "basis_k2"}
                    }
                    for info in (value or [])
                ]
            else:
                persistent_results[key] = value
        if curve_snapshot is not None:
            persistent_results["fit_curve_snapshot"] = curve_snapshot
        return {
            "sample_id": str(sample.sample_id),
            "path": str(sample.path),
            "name": str(sample.name),
            "metadata": dict(sample.metadata or {}),
            "data_fingerprint": str(sample.data_fingerprint or data_sha256(sample.x_data, sample.y_data)),
            "file_fingerprint": str(sample.file_fingerprint or ""),
            "status": str(sample.status),
            "compare_visible": bool(sample.compare_visible),
            "parameter_state": dict(sample.parameter_state or {}),
            "peak_states": list(sample.peak_states or []),
            "analysis_state": dict(sample.analysis_state or {}),
            "baseline_state": dict(sample.baseline_state or {}),
            "marker_label_state": dict(sample.marker_label_state or {}),
            "plot_view_state": dict(sample.plot_view_state or {}),
            "size_visibility_state": dict(sample.size_visibility_state or {}),
            "size_total_inclusion_state": dict(sample.size_total_inclusion_state or {}),
            "result_signature": str(sample.result_signature or ""),
            "result_is_current": bool(sample.result_is_current),
            "x_data": np.asarray(sample.x_data, dtype=float),
            "y_data": np.asarray(sample.y_data, dtype=float),
            "results": persistent_results,
        }

    @staticmethod
    def _remove_prepared_snapshot_file(path: str | Path | None) -> None:
        if not path:
            return
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    def _prepared_snapshot_for_revision(
        self,
        sample_id: str,
        revision: int,
        project_uuid: str,
    ) -> str | None:
        with self._prepared_project_snapshot_lock:
            snapshot = self._prepared_project_snapshots.get(str(sample_id))
        if snapshot is None:
            return None
        snapshot_revision, snapshot_project_uuid, snapshot_path = snapshot
        if snapshot_revision != int(revision) or snapshot_project_uuid != str(project_uuid):
            return None
        if not Path(snapshot_path).is_file():
            return None
        return snapshot_path

    def _queue_prepared_project_snapshot(self, sample: XRDSample) -> None:
        """Precompress one exact calculated revision for a faster later Save."""
        if not sample.results:
            return
        sample_id = str(sample.sample_id)
        revision = int(sample.project_revision)
        project_uuid = str(sample.project_uuid)
        record = self._sample_to_project_record(sample)
        snapshot_path = self._prepared_project_snapshot_dir / f"{uuid.uuid4()}{PROJECT_EXTENSION}"

        with self._prepared_project_snapshot_lock:
            if self._prepared_project_snapshot_closing:
                return
            self._prepared_project_snapshot_generation += 1
            generation = self._prepared_project_snapshot_generation
            self._prepared_project_snapshot_jobs[sample_id] = generation

        def worker() -> None:
            error = None
            try:
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                with self._project_io_lock:
                    save_project(
                        snapshot_path,
                        [record],
                        active_sample_index=0,
                        app_version=APP_VERSION,
                        project_uuid=project_uuid,
                    )
            except Exception as exc:
                error = exc

            keep_snapshot = False
            old_snapshot_path = None
            with self._prepared_project_snapshot_lock:
                is_latest = self._prepared_project_snapshot_jobs.get(sample_id) == generation
                if is_latest:
                    self._prepared_project_snapshot_jobs.pop(sample_id, None)
                if is_latest and error is None and not self._prepared_project_snapshot_closing:
                    old_snapshot = self._prepared_project_snapshots.get(sample_id)
                    if old_snapshot is not None:
                        old_snapshot_path = old_snapshot[2]
                    self._prepared_project_snapshots[sample_id] = (
                        revision,
                        project_uuid,
                        str(snapshot_path),
                    )
                    keep_snapshot = True

            if old_snapshot_path and old_snapshot_path != str(snapshot_path):
                self._remove_prepared_snapshot_file(old_snapshot_path)
            if not keep_snapshot:
                self._remove_prepared_snapshot_file(snapshot_path)

        try:
            self._project_io_executor.submit(worker)
        except Exception:
            with self._prepared_project_snapshot_lock:
                if self._prepared_project_snapshot_jobs.get(sample_id) == generation:
                    self._prepared_project_snapshot_jobs.pop(sample_id, None)

    def _discard_prepared_project_snapshot(self, sample_id: str) -> None:
        sample_id = str(sample_id)
        with self._prepared_project_snapshot_lock:
            self._prepared_project_snapshot_jobs.pop(sample_id, None)
            snapshot = self._prepared_project_snapshots.pop(sample_id, None)
        if snapshot is not None:
            self._remove_prepared_snapshot_file(snapshot[2])

    def _close_prepared_project_snapshots(self) -> None:
        with self._prepared_project_snapshot_lock:
            self._prepared_project_snapshot_closing = True
            self._prepared_project_snapshot_jobs.clear()
            paths = [snapshot[2] for snapshot in self._prepared_project_snapshots.values()]
            self._prepared_project_snapshots.clear()
        for path in paths:
            self._remove_prepared_snapshot_file(path)
        try:
            self._prepared_project_snapshot_dir.rmdir()
        except OSError:
            pass

    @staticmethod
    def _sample_from_project_record(record: dict) -> XRDSample:
        size_visibility_state = dict(record.get("size_visibility_state") or {})
        if "size_total_inclusion_state" in record:
            size_total_inclusion_state = dict(record.get("size_total_inclusion_state") or {})
        else:
            # Before the controls were separated, hiding a Peak also removed
            # it from Total. Preserve that Total when opening an older project.
            size_total_inclusion_state = dict(size_visibility_state)
        return XRDSample(
            path=str(record.get("path") or ""),
            x_data=np.asarray(record.get("x_data", []), dtype=float),
            y_data=np.asarray(record.get("y_data", []), dtype=float),
            name=str(record.get("name") or "样品"),
            metadata=dict(record.get("metadata") or {}),
            sample_id=str(record.get("sample_id") or uuid.uuid4()),
            data_fingerprint=str(record.get("data_fingerprint") or ""),
            file_fingerprint=str(record.get("file_fingerprint") or ""),
            status=str(record.get("status") or "pending"),
            compare_visible=bool(record.get("compare_visible", True)),
            parameter_state=dict(record.get("parameter_state") or {}),
            peak_states=list(record.get("peak_states") or []),
            analysis_state=dict(record.get("analysis_state") or {}),
            baseline_state=dict(record.get("baseline_state") or {}),
            marker_label_state=dict(record.get("marker_label_state") or {}),
            plot_view_state=dict(record.get("plot_view_state") or {}),
            size_visibility_state=size_visibility_state,
            size_total_inclusion_state=size_total_inclusion_state,
            results=dict(record.get("results") or {}),
            result_signature=str(record.get("result_signature") or ""),
            result_is_current=bool(record.get("result_is_current", False)),
        )

    def _capture_active_sample_state(self) -> None:
        if not (0 <= self.active_sample_index < len(self.samples)):
            return
        self._save_current_peak_states()
        self._save_current_analysis_state()
        self._save_current_parameter_state()
        self._save_current_manual_baseline_state()
        self._save_current_marker_label_state()
        self._save_current_plot_view_state()
        self._save_current_size_visibility_state()
        self._save_current_size_total_inclusion_state()

    def _suggested_project_name(self, sample: XRDSample) -> str:
        if sample.project_path:
            base = Path(sample.project_path).stem
        elif sample.path:
            base = Path(sample.path).stem
        else:
            base = str(sample.name or "XRD工程")
        for char in '<>:"/\\|?*':
            base = base.replace(char, "_")
        return (base.strip() or "XRD工程") + PROJECT_EXTENSION

    def _project_operation_blocked_by_calculation(self) -> bool:
        running = bool(
            getattr(self, "_fit_worker_running", False)
            or getattr(self, "_alpha_fast_running", False)
        )
        if running:
            QMessageBox.information(self, "计算进行中", "请等待当前计算结束或先停止计算，再操作工程文件。")
        return running

    @staticmethod
    def _idle_sample_status_tooltip(sample: XRDSample) -> str:
        if sample.status == "complete" and sample.result_is_current:
            return "计算完成，结果与当前参数一致"
        if sample.status == "complete":
            return "已恢复上一次计算结果；当前参数或算法已变化，建议重新计算"
        return "待计算"

    def _sample_row_by_id(self, sample_id: str) -> int | None:
        for row, sample in enumerate(self.samples):
            if str(sample.sample_id) == str(sample_id):
                return row
        return None

    def _update_sample_status_cell(self, sample_id: str) -> None:
        row = self._sample_row_by_id(sample_id)
        if row is None or not hasattr(self, "sample_table"):
            return
        item = self.sample_table.item(row, getattr(self, "sample_status_col", 2))
        if item is None:
            return
        task = self._sample_status_tasks.get(str(sample_id))
        if task is not None:
            progress = max(0, min(100, int(task.get("progress", 0))))
            operation = str(task.get("operation") or "处理中")
            stage = str(task.get("stage") or operation)
            item.setData(SAMPLE_STATUS_PROGRESS_ROLE, progress)
            item.setToolTip(f"{operation}：{stage} · {progress}%")
        else:
            sample = self.samples[row]
            item.setData(SAMPLE_STATUS_PROGRESS_ROLE, None)
            item.setIcon(self._status_icon(sample.status))
            item.setToolTip(self._idle_sample_status_tooltip(sample))
        self.sample_table.viewport().update(self.sample_table.visualItemRect(item))

    def _set_sample_status_progress(
        self,
        sample_id: str,
        value: int,
        operation: str,
        stage: str = "",
        *,
        task_token: str | None = None,
    ) -> None:
        sample_id = str(sample_id)
        self._sample_status_tasks[sample_id] = {
            "progress": max(0, min(100, int(value))),
            "operation": str(operation),
            "stage": str(stage or operation),
            "task_token": str(task_token or ""),
        }
        self._update_sample_status_cell(sample_id)

    def _clear_sample_status_progress(
        self,
        sample_id: str,
        *,
        task_token: str | None = None,
    ) -> None:
        sample_id = str(sample_id)
        current = self._sample_status_tasks.get(sample_id)
        if current is None:
            return
        if task_token is not None and str(current.get("task_token") or "") != str(task_token):
            return
        self._sample_status_tasks.pop(sample_id, None)
        self._update_sample_status_cell(sample_id)

    def _complete_sample_status_progress(self, sample_id: str, task_token: str) -> None:
        current = self._sample_status_tasks.get(str(sample_id))
        if current is None or str(current.get("task_token") or "") != str(task_token):
            return
        self._set_sample_status_progress(
            sample_id,
            100,
            str(current.get("operation") or "处理中"),
            "完成",
            task_token=task_token,
        )
        QTimer.singleShot(
            180,
            lambda sid=str(sample_id), token=str(task_token): self._clear_sample_status_progress(
                sid,
                task_token=token,
            ),
        )

    def _on_fit_progress_value_changed(self, value: int) -> None:
        if not getattr(self, "_fit_worker_running", False):
            return
        sample_id = getattr(self, "_fit_task_sample_id", None)
        task_token = getattr(self, "_fit_task_token", None)
        if not sample_id or not task_token:
            return
        self._set_sample_status_progress(
            sample_id,
            value,
            getattr(self, "_fit_task_operation", "计算") or "计算",
            "正在计算",
            task_token=task_token,
        )

    def _finish_sample_calculation_status(
        self,
        sample_id: str,
        task_token: str,
        success: bool,
    ) -> None:
        if success:
            self._complete_sample_status_progress(sample_id, task_token)
        else:
            self._clear_sample_status_progress(sample_id, task_token=task_token)
        if getattr(self, "_fit_task_token", None) == task_token:
            self._fit_task_sample_id = None
            self._fit_task_token = None
            self._fit_task_operation = ""

    def _project_io_job_is_active(self, job_token: str) -> bool:
        return job_token in self._project_save_jobs.values() or job_token in self._file_load_jobs.values()

    def _update_pending_file_load_progress(
        self,
        job_token: str,
        value: int,
        stage: str,
    ) -> None:
        pending_items = list(self._pending_file_loads.items())
        for offset, (_key, pending) in enumerate(pending_items):
            if pending.get("job_token") != job_token:
                continue
            pending["progress"] = max(0, min(100, int(value)))
            pending["stage"] = str(stage)
            row = len(self.samples) + offset
            status_item = self.sample_table.item(
                row,
                getattr(self, "sample_status_col", 2),
            )
            if status_item is not None:
                status_item.setData(SAMPLE_STATUS_PROGRESS_ROLE, pending["progress"])
                status_item.setToolTip(f"{pending['stage']} · {pending['progress']}%")
                self.sample_table.viewport().update(
                    self.sample_table.visualItemRect(status_item)
                )
            return

    def _update_project_io_progress(
        self,
        job_token: str,
        value: int,
        operation: str,
        stage: str,
        file_name: str,
    ) -> None:
        if not self._project_io_job_is_active(job_token):
            return
        self._update_pending_file_load_progress(job_token, value, stage)
        for sample_id, save_token in self._project_save_jobs.items():
            if save_token == job_token:
                self._set_sample_status_progress(
                    sample_id,
                    value,
                    operation,
                    stage,
                    task_token=job_token,
                )
                break
        if getattr(self, "_fit_worker_running", False) or getattr(self, "_alpha_fast_running", False):
            return
        self._project_io_progress_generation += 1
        value = max(0, min(100, int(value)))
        self.progress_label.setText(f"{operation}：{stage} · {value}%")
        self.progress_label.setToolTip(str(file_name))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)
        self.progress_label.show()
        self.progress_bar.show()

    def _finish_project_io_progress(self, message: str, *, success: bool) -> None:
        if self._project_save_jobs or self._file_load_jobs:
            return
        if getattr(self, "_fit_worker_running", False) or getattr(self, "_alpha_fast_running", False):
            return
        self._project_io_progress_generation += 1
        generation = self._project_io_progress_generation
        self.progress_label.setText(message)
        if success:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
        self.progress_label.show()
        self.progress_bar.show()

        def hide_when_idle() -> None:
            if generation != self._project_io_progress_generation:
                return
            if self._project_save_jobs or self._file_load_jobs:
                return
            if getattr(self, "_fit_worker_running", False) or getattr(self, "_alpha_fast_running", False):
                return
            self.progress_label.hide()
            self.progress_bar.hide()

        QTimer.singleShot(1600 if success else 3000, hide_when_idle)

    def save_project_file(
        self,
        _checked: bool = False,
        *,
        save_as: bool = False,
        sample_index: int | None = None,
    ) -> bool:
        """Save exactly one sample and its current analysis to one project file."""
        if self._project_operation_blocked_by_calculation():
            return False
        index = self.active_sample_index if sample_index is None else int(sample_index)
        if not (0 <= index < len(self.samples)):
            QMessageBox.information(self, "保存工程", "请先选择一个样品。")
            return False
        if index == self.active_sample_index:
            self._capture_active_sample_state()
        sample = self.samples[index]
        if sample.sample_id in self._project_save_jobs:
            self.statusBar().showMessage(f"样品“{sample.name}”正在后台保存，请稍候", 3000)
            return False

        target = None if save_as else sample.project_path
        if not target:
            initial_dir = str(
                Path(self.import_directory or Path.home()) / self._suggested_project_name(sample)
            )
            target, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "保存 XRD 工程",
                initial_dir,
                "XRD Analyzer 工程 (*.xrdproj);;所有文件 (*)",
            )
            if not target:
                return False
            if not str(target).lower().endswith(PROJECT_EXTENSION):
                target = str(target) + PROJECT_EXTENSION

        target = str(Path(target).resolve())
        record = self._sample_to_project_record(sample)
        saved_revision = int(sample.project_revision)
        sample_id = str(sample.sample_id)
        project_uuid = str(sample.project_uuid)
        job_token = str(uuid.uuid4())
        self._project_save_jobs[sample_id] = job_token
        self._set_sample_status_progress(
            sample_id,
            0,
            "保存工程",
            "准备保存",
            task_token=job_token,
        )

        def worker() -> None:
            error = None
            progress = lambda value, stage: self.ui(
                self._update_project_io_progress,
                job_token,
                value,
                "保存工程",
                stage,
                Path(target).name,
            )
            try:
                # Serialize project writes so several large manual saves
                # cannot saturate the CPU at once.
                with self._project_io_lock:
                    prepared_snapshot = self._prepared_snapshot_for_revision(
                        sample_id,
                        saved_revision,
                        project_uuid,
                    )
                    if prepared_snapshot:
                        try:
                            materialize_project_snapshot(
                                prepared_snapshot,
                                target,
                                progress_callback=progress,
                            )
                        except Exception:
                            # A stale or externally removed cache must never
                            # prevent a normal, authoritative project save.
                            save_project(
                                target,
                                [record],
                                active_sample_index=0,
                                app_version=APP_VERSION,
                                project_uuid=project_uuid,
                                progress_callback=progress,
                            )
                    else:
                        save_project(
                            target,
                            [record],
                            active_sample_index=0,
                            app_version=APP_VERSION,
                            project_uuid=project_uuid,
                            progress_callback=progress,
                        )
            except Exception as exc:
                error = str(exc)
            self.ui(
                self._finish_project_save,
                sample_id,
                job_token,
                target,
                saved_revision,
                error,
            )

        try:
            self._project_io_executor.submit(worker)
        except Exception as exc:
            self._project_save_jobs.pop(sample_id, None)
            self._clear_sample_status_progress(sample_id, task_token=job_token)
            QMessageBox.warning(self, "保存工程失败", str(exc))
            return False

        self.statusBar().showMessage(f"正在后台保存工程：{Path(target).name}")
        return True

    def _finish_project_save(
        self,
        sample_id: str,
        job_token: str,
        target: str,
        saved_revision: int,
        error: str | None,
    ) -> None:
        if self._project_save_jobs.get(sample_id) != job_token:
            return
        self._project_save_jobs.pop(sample_id, None)
        sample = next((item for item in self.samples if item.sample_id == sample_id), None)
        if error:
            self._clear_sample_status_progress(sample_id, task_token=job_token)
            self._finish_project_io_progress("工程保存失败", success=False)
            QMessageBox.warning(self, "保存工程失败", error)
            self.statusBar().showMessage(f"工程保存失败：{Path(target).name}", 5000)
            return

        if sample is not None:
            sample.project_path = target
            if sample.project_revision == saved_revision:
                sample.project_dirty = False
            self.settings.setValue("recent_project", target)
            self.refresh_sample_table()
        self._update_window_title()
        if sample is not None and sample.project_dirty:
            self.statusBar().showMessage(
                f"工程快照已保存：{Path(target).name}；保存期间产生了新更改",
                6000,
            )
        else:
            self.statusBar().showMessage(f"工程已保存：{Path(target).name}", 4000)
        self._complete_sample_status_progress(sample_id, job_token)
        self._finish_project_io_progress("工程保存完成", success=True)

    def _read_project_sample(
        self,
        path: str | Path,
        *,
        progress_callback=None,
    ) -> tuple[XRDSample, bool]:
        payload = load_project(path, progress_callback=progress_callback)
        records = list(payload.get("samples", []))
        if len(records) != 1:
            raise ProjectFormatError("工程中没有样品")

        sample = self._sample_from_project_record(records[0])
        # ``file_name`` describes the artifact the user actually imported,
        # whereas ``sample_name`` remains the specimen name recorded inside
        # the original RAW/TXT data. Project variants must therefore show
        # their own complete .xrdproj filenames in the parameter panel.
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["file_name"] = Path(path).name
        incompatible_algorithm = str(payload.get("algorithm_version") or "") != ALGORITHM_VERSION
        sample.project_path = str(Path(path).resolve())
        sample.project_uuid = str(payload.get("project_uuid") or uuid.uuid4())
        sample.project_dirty = False
        sample.result_is_current = bool(
            not incompatible_algorithm
            and sample.results
            and sample.result_signature
            and sample.result_signature == self._result_signature_for_sample(sample)
        )
        return sample, incompatible_algorithm

    def load_project_file(self, path: str | Path) -> bool:
        """Compatibility entry point for importing one project file."""
        return self._start_file_load(path)

    def _start_file_load(self, path: str | Path) -> bool:
        """Load TXT, RAW, or a project in the background with stage progress."""
        if self._project_operation_blocked_by_calculation():
            return False
        key = self._path_key(path)
        if key in self._file_load_jobs or any(
            key in self._sample_import_keys(sample) for sample in self.samples
        ):
            self.statusBar().showMessage(f"文件已在列表中：{Path(path).name}", 3000)
            return False
        resolved_path = str(Path(path).resolve())
        is_project = Path(resolved_path).suffix.lower() == PROJECT_EXTENSION
        operation = "读取工程" if is_project else "读取数据"
        parameter_state = self._current_parameter_state()
        peak_states = self._default_peak_states()
        job_token = str(uuid.uuid4())
        self._file_load_jobs[key] = job_token
        self._pending_file_loads[key] = {
            "job_token": job_token,
            "path": resolved_path,
            "is_project": is_project,
            "progress": 0,
            "stage": "等待后台读取",
        }
        self.refresh_sample_table()
        self._update_project_io_progress(
            job_token,
            0,
            operation,
            "等待后台读取",
            Path(resolved_path).name,
        )

        def worker() -> None:
            sample = None
            incompatible_algorithm = False
            error = None
            progress = lambda value, stage: self.ui(
                self._update_project_io_progress,
                job_token,
                value,
                operation,
                stage,
                Path(resolved_path).name,
            )
            try:
                with self._project_io_lock:
                    if is_project:
                        sample, incompatible_algorithm = self._read_project_sample(
                            resolved_path,
                            progress_callback=progress,
                        )
                    else:
                        sample = self._load_sample_from_path(
                            resolved_path,
                            parameter_state=parameter_state,
                            peak_states=peak_states,
                            progress_callback=progress,
                        )
            except Exception as exc:
                error = str(exc)
            self.ui(
                self._finish_file_load,
                key,
                job_token,
                resolved_path,
                sample,
                incompatible_algorithm,
                is_project,
                error,
            )

        try:
            self._project_io_executor.submit(worker)
        except Exception as exc:
            self._file_load_jobs.pop(key, None)
            self._pending_file_loads.pop(key, None)
            self.refresh_sample_table()
            self._finish_project_io_progress("文件读取失败", success=False)
            QMessageBox.warning(self, "读取文件失败", str(exc))
            return False
        self.statusBar().showMessage(f"正在后台读取：{Path(path).name}")
        return True

    def _finish_file_load(
        self,
        key: str,
        job_token: str,
        path: str,
        sample: XRDSample | None,
        incompatible_algorithm: bool,
        is_project: bool,
        error: str | None,
    ) -> None:
        if self._file_load_jobs.get(key) != job_token:
            return
        self._file_load_jobs.pop(key, None)
        self._pending_file_loads.pop(key, None)
        if error or sample is None:
            self.refresh_sample_table()
            self._finish_project_io_progress("文件读取失败", success=False)
            QMessageBox.warning(self, "读取文件失败", error or "文件中没有有效数据")
            self.statusBar().showMessage(f"文件读取失败：{Path(path).name}", 5000)
            return

        if any(key in self._sample_import_keys(item) for item in self.samples):
            self.refresh_sample_table()
            self._finish_project_io_progress("文件已在列表中", success=True)
            return
        first_index = len(self.samples)
        if sample.sample_id in {item.sample_id for item in self.samples}:
            sample.sample_id = str(uuid.uuid4())
        self.samples.append(sample)

        clean_states = [item.project_dirty for item in self.samples]
        self._suspend_project_dirty = True
        try:
            self.refresh_sample_table()
            self.select_sample(first_index)
        finally:
            self._suspend_project_dirty = False
            for item, dirty in zip(self.samples, clean_states):
                item.project_dirty = dirty
            self._update_window_title()
        if is_project:
            self.settings.setValue("recent_project", path)
        details = []
        if incompatible_algorithm:
            details.append("历史算法结果已恢复，建议重新计算")
        suffix = "；" + "；".join(details) if details else ""
        kind = "工程" if is_project else "数据"
        self.statusBar().showMessage(f"已导入{kind}：{Path(path).name}{suffix}", 6000)
        self._finish_project_io_progress(f"{kind}读取完成", success=True)

    def _selected_sample_rows(self) -> list[int]:
        table = getattr(self, "sample_table", None)
        if table is None:
            return []
        return sorted({index.row() for index in table.selectedIndexes()})

    def _build_sample_context_menu(self, rows: list[int]) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self.sample_table)
        delete_action = menu.addAction("删除")
        delete_action.triggered.connect(
            lambda _checked=False, selected=tuple(rows): self._remove_sample_rows(selected)
        )
        if len(rows) == 1:
            row = rows[0]
            menu.addSeparator()
            save_action = menu.addAction("保存工程")
            save_action.triggered.connect(
                lambda _checked=False, index=row: self.save_project_file(sample_index=index)
            )
            save_as_action = menu.addAction("另存为工程")
            save_as_action.triggered.connect(
                lambda _checked=False, index=row: self.save_project_file(
                    save_as=True, sample_index=index
                )
            )
        return menu

    def _show_sample_context_menu(self, pos) -> None:
        item = self.sample_table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        if row >= len(self.samples):
            return
        rows = self._selected_sample_rows()
        if row not in rows:
            self.sample_table.clearSelection()
            self.sample_table.setCurrentCell(row, getattr(self, "sample_file_col", 1))
            self.sample_table.selectRow(row)
            rows = [row]
        if not rows:
            return
        menu = self._build_sample_context_menu(rows)
        menu.exec_(self.sample_table.viewport().mapToGlobal(pos))

    def _remove_sample_rows(self, rows) -> None:
        if self._project_operation_blocked_by_calculation():
            return
        rows = sorted({int(row) for row in rows if 0 <= int(row) < len(self.samples)})
        if not rows:
            return
        old_active = self.active_sample_index
        removed = set(rows)
        for row in rows:
            sample_id = self.samples[row].sample_id
            self._discard_prepared_project_snapshot(sample_id)
            self._sample_status_tasks.pop(str(sample_id), None)
        self.samples = [sample for index, sample in enumerate(self.samples) if index not in removed]

        if not self.samples:
            target_index = -1
        elif old_active in removed:
            target_index = min(rows[0], len(self.samples) - 1)
        else:
            target_index = old_active - sum(row < old_active for row in rows)

        self.active_sample_index = -1
        self.data_loaded = False
        self.results_ready = False
        self._fit_cache = None
        self.refresh_sample_table()
        if target_index >= 0:
            self.select_sample(target_index)
        else:
            self.fit_quality_history = []
            self.current_rfit_percent = None
            self.clear_result_table()
            self._clear_plot(self.preview_plot, title="完整数据预览")
            self._clear_plot(self.fit_plot, title="拟合范围预览")
            self._clear_plot(self.size_plot, title="粒径分布 (计算后显示)")
            self._update_fit_quality_display()
            self._update_window_title()
        self.statusBar().showMessage(f"已从列表移除 {len(rows)} 个样品", 3000)

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

    @staticmethod
    def _build_particle_size_grid(d_min: float, d_max: float, d_step: float) -> np.ndarray:
        d_min = float(d_min)
        d_max = float(d_max)
        d_step = max(float(d_step), 1e-4)
        if d_max <= d_min:
            d_max = d_min + d_step
        count = max(2, int(np.floor((d_max - d_min) / d_step)) + 1)
        grid = d_min + np.arange(count, dtype=float) * d_step
        if grid[-1] < d_max - d_step * 0.25:
            grid = np.append(grid, d_max)
        else:
            grid[-1] = min(grid[-1], d_max)
        return grid

    def _current_sample_key(self) -> str:
        index = getattr(self, "active_sample_index", -1)
        samples = getattr(self, "samples", [])
        if 0 <= index < len(samples):
            sample = samples[index]
            return str(sample.sample_id or sample.data_fingerprint or self._path_key(sample.path))
        return ""

    @staticmethod
    def _signature_value(value):
        if isinstance(value, dict):
            return tuple(
                (str(key), XRDApp._signature_value(value[key]))
                for key in sorted(value, key=str)
            )
        if isinstance(value, np.ndarray):
            return tuple(XRDApp._signature_value(v) for v in value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(XRDApp._signature_value(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(XRDApp._signature_value(v) for v in value))
        if isinstance(value, (float, np.floating)):
            return round(float(value), 8)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if value is None:
            return None
        return str(value)

    def _fit_signature_from_params(self, params: dict, mu_centers=None) -> tuple:
        mu_values = params.get("mu_centers", []) if mu_centers is None else mu_centers
        return (
            ("sample", str(params.get("sample_key") or self._current_sample_key())),
            ("source", str(params.get("source", "Cu"))),
            ("angle_min", round(float(params.get("angle_min", 0.0)), 8)),
            ("angle_max", round(float(params.get("angle_max", 0.0)), 8)),
            ("d_min", round(float(params.get("d_min", 0.0)), 8)),
            ("d_max", round(float(params.get("d_max", 0.0)), 8)),
            ("d_step", round(float(params.get("d_step", 0.0)), 8)),
            ("instrument_fwhm", round(float(params.get("instrument_fwhm", 0.0)), 8)),
            ("regularization_method", str(params.get("regularization_method", "l2")).lower()),
            ("peak_kernel", str(params.get("peak_kernel", "pearson7") or "pearson7").lower()),
            ("active_peak_indices", tuple(int(i) for i in params.get("active_peak_indices", []))),
            ("mu_centers", tuple(round(float(v), 8) for v in mu_values)),
            ("baseline_state", self._signature_value(params.get("baseline_state"))),
        )

    @staticmethod
    def _basis_total_from_components(basis_k1_list, basis_k2_list):
        pairs = list(zip(basis_k1_list or [], basis_k2_list or []))
        if not pairs:
            return None
        return np.hstack([
            np.asarray(k1, dtype=float) + np.asarray(k2, dtype=float)
            for k1, k2 in pairs
        ])

    def _build_fit_cache(
        self,
        params: dict,
        best_mu: list[float],
        basis_k1_list,
        basis_k2_list,
        y_scaled,
        L_single,
        alpha_val: float,
        resid: float,
    ) -> dict | None:
        basis_total = self._basis_total_from_components(basis_k1_list, basis_k2_list)
        if basis_total is None:
            return None
        return {
            "signature": self._fit_signature_from_params(params, best_mu),
            "basis_total": basis_total,
            "y_scaled": np.asarray(y_scaled, dtype=float).copy(),
            "L_single": np.asarray(L_single, dtype=float).copy(),
            "n_peaks": int(len(best_mu)),
            "alpha": float(alpha_val),
            "last_resid": float(resid),
            "regularization_method": str(params.get("regularization_method", "l2") or "l2").lower(),
            "peak_kernel": str(params.get("peak_kernel", "pearson7") or "pearson7").lower(),
            "active_peak_indices": list(params.get("active_peak_indices", [])),
            "best_mu": [float(v) for v in best_mu],
            "sample_key": str(params.get("sample_key") or self._current_sample_key()),
        }

    def _alpha_fast_request(self) -> dict | None:
        if not getattr(self, "data_loaded", False) or not getattr(self, "results_ready", False):
            return None
        cache = getattr(self, "_fit_cache", None)
        if not cache and 0 <= getattr(self, "active_sample_index", -1) < len(getattr(self, "samples", [])):
            cache = self.samples[self.active_sample_index].results.get("_fit_cache")
            self._fit_cache = cache
        if not cache:
            self.statusBar().showMessage("请先完成一次完整计算，再实时调整 α", 2500)
            return None

        current_signature = cache.get("signature")
        cache_sample_key = str(cache.get("sample_key") or "")
        if cache_sample_key and cache_sample_key != self._current_sample_key():
            self.statusBar().showMessage("alpha 快速重算缓存与当前样品不匹配，请重新计算", 3000)
            return None

        basis_total = cache.get("basis_total")
        y_scaled = cache.get("y_scaled")
        L_single = cache.get("L_single")
        n_peaks = int(cache.get("n_peaks") or 0)
        if basis_total is None or y_scaled is None or L_single is None or n_peaks <= 0:
            self.statusBar().showMessage("α 快速重算缓存不完整，请重新计算一次", 3000)
            return None

        return {
            "revision": int(self._alpha_fast_revision),
            "signature": current_signature,
            "sample_key": str(cache.get("sample_key") or self._current_sample_key()),
            "basis_total": basis_total,
            "y_scaled": y_scaled,
            "L_single": L_single,
            "n_peaks": n_peaks,
            "alpha": float(self.slider_alpha.get()),
            "regularization_method": str(cache.get("regularization_method", "l2") or "l2").lower(),
            "peak_kernel": str(cache.get("peak_kernel", "pearson7") or "pearson7").lower(),
            "active_peak_indices": list(cache.get("active_peak_indices") or []),
        }

    def _on_alpha_value_changed(self, _value: float) -> None:
        self._alpha_fast_revision += 1
        self._save_current_parameter_state()
        if getattr(self, "_fit_worker_running", False):
            return
        if not getattr(self, "results_ready", False):
            return
        self._alpha_fast_timer.start()

    def _start_alpha_fast_recompute(self) -> None:
        if getattr(self, "_fit_worker_running", False):
            return
        if getattr(self, "_alpha_fast_running", False):
            self._alpha_fast_pending = True
            return
        request = self._alpha_fast_request()
        if request is None:
            return

        self._alpha_fast_running = True
        self.statusBar().showMessage(f"正在按 α={request['alpha']:.2f} 快速重算分布...", 1200)
        threading.Thread(target=self._alpha_fast_worker, args=(request,), daemon=True).start()

    def _alpha_fast_worker(self, request: dict) -> None:
        f_total = None
        resid = None
        error = None
        try:
            f_total, resid = solve_regularized_from_basis(
                request["basis_total"],
                request["y_scaled"],
                request["L_single"],
                request["n_peaks"],
                request["alpha"],
                request["regularization_method"],
            )
            if f_total is None or f_total.sum() <= 1e-9 or not np.isfinite(f_total).all():
                raise RuntimeError("求解结果为空")
        except Exception as exc:
            error = str(exc)
        self.ui(self._finish_alpha_fast_recompute, request, f_total, resid, error)

    def _finish_alpha_fast_recompute(self, request: dict, f_total, resid, error) -> None:
        self._alpha_fast_running = False
        try:
            if int(request.get("revision", -1)) != int(getattr(self, "_alpha_fast_revision", -2)):
                return
            if error:
                self.statusBar().showMessage(f"α 快速重算失败：{error}", 3500)
                return

            request_sample_key = str(request.get("sample_key") or "")
            if request_sample_key and request_sample_key != self._current_sample_key():
                return

            if hasattr(self, "_save_current_marker_label_state"):
                self._save_current_marker_label_state()
            if hasattr(self, "_save_current_plot_view_state"):
                self._save_current_plot_view_state()
            views = self._capture_plot_views() if hasattr(self, "_capture_plot_views") else {}

            self.best_f_total = np.asarray(f_total, dtype=float)
            self.result_regularization_method = request["regularization_method"]
            self.result_peak_kernel = request.get("peak_kernel", "pearson7")
            self.result_active_peak_indices = list(request["active_peak_indices"])
            if isinstance(getattr(self, "_fit_cache", None), dict):
                self._fit_cache["alpha"] = float(request["alpha"])
                self._fit_cache["last_resid"] = float(resid)
            self.process_multi_peak_results(self.result_active_peak_indices)
            if views and hasattr(self, "_restore_plot_views"):
                self._restore_plot_views(views)
            self.statusBar().showMessage(f"α={request['alpha']:.2f} 已快速重算", 1800)
        finally:
            if getattr(self, "_alpha_fast_pending", False):
                self._alpha_fast_pending = False
                self._alpha_fast_timer.start(1)

    def _collect_fit_params(self):
        angle_min = self.slider_min.get()
        angle_max = self.slider_max.get()
        fit_peak_indices = self._selected_peak_indices_in_fit_range(angle_min, angle_max)
        return {
            "sample_key": self._current_sample_key(),
            "source": self.source_var.get(),
            "mu_centers": [self.peak_mu_sliders[i].get() for i in fit_peak_indices],
            "angle_min": angle_min,
            "angle_max": angle_max,
            "d_min": float(getattr(self, "particle_size_min", 0.1)),
            "d_max": float(getattr(self, "particle_size_max", 100.0)),
            "d_step": float(getattr(self, "particle_size_step", 0.1)),
            "alpha": float(self.slider_alpha.get()),
            "instrument_fwhm": float(getattr(self, "instrument_fwhm", 0.0)),
            "regularization_method": str(getattr(self, "regularization_method", "l2")),
            "peak_kernel": str(getattr(self, "peak_kernel", "pearson7") or "pearson7"),
            "active_peak_indices": list(fit_peak_indices),
            "baseline_state": self._current_manual_baseline_state(),
        }

    def _set_default_import_range_and_peak(self):
        """Set import defaults and place peak 1 at the strongest point in range."""
        if not self.data_loaded:
            return

        self._refresh_angle_control_bounds()
        default_state = self._default_analysis_state_for_current_data()
        angle_min = float(default_state["angle_min"])
        angle_max = float(default_state["angle_max"])
        self.slider_min.set(angle_min, emit=False)
        self.slider_max.set(angle_max, emit=False)
        self._normalize_angle_range()
        self._refresh_peak_slider_bounds()
        self._save_current_analysis_state()

        x = np.asarray(self.x_data, dtype=float)
        y = np.asarray(self.y_data, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x >= angle_min) & (x <= angle_max)
        if not np.any(mask):
            mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return

        local_x = x[mask]
        local_y = y[mask]
        peak_center = float(local_x[int(np.argmax(local_y))])

        for peak_idx in self.active_peak_indices:
            if peak_idx == 0 and peak_idx < len(self.peak_mu_sliders):
                self.peak_mu_sliders[peak_idx].set(peak_center, emit=False)
                break
        self._save_current_peak_states()
        self._save_current_analysis_state()

    def _data_angle_bounds(self) -> tuple[float, float]:
        if getattr(self, "data_loaded", False) and hasattr(self, "x_data"):
            try:
                x = np.asarray(self.x_data, dtype=float)
                finite = x[np.isfinite(x)]
                if finite.size:
                    low = float(np.nanmin(finite))
                    high = float(np.nanmax(finite))
                    if high > low:
                        return low, high
            except Exception:
                pass
        return float(DEFAULT_ANGLE_MIN), float(DEFAULT_ANGLE_MAX)

    def _refresh_angle_control_bounds(self) -> None:
        if not hasattr(self, "slider_min") or not hasattr(self, "slider_max"):
            return
        low, high = self._data_angle_bounds()
        if high <= low:
            high = low + 0.01
        self.slider_min.config(from_=low, to=high)
        self.slider_max.config(from_=low, to=high)

    def _default_analysis_state_for_current_data(self) -> dict[str, float]:
        low, high = self._data_angle_bounds()
        span = high - low
        if span > 20.0:
            angle_min = low + 10.0
            angle_max = high - 10.0
        elif span > 0.02:
            margin = min(span * 0.25, max(0.0, span / 2.0 - 0.01))
            angle_min = low + margin
            angle_max = high - margin
        else:
            angle_min, angle_max = low, high
        if angle_max <= angle_min:
            angle_min, angle_max = low, high
        return {"angle_min": round(float(angle_min), 2), "angle_max": round(float(angle_max), 2)}

    def load_file(self):
        """Open the unified TXT/RAW/project import dialog."""
        dialog = XRDFileImportDialog(
            self,
            initial_dir=self.import_directory,
            existing_paths=None,
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
        self.load_files(paths)

    @staticmethod
    def _path_key(path: str | Path) -> str:
        try:
            return str(Path(path).resolve()).lower()
        except OSError:
            return str(path).lower()

    def _load_sample_from_path(
        self,
        path: str | Path,
        *,
        parameter_state: dict | None = None,
        peak_states: list[dict] | None = None,
        progress_callback=None,
    ) -> XRDSample:
        def progress(value: int, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(value, stage)

        progress(0, "打开数据文件")
        progress(10, "解析 TXT/RAW 数据")
        x, y, name, meta = load_xrd_file(str(path))
        # Keep the complete disk filename, including .raw/.txt. The separate
        # sample_name metadata continues to represent the name inside the scan.
        meta["file_name"] = os.path.basename(str(path))
        progress(38, "计算数据内容指纹")
        fingerprint = data_sha256(x, y)
        try:
            progress(46, "校验源文件")
            source_fingerprint = file_sha256(path)
        except OSError:
            source_fingerprint = ""

        progress(92, "创建样品记录")
        if parameter_state is None:
            parameter_state = self._current_parameter_state()
        if peak_states is None:
            peak_states = self._default_peak_states()
        sample = XRDSample(
            path=str(path),
            x_data=x,
            y_data=y,
            name=name,
            metadata=meta,
            data_fingerprint=fingerprint,
            file_fingerprint=source_fingerprint,
            parameter_state=dict(parameter_state),
            peak_states=[dict(item) for item in peak_states],
        )
        progress(100, "数据读取完成")
        return sample

    def _sample_import_keys(self, sample: XRDSample) -> set[str]:
        # The row represents the artifact the user explicitly imported.  A
        # project keeps its original RAW/TXT path only as provenance; that
        # embedded source path must not block importing the raw data as a new,
        # uncalculated sample alongside one or more project variants.
        imported_path = sample.project_path or sample.path
        return {self._path_key(imported_path)} if imported_path else set()

    def sync_files(self, paths: list[str]) -> None:
        """Backward-compatible alias: imports now append and deletion lives in the table menu."""
        self.load_files(paths)

    def load_files(self, paths: list[str]):
        """Queue TXT, RAW, or project files for responsive background loading."""
        if self._project_operation_blocked_by_calculation():
            return
        existing = set().union(*(self._sample_import_keys(sample) for sample in self.samples)) if self.samples else set()
        existing.update(self._file_load_jobs)
        queued = 0
        for path in paths:
            key = self._path_key(path)
            if key in existing:
                continue
            if self._start_file_load(path):
                existing.add(key)
                queued += 1
        if queued > 1:
            self.statusBar().showMessage(f"已加入后台读取队列：{queued} 个文件", 4000)

    def refresh_sample_table(self):
        compare_col = getattr(self, "sample_compare_col", 0)
        file_col = getattr(self, "sample_file_col", 1)
        status_col = getattr(self, "sample_status_col", 2)
        self.sample_table.blockSignals(True)
        self.sample_table.setRowCount(len(self.samples) + len(self._pending_file_loads))
        for row, sample in enumerate(self.samples):
            compare_item = QTableWidgetItem()
            compare_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            compare_item.setCheckState(Qt.Checked if getattr(sample, "compare_visible", True) else Qt.Unchecked)
            compare_item.setTextAlignment(Qt.AlignCenter)
            compare_item.setToolTip("在对比分析中显示" if getattr(sample, "compare_visible", True) else "在对比分析中隐藏")

            if sample.project_path:
                display_file_name = Path(sample.project_path).name
            elif sample.path:
                display_file_name = Path(sample.path).name
            else:
                display_file_name = str(sample.name or "样品")
            file_item = QTableWidgetItem(display_file_name)
            file_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            file_item.setForeground(QBrush(QColor("#111827")))
            tooltip_lines = []
            if sample.path:
                tooltip_lines.append(f"数据文件：{sample.path}")
            if sample.project_path:
                tooltip_lines.append(f"工程文件：{sample.project_path}")
            file_item.setToolTip("\n".join(tooltip_lines))
            status_item = QTableWidgetItem("")
            status_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            status_item.setIcon(self._status_icon(sample.status))
            status_item.setTextAlignment(Qt.AlignCenter)
            task = self._sample_status_tasks.get(str(sample.sample_id))
            if task is not None:
                progress = max(0, min(100, int(task.get("progress", 0))))
                operation = str(task.get("operation") or "处理中")
                stage = str(task.get("stage") or operation)
                status_item.setData(SAMPLE_STATUS_PROGRESS_ROLE, progress)
                status_item.setToolTip(f"{operation}：{stage} · {progress}%")
            else:
                status_item.setToolTip(self._idle_sample_status_tooltip(sample))
            self.sample_table.setItem(row, compare_col, compare_item)
            self.sample_table.setItem(row, file_col, file_item)
            old_widget = self.sample_table.cellWidget(row, status_col)
            if old_widget is not None:
                self.sample_table.removeCellWidget(row, status_col)
                old_widget.deleteLater()
            self.sample_table.setItem(row, status_col, status_item)
        for offset, pending in enumerate(self._pending_file_loads.values()):
            row = len(self.samples) + offset
            compare_item = QTableWidgetItem()
            compare_item.setFlags(Qt.ItemIsEnabled)

            file_item = QTableWidgetItem(Path(pending["path"]).name)
            file_item.setFlags(Qt.ItemIsEnabled)
            file_item.setForeground(QBrush(QColor("#374151")))
            file_item.setToolTip(str(pending["path"]))

            progress = max(0, min(100, int(pending.get("progress", 0))))
            stage = str(pending.get("stage") or "等待后台读取")
            status_item = QTableWidgetItem("")
            status_item.setFlags(Qt.ItemIsEnabled)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setData(SAMPLE_STATUS_PROGRESS_ROLE, progress)
            status_item.setToolTip(f"{stage} · {progress}%")

            self.sample_table.setItem(row, compare_col, compare_item)
            self.sample_table.setItem(row, file_col, file_item)
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
        self._mark_project_dirty(self.samples[row])
        self._sync_compare_select_all_state()
        self._update_comparison_plots_if_available()

    def _on_compare_select_all_changed(self, state: int):
        if getattr(self, "_updating_compare_checks", False):
            return
        if state == Qt.PartiallyChecked:
            return
        checked = state == Qt.Checked
        for sample in self.samples:
            if sample.compare_visible != checked:
                sample.compare_visible = checked
                self._mark_project_dirty(sample)
        compare_col = getattr(self, "sample_compare_col", 0)
        self.sample_table.blockSignals(True)
        for row in range(len(self.samples)):
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
        self._save_current_parameter_state()
        self._save_current_manual_baseline_state()
        self._save_current_marker_label_state()
        self._save_current_plot_view_state()
        self._save_current_size_visibility_state()
        self._save_current_size_total_inclusion_state()
        self.active_sample_index = index
        sample = self.samples[index]
        self.x_data = sample.x_data
        self.y_data = sample.y_data
        self.data_loaded = True
        self.current_file_name = sample.name
        self.current_metadata = sample.metadata
        self._restoring_sample_state = True
        try:
            if not sample.parameter_state:
                sample.parameter_state = self._current_parameter_state()
            self._apply_parameter_state(sample.parameter_state)
            had_analysis_state = bool(sample.analysis_state)
            self._refresh_angle_control_bounds()
            if sample.peak_states is None:
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
                self._apply_analysis_state(self._default_analysis_state_for_current_data())
            self._apply_peak_states(sample.peak_states, refresh_preview=False)
            self._apply_manual_baseline_state(sample.baseline_state, redraw=False)
            self._apply_marker_label_state(sample.marker_label_state, apply_items=False)
            self._apply_plot_view_state(sample.plot_view_state)
            self._apply_size_visibility_state(sample.size_visibility_state)
            self._apply_size_total_inclusion_state(sample.size_total_inclusion_state)
        finally:
            self._restoring_sample_state = False
        self.update_info_panel(sample.metadata)
        if sample.results:
            self._restore_sample_results(sample)
        else:
            self.results_ready = False
            self._fit_cache = None
            self.fit_quality_history = []
            self.current_rfit_percent = None
            if hasattr(self, "_update_fit_quality_display"):
                self._update_fit_quality_display()
            self.clear_result_table()
            if not had_analysis_state:
                self._set_default_import_range_and_peak()
        self._save_current_analysis_state()
        self._update_window_title()
        # Restarting a zero-delay timer coalesces rapid row clicks.  All state
        # is already active, while expensive plot rebuilding happens once for
        # only the final selected sample after Qt repaints the selection.
        self._sample_render_timer.start(0)

    def _render_selected_sample(self) -> None:
        index = self.active_sample_index
        if not (0 <= index < len(self.samples)):
            return
        sample = self.samples[index]
        self._suspend_plot_updates = True
        try:
            self.update_preview(None)
            if sample.results and self.results_ready:
                self.update_multi_peak_plots()
                self._restore_sample_plot_view_state()
                self.update_result_table()
        finally:
            self._suspend_plot_updates = False
        self._safe_draw_idle()
        if (
            hasattr(self, "right_tabs")
            and hasattr(self, "compare_tab")
            and self.right_tabs.currentWidget() is self.compare_tab
        ):
            self._update_comparison_plots_if_available()

    def _store_current_sample_results(self):
        if not (0 <= self.active_sample_index < len(self.samples)):
            return
        sample = self.samples[self.active_sample_index]
        sample.status = "complete"
        sample.analysis_state = self._current_analysis_state()
        sample.parameter_state = self._current_parameter_state()
        sample.baseline_state = self._current_manual_baseline_state()
        self.marker_label_state = self._current_marker_label_state()
        sample.marker_label_state = self.marker_label_state
        sample.plot_view_state = self._current_plot_view_state()
        sample.size_visibility_state = dict(getattr(self, "_size_visibility", {}) or {})
        sample.size_total_inclusion_state = dict(
            getattr(self, "_size_total_inclusion", {}) or {}
        )
        sample.results = {
            "best_f_total": self.best_f_total,
            "all_basis_k1": self.all_basis_k1,
            "all_basis_k2": self.all_basis_k2,
            "D_range": self.D_range,
            "all_peak_info": self.all_peak_info,
            "global_max_component_area": self.global_max_component_area,
            "result_active_peak_indices": self.result_active_peak_indices,
            "result_regularization_method": getattr(self, "result_regularization_method", "l2"),
            "result_peak_kernel": getattr(self, "result_peak_kernel", "pearson7"),
            "x_segment": self.x_segment,
            "y_segment_raw": self.y_segment_raw,
            "y_segment": self.y_segment,
            "background": self.background,
            "fit_quality_history": list(getattr(self, "fit_quality_history", [])),
            "current_rfit_percent": getattr(self, "current_rfit_percent", None),
            "_fit_cache": getattr(self, "_fit_cache", None),
        }
        self.fit_curve_snapshot = None
        if not sample.data_fingerprint:
            sample.data_fingerprint = data_sha256(sample.x_data, sample.y_data)
        sample.result_signature = self._result_signature_for_sample(sample)
        sample.result_is_current = True
        sample.runtime_plot_cache.clear()
        # Build the one-dimensional display curves once in the calculation
        # thread. Plotting and compact project serialization then reuse them.
        try:
            self._fit_curve_data_cache(self.result_active_peak_indices)
        except Exception:
            pass
        self._mark_project_dirty()
        self._queue_prepared_project_snapshot(sample)
        self.ui(self.refresh_sample_table)
        self.ui(self._update_comparison_plots_if_available)

    def _restore_sample_results(self, sample: XRDSample):
        self._fit_cache = sample.results.get("_fit_cache")
        self.all_basis_k1 = list(sample.results.get("all_basis_k1") or [])
        self.all_basis_k2 = list(sample.results.get("all_basis_k2") or [])
        self.fit_curve_snapshot = sample.results.get("fit_curve_snapshot")
        for key, value in sample.results.items():
            setattr(self, key, value)
        self.fit_quality_history = list(sample.results.get("fit_quality_history", []))
        restored_rfit = sample.results.get("current_rfit_percent")
        if restored_rfit is None:
            restored_rfit = self._calculate_current_rfit_percent()
        self.current_rfit_percent = restored_rfit
        if hasattr(self, "_update_fit_quality_display"):
            self._update_fit_quality_display()
        self.results_ready = True

    def _apply_fit_peak_positions(self, active_peak_indices, mu_values) -> None:
        for peak_idx, mu in zip(active_peak_indices, mu_values):
            if peak_idx < len(self.peak_mu_sliders):
                self.peak_mu_sliders[peak_idx].set(float(mu), emit=False)
        self._save_current_peak_states()
        if hasattr(self, "_sync_all_peak_lines"):
            self._sync_all_peak_lines()

    def _eval_peak_candidate_batch(
        self,
        executor,
        candidates,
        base_mu,
        peak_idx,
        x,
        y_scaled,
        lam1,
        lam2,
        L_single,
        D_range,
        alpha_val,
        inst_fwhm,
        progress_state: dict,
        peak_kernel: str = "pearson7",
    ):
        args_common = (
            tuple(base_mu),
            int(peak_idx),
            x,
            y_scaled,
            lam1,
            lam2,
            INTENSITY_RATIO,
            L_single,
            D_range,
            alpha_val,
            inst_fwhm,
            peak_kernel,
        )
        futs = [executor.submit(_eval_candidate_for_index, float(mu), *args_common) for mu in candidates]
        pending = set(futs)
        results = []
        while pending:
            if self.stop_flag.is_set():
                self._stop_process_pool(executor, pending)
                self.ui_set(self.progress_var, "已停止")
                return None

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

                progress_state["done"] = int(progress_state.get("done", 0)) + 1
                estimate = max(1, int(progress_state.get("estimate", 1)))
                pct = min(95, int(progress_state["done"] * 95 / estimate))
                self.ui(self.progress_bar.setValue, pct)
                self.ui_set(
                    self.progress_var,
                    f"扫描峰 {int(peak_idx) + 1}/{int(progress_state.get('peak_count', 1))}: {pct}%",
                )
                if mu_val is not None:
                    results.append((float(loss), float(mu_val)))
        return results

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

        sample = self.samples[self.active_sample_index]
        task_token = str(uuid.uuid4())
        operation = "极速计算" if str(mode).lower() == "fast" else "精细计算"
        params["task_sample_id"] = str(sample.sample_id)
        params["task_token"] = task_token
        self._fit_task_sample_id = str(sample.sample_id)
        self._fit_task_token = task_token
        self._fit_task_operation = operation
        self._alpha_fast_revision += 1
        self._alpha_fast_pending = False
        self._alpha_fast_timer.stop()
        self._fit_worker_running = True
        self._set_sample_status_progress(
            sample.sample_id,
            0,
            operation,
            "准备计算",
            task_token=task_token,
        )
        self.stop_flag.clear()
        self.progress_label.show()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.ui_set(self.progress_var, "计算中...")

        for btn in (self.btn_fast, self.btn_fine):
            btn.setEnabled(False)

        worker = threading.Thread(target=self.compute_fit, args=(mode, params), daemon=True)
        worker.start()
        # Peak placement is a transient editing mode. Exit it only after the
        # calculation thread has actually started; validation failures above
        # intentionally leave the mode active so the user can keep editing.
        if getattr(self, "_peak_placement_mode", False):
            self._set_peak_placement_mode(False)

    def compute_fit(self, mode: str = "fine", params: dict | None = None):
        """执行多峰拟合（子线程）。"""
        calculation_succeeded = False
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
            d_step = float(params.get("d_step", getattr(self, "particle_size_step", 0.1)) or 0.1)
            self.D_range = self._build_particle_size_grid(d_min, d_max, d_step)
            L_single = build_regularization_matrix(len(self.D_range))
            scan_D_range = self.D_range
            scan_L_single = L_single

            alpha_val = float(params["alpha"])
            inst_fwhm = float(params["instrument_fwhm"])
            regularization_method = str(params.get("regularization_method", "l2") or "l2").lower()
            peak_kernel = str(params.get("peak_kernel", "pearson7") or "pearson7").lower()

            if mode == "fast":
                halfwidth, steps = 0.0, 0
                self.ui(self.progress_bar.setValue, 20)
                self.ui_set(self.progress_var, "固定峰位拟合...")
            else:
                halfwidth, steps = 0.1, 11
                scan_target_points = 220
                span = max(float(d_max) - float(d_min), 0.0)
                scan_d_step = max(float(d_step), span / max(1, scan_target_points - 1))
                scan_D_range = self._build_particle_size_grid(d_min, d_max, scan_d_step)
                if len(scan_D_range) < len(self.D_range):
                    scan_L_single = build_regularization_matrix(len(scan_D_range))
                else:
                    scan_D_range = self.D_range
                    scan_L_single = L_single

            total = max(1, len(mu_centers) * max(1, steps))
            done = 0
            best_mu = list(mu_centers)
            scan_executor = None
            scan_executor_stopped = False
            scan_workers = min(4, os.cpu_count() or 1)
            if mode != "fast":
                scan_executor = ProcessPoolExecutor(max_workers=scan_workers)

            for i in ([] if mode == "fast" else range(len(best_mu))):
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

                ex = scan_executor
                chunks = [
                    np.asarray(chunk, dtype=float)
                    for chunk in np.array_split(candidates, min(scan_workers, len(candidates)))
                    if len(chunk)
                ]
                args_common = (
                    tuple(best_mu),
                    i,
                    x,
                    y_scaled,
                    lam1,
                    lam2,
                    INTENSITY_RATIO,
                    scan_L_single,
                    scan_D_range,
                    alpha_val,
                    inst_fwhm,
                    peak_kernel,
                )
                future_counts = {
                    ex.submit(
                        _eval_candidate_chunk_for_index,
                        tuple(float(mu) for mu in chunk),
                        *args_common,
                    ): len(chunk)
                    for chunk in chunks
                }
                pending = set(future_counts)
                stopped = False
                try:
                    while pending:
                        if self.stop_flag.is_set():
                            stopped = True
                            scan_executor_stopped = True
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
                                results = list(fut.result())
                            except Exception:
                                results = [(np.inf, None)] * int(future_counts.get(fut, 1))

                            expected_count = int(future_counts.get(fut, len(results)))
                            if len(results) < expected_count:
                                results.extend([(np.inf, None)] * (expected_count - len(results)))

                            for loss, mu_val in results:
                                done += 1
                                pct = min(95, int(done * 95 / total))
                                self.ui(self.progress_bar.setValue, pct)
                                self.ui_set(
                                    self.progress_var,
                                f"扫描峰 {i + 1}/{len(best_mu)}：{pct}%",
                                )

                                if mu_val is not None and (best_loss is None or loss < best_loss):
                                    best_loss, best_val = loss, mu_val
                finally:
                    if scan_executor is None and not stopped:
                        ex.shutdown(wait=False, cancel_futures=True)

                best_mu[i] = best_val

            if self.stop_flag.is_set():
                self.ui_set(self.progress_var, "已停止")
                return

            ex = scan_executor if scan_executor is not None else _ImmediateExecutor()
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
                regularization_method=regularization_method,
                kernel=peak_kernel,
            )
            stopped = False
            try:
                while not fut.done():
                    if self.stop_flag.is_set():
                        stopped = True
                        if ex is scan_executor:
                            scan_executor_stopped = True
                        self._stop_process_pool(ex, (fut,))
                        self.ui_set(self.progress_var, "已停止")
                        return
                    wait((fut,), timeout=0.05)
                resid, f_total, basis_k1_list, basis_k2_list = fut.result()
            finally:
                if not stopped:
                    ex.shutdown(wait=False, cancel_futures=True)
                    if ex is scan_executor:
                        scan_executor_stopped = True

            if f_total is None:
                self.ui_set(self.progress_var, "拟合失败：解全为零")
                return

            self.best_f_total = f_total
            self.all_basis_k1 = basis_k1_list
            self.all_basis_k2 = basis_k2_list
            self.result_active_peak_indices = list(params["active_peak_indices"])

            self.ui(self._apply_fit_peak_positions, list(params["active_peak_indices"]), list(best_mu))

            self.x_segment = x
            self.y_segment_raw = y_raw
            self.y_segment = y
            self.background = background
            self.result_regularization_method = regularization_method
            self.result_peak_kernel = peak_kernel
            self._fit_cache = self._build_fit_cache(
                params,
                best_mu,
                basis_k1_list,
                basis_k2_list,
                y_scaled,
                L_single,
                alpha_val,
                resid,
            )

            self.process_multi_peak_results(self.result_active_peak_indices, history_mode=mode)
            calculation_succeeded = True
            self.ui_set(self.progress_var, "拟合成功！")

        except Exception as exc:
            self.ui(messagebox.showwarning, "提示", f"计算过程中发生错误: {exc}")
            self.ui_set(self.progress_var, "计算失败")
        finally:
            executor = locals().get("scan_executor")
            if executor is not None and not locals().get("scan_executor_stopped", False):
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            self._fit_worker_running = False
            task_sample_id = str((params or {}).get("task_sample_id") or "")
            task_token = str((params or {}).get("task_token") or "")
            if task_sample_id and task_token:
                self.ui(
                    self._finish_sample_calculation_status,
                    task_sample_id,
                    task_token,
                    calculation_succeeded,
                )
            for btn in (self.btn_fast, self.btn_fine):
                self.ui(btn.setEnabled, True)

    def stop_compute(self):
        """中断正在运行的计算。"""
        self.stop_flag.set()
        self.ui_set(self.progress_var, "正在停止...")

    def _calculate_current_rfit_percent(self) -> float | None:
        """Calculate Rfit from the exact normalized net profile used by NNLS."""
        cache = getattr(self, "_fit_cache", None)
        if not isinstance(cache, dict):
            return None
        basis_total = cache.get("basis_total")
        y_scaled = cache.get("y_scaled")
        f_total = getattr(self, "best_f_total", None)
        if basis_total is None or y_scaled is None or f_total is None:
            return None
        try:
            calculated = np.asarray(basis_total, dtype=float).dot(np.asarray(f_total, dtype=float))
            value = calculate_rfit_percent(y_scaled, calculated)
        except (TypeError, ValueError):
            return None
        return float(value) if np.isfinite(value) else None

    def _record_fit_quality(self, mode: str) -> None:
        value = self._calculate_current_rfit_percent()
        self.current_rfit_percent = value
        if value is None:
            return

        history = list(getattr(self, "fit_quality_history", []))
        next_iteration = int(history[-1].get("iteration", len(history)) + 1) if history else 1
        cache = getattr(self, "_fit_cache", {}) or {}
        history.append(
            {
                "iteration": next_iteration,
                "rfit_percent": float(value),
                "mode": "fast" if str(mode).lower() == "fast" else "fine",
                "peak_count": int(len(getattr(self, "result_active_peak_indices", []))),
                "alpha": float(cache.get("alpha", 0.0)),
            }
        )
        self.fit_quality_history = history

    def process_multi_peak_results(self, active_peak_indices=None, *, history_mode: str | None = None):
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
        if history_mode is not None:
            self._record_fit_quality(history_mode)
        else:
            self.current_rfit_percent = self._calculate_current_rfit_percent()
        self._store_current_sample_results()
        self.ui(self.progress_bar.setValue, 100)
        if hasattr(self, "_update_fit_quality_display"):
            self.ui(self._update_fit_quality_display)
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

    def _confirm_close_with_unsaved_samples(self) -> bool:
        if self._project_save_jobs or self._file_load_jobs:
            self.statusBar().showMessage("工程正在后台读取或保存，请完成后再关闭窗口", 4000)
            QMessageBox.information(
                self,
                "工程读写进行中",
                "工程正在后台读取或保存，请完成后再关闭窗口。",
            )
            return False
        dirty_rows = [index for index, sample in enumerate(self.samples) if sample.project_dirty]
        if not dirty_rows:
            return True
        if len(dirty_rows) == 1:
            row = dirty_rows[0]
            answer = QMessageBox.question(
                self,
                "样品工程尚未保存",
                f"样品“{self.samples[row].name}”有尚未保存到工程文件的更改，是否现在保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if answer == QMessageBox.Cancel:
                return False
            if answer == QMessageBox.Save:
                self.save_project_file(sample_index=row)
                return False
            return True

        answer = QMessageBox.warning(
            self,
            "多个样品尚未保存",
            f"当前有 {len(dirty_rows)} 个样品尚未分别保存到工程文件。\n"
            "如需保存，请取消关闭并在样品列表中逐个右键保存。",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Discard

    def closeEvent(self, event):
        if not self._confirm_close_with_unsaved_samples():
            event.ignore()
            return
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
        self._close_prepared_project_snapshots()
        self._project_io_executor.shutdown(wait=False, cancel_futures=True)
        event.accept()
