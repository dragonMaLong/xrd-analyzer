import os
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QItemSelectionModel, QStandardPaths, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QHeaderView,
    QTableWidgetSelectionRange,
)

from xrd_analyzer.io.project_file import load_project, save_project
import xrd_analyzer.ui.app_window as app_window_module
from xrd_analyzer.ui.app_window import XRDApp
from xrd_analyzer.ui.control_panel_mixin import SAMPLE_STATUS_PROGRESS_ROLE
from xrd_analyzer.ui.import_dialog import SUPPORTED_SUFFIXES


_QT_APP = None


def _app():
    global _QT_APP
    QStandardPaths.setTestModeEnabled(True)
    _QT_APP = QApplication.instance() or QApplication([])
    return _QT_APP


def _write_scan(path: Path, offset: float = 0.0):
    rows = ["内部样品名"]
    rows.extend(f"{60.0 + i * 0.1:.2f} {offset + value:.6f}" for i, value in enumerate([1, 3, 8, 3, 1]))
    path.write_text("\n".join(rows), encoding="utf-8")


def _wait_for_project_io(window: XRDApp, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while (
        window._project_save_jobs
        or window._file_load_jobs
        or window._prepared_project_snapshot_jobs
    ) and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    assert not window._project_save_jobs
    assert not window._file_load_jobs
    assert not window._prepared_project_snapshot_jobs


def test_sample_table_file_column_uses_disk_filename(tmp_path):
    _app()
    source = tmp_path / "真实文件名.txt"
    _write_scan(source)

    window = XRDApp()
    sample = window._load_sample_from_path(source)
    assert sample.name == "内部样品名"
    assert sample.metadata["file_name"] == source.name
    assert sample.metadata["sample_name"] == "内部样品名"
    window.samples = [sample]
    window.refresh_sample_table()

    header = window.sample_table.horizontalHeader()
    assert header.sectionResizeMode(window.sample_compare_col) == QHeaderView.Fixed
    assert header.sectionResizeMode(window.sample_file_col) == QHeaderView.Stretch
    assert header.sectionResizeMode(window.sample_status_col) == QHeaderView.Fixed
    assert window.sample_table.columnWidth(window.sample_status_col) == 64
    assert (
        window.sample_table.horizontalHeaderItem(window.sample_file_col).textAlignment()
        & int(Qt.AlignHCenter)
    )
    assert (
        window.sample_table.horizontalHeaderItem(window.sample_status_col).textAlignment()
        & int(Qt.AlignHCenter)
    )
    file_item = window.sample_table.item(0, window.sample_file_col)
    assert file_item.text() == source.name
    assert window._suggested_project_name(sample) == "真实文件名.xrdproj"
    sample.project_dirty = False
    window.close()


def test_calculation_results_support_cell_selection_and_copy():
    _app()
    window = XRDApp()
    table = window.sample_result_table

    assert table.selectionMode() == QAbstractItemView.ExtendedSelection
    assert table.selectionBehavior() == QAbstractItemView.SelectItems
    assert any(action.text() == "复制" for action in table.actions())

    table.setRowCount(2)
    values = (("Peak1", "12.34"), ("Peak2", "56.78"))
    for row, row_values in enumerate(values):
        for column, value in enumerate(row_values):
            table.setItem(row, column, window._detail_table_item(value))
    table.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)

    window._copy_table_selection_to_clipboard(table)
    assert QApplication.clipboard().text() == (
        "Peak\t峰面积\nPeak1\t12.34\nPeak2\t56.78"
    )

    stats_table = window.sample_stats_table
    stats_table.setRowCount(1)
    stats_table.setItem(0, 0, window._detail_table_item("5–10 nm"))
    stats_table.setItem(0, 1, window._detail_table_item("42.50%"))
    stats_table.setRangeSelected(QTableWidgetSelectionRange(0, 0, 0, 1), True)
    window._copy_table_selection_to_clipboard(stats_table)
    assert QApplication.clipboard().text() == "粒径区间\t百分比\n5–10 nm\t42.50%"
    window.close()


def test_calculation_start_exits_peak_placement_but_validation_failure_keeps_it(
    tmp_path, monkeypatch
):
    _app()
    source = tmp_path / "calculation-placement.txt"
    _write_scan(source)
    window = XRDApp()
    sample = window._load_sample_from_path(source)
    sample.project_dirty = False
    window.samples = [sample]
    window.refresh_sample_table()
    window.select_sample(0)
    QApplication.processEvents()

    warnings = []
    monkeypatch.setattr(
        app_window_module.messagebox,
        "showwarning",
        lambda title, message: warnings.append((title, message)),
    )
    window._set_peak_placement_mode(True)
    window.active_peak_indices = []
    window.compute_thread("fast")
    assert warnings
    assert window._peak_placement_mode is True
    assert window.add_peak_button.isChecked()

    started = []

    class _StartedThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.args, self.daemon))

    monkeypatch.setattr(app_window_module.threading, "Thread", _StartedThread)
    monkeypatch.setattr(window, "_collect_fit_params", lambda: {"mu_centers": [60.2]})
    window.active_peak_indices = [0]
    window.compute_thread("fast")

    assert len(started) == 1
    assert window._fit_worker_running is True
    assert window._peak_placement_mode is False
    assert not window.add_peak_button.isChecked()
    assert window.preview_plot.cursor().shape() == Qt.ArrowCursor
    assert all(not guide.isVisible() for guide in window._peak_placement_guides.values())

    window._fit_worker_running = False
    window._sample_status_tasks.clear()
    for button in (window.btn_fast, window.btn_fine):
        button.setEnabled(True)
    sample.project_dirty = False
    window.close()


def test_baseline_is_always_visible_and_peak_placement_only_disables_editing(tmp_path):
    _app()
    source = tmp_path / "baseline-default.txt"
    _write_scan(source)

    window = XRDApp()
    sample = window._load_sample_from_path(source)
    window.samples = [sample]
    window.refresh_sample_table()
    window.select_sample(0)
    QApplication.processEvents()

    assert window.btn_manual_baseline.text() == "基线"
    assert window.btn_manual_baseline.isChecked()
    assert window.btn_manual_baseline.toolTip() == "左键添加锚点，右键删除锚点"
    assert window.manual_baseline_enabled is True
    assert window.manual_baseline_editing is True
    assert window._manual_baseline_curve_item is not None
    assert window._manual_baseline_curve_item.toolTip() == "左键添加锚点，右键删除锚点"
    active_pen = window._manual_baseline_curve_item.opts["pen"]
    assert active_pen.widthF() > 2.0
    assert active_pen.style() == Qt.SolidLine
    assert len(window._manual_baseline_anchor_items) >= 2

    window._add_manual_baseline_anchor(60.2, 4.0)
    state = window._current_manual_baseline_state()
    baseline_before = window._compute_background_for_segment(
        window.x_data,
        window.y_data,
        float(np.min(window.x_data)),
        float(np.max(window.x_data)),
        state,
    )
    sample.result_is_current = True
    window._set_peak_placement_mode(True)

    assert not window.btn_manual_baseline.isChecked()
    assert window.manual_baseline_enabled is True
    assert window.manual_baseline_editing is False
    assert window._manual_baseline_curve_item is not None
    inactive_pen = window._manual_baseline_curve_item.opts["pen"]
    assert inactive_pen.widthF() < active_pen.widthF()
    assert inactive_pen.style() == Qt.DashLine
    assert window._manual_baseline_anchor_items == []
    assert sample.result_is_current is True
    baseline_after = window._compute_background_for_segment(
        window.x_data,
        window.y_data,
        float(np.min(window.x_data)),
        float(np.max(window.x_data)),
        window._current_manual_baseline_state(),
    )
    np.testing.assert_allclose(baseline_after, baseline_before)

    sample.project_dirty = False
    window.close()


def test_baseline_curve_glows_and_drag_creates_anchor(tmp_path, monkeypatch):
    _app()
    source = tmp_path / "baseline-drag.txt"
    _write_scan(source)

    window = XRDApp()
    sample = window._load_sample_from_path(source)
    window.samples = [sample]
    window.refresh_sample_table()
    window.select_sample(0)
    QApplication.processEvents()

    curve = window._manual_baseline_curve_item
    assert curve is not None
    assert curve.opts["mouseWidth"] == 8
    normal_width = curve.opts["pen"].widthF()

    class HoverEvent:
        def __init__(self, exiting=False):
            self._exiting = exiting
            self.accepted_drag = False

        def isExit(self):
            return self._exiting

        def acceptDrags(self, button):
            self.accepted_drag = button == Qt.LeftButton

    hover = HoverEvent()
    curve.hoverEvent(hover)
    assert hover.accepted_drag
    assert curve.cursor().shape() == Qt.ArrowCursor
    assert curve.opts["pen"].widthF() > normal_width
    assert curve.opts["shadowPen"] is not None
    curve.hoverEvent(HoverEvent(exiting=True))
    assert curve.opts["pen"].widthF() == normal_width
    assert curve.opts["shadowPen"] is None

    class DragEvent:
        def __init__(self, position, *, start=False, finish=False):
            self.position = position
            self._start = start
            self._finish = finish

        def isStart(self):
            return self._start

        def isFinish(self):
            return self._finish

    monkeypatch.setattr(
        window,
        "_baseline_curve_event_position",
        lambda event: event.position,
    )
    curve_x, curve_y = curve.getData()
    x_start = float(curve_x[len(curve_x) // 2])
    y_start = float(curve_y[len(curve_y) // 2])
    initial_count = len(window.manual_baseline_user_points)
    sample.result_is_current = True

    assert window._drag_manual_baseline_from_curve(
        curve,
        DragEvent((x_start, y_start), start=True),
    )
    drag_id = window._manual_baseline_drag_anchor_id
    assert drag_id is not None
    assert len(window.manual_baseline_user_points) == initial_count + 1

    x_end = x_start + 0.04
    y_end = y_start + 2.5
    assert window._drag_manual_baseline_from_curve(
        curve,
        DragEvent((x_end, y_end), finish=True),
    )
    dragged = next(
        point for point in window.manual_baseline_user_points
        if int(point["id"]) == int(drag_id)
    )
    assert dragged["x"] == x_end
    assert dragged["y"] == y_end
    assert window._manual_baseline_drag_anchor is None
    assert window._manual_baseline_drag_anchor_id is None
    assert window._manual_baseline_curve_item is not curve
    assert sample.result_is_current is False

    sample.project_dirty = False
    window.close()


def test_peak_placement_keeps_range_and_existing_peak_lines_draggable(tmp_path, monkeypatch):
    _app()
    source = tmp_path / "placement-adjustments.txt"
    _write_scan(source)

    window = XRDApp()
    sample = window._load_sample_from_path(source)
    window.samples = [sample]
    window.refresh_sample_table()
    window.select_sample(0)
    QApplication.processEvents()

    window._set_peak_placement_mode(True)
    assert window._peak_placement_mode is True
    assert window.preview_range_region.movable is False
    assert all(line.movable for line in window.preview_range_region.lines)

    preview_peak_line = next(line for line in window.peak_mu_lines_preview if line is not None)
    fit_peak_line = next(line for line in window.peak_mu_lines_axes0 if line is not None)
    assert preview_peak_line.movable is True
    assert fit_peak_line.movable is True
    assert preview_peak_line.cursor().shape() == Qt.SizeHorCursor
    assert fit_peak_line.cursor().shape() == Qt.SizeHorCursor

    window.line_min.setValue(60.05)
    QApplication.processEvents()
    assert np.isclose(window.slider_min.get(), 60.05)

    original_peak_count = len(window.peak_mu_sliders)
    preview_peak_line.setValue(60.3)
    QApplication.processEvents()
    assert np.isclose(window.peak_mu_sliders[0].get(), 60.3)
    assert np.isclose(fit_peak_line.value(), 60.3)
    assert len(window.peak_mu_sliders) == original_peak_count

    class LineClickEvent:
        def __init__(self):
            self.accepted = False

        def button(self):
            return Qt.LeftButton

        def double(self):
            return False

        def scenePos(self):
            return None

        def accept(self):
            self.accepted = True

    monkeypatch.setattr(window, "_peak_placement_adjustment_line_at", lambda _plot, _pos: True)
    event = LineClickEvent()
    assert window._try_add_peak_from_plot_click(window.preview_plot, event)
    assert event.accepted
    assert len(window.peak_mu_sliders) == original_peak_count

    sample.project_dirty = False
    window.close()


def test_rapid_sample_switch_coalesces_expensive_rendering(tmp_path, monkeypatch):
    _app()
    first_path = tmp_path / "first-switch.txt"
    second_path = tmp_path / "second-switch.txt"
    _write_scan(first_path, offset=1.0)
    _write_scan(second_path, offset=2.0)

    window = XRDApp()
    first = window._load_sample_from_path(first_path)
    second = window._load_sample_from_path(second_path)
    for sample in (first, second):
        sample.results = {"current_rfit_percent": None}
        sample.status = "complete"
        sample.project_dirty = False
    window.samples = [first, second]
    window.refresh_sample_table()

    render_calls = []
    monkeypatch.setattr(
        window,
        "update_preview",
        lambda _value=None: render_calls.append(("preview", window.active_sample_index)),
    )
    monkeypatch.setattr(
        window,
        "update_multi_peak_plots",
        lambda: render_calls.append(("fit", window.active_sample_index)),
    )
    monkeypatch.setattr(
        window,
        "update_result_table",
        lambda: render_calls.append(("table", window.active_sample_index)),
    )

    window.select_sample(0)
    first_peak_row = window.peak_rows[0]
    window.select_sample(1)
    assert window.peak_rows[0] is first_peak_row
    assert render_calls == []

    QApplication.processEvents()
    assert render_calls == [("preview", 1), ("fit", 1), ("table", 1)]
    for sample in window.samples:
        sample.project_dirty = False
    window.close()


def test_imported_projects_use_each_project_filename_in_sample_table(tmp_path):
    _app()
    source = tmp_path / "1.txt"
    first_project = tmp_path / "1-1.xrdproj"
    second_project = tmp_path / "1-2.xrdproj"
    _write_scan(source)

    source_window = XRDApp()
    sample = source_window._load_sample_from_path(source)
    record = source_window._sample_to_project_record(sample)
    save_project(first_project, [record], project_uuid=str(uuid.uuid4()))
    save_project(second_project, [record], project_uuid=str(uuid.uuid4()))
    sample.project_dirty = False
    source_window.close()

    window = XRDApp()
    window.load_files([str(first_project), str(second_project)])
    _wait_for_project_io(window)

    assert len(window.samples) == 2
    displayed_names = [
        window.sample_table.item(row, window.sample_file_col).text()
        for row in range(window.sample_table.rowCount())
    ]
    assert displayed_names == [first_project.name, second_project.name]
    assert window._suggested_project_name(window.samples[0]) == first_project.name
    assert window._suggested_project_name(window.samples[1]) == second_project.name
    assert window.samples[0].path == window.samples[1].path == str(source)
    assert window.samples[0].sample_id != window.samples[1].sample_id
    assert window.samples[0].metadata["file_name"] == first_project.name
    assert window.samples[1].metadata["file_name"] == second_project.name
    assert window.samples[0].metadata["sample_name"] == "内部样品名"
    assert window.samples[1].metadata["sample_name"] == "内部样品名"

    # A project's embedded source path is provenance, not the row's import
    # identity.  The original data must remain independently importable.
    window.load_files([str(source)])
    _wait_for_project_io(window)
    assert len(window.samples) == 3
    assert window.sample_table.item(2, window.sample_file_col).text() == source.name
    assert window.samples[2].project_path == ""
    assert window.samples[2].results == {}
    for imported_sample in window.samples:
        imported_sample.project_dirty = False
    window.close()


def test_project_row_appears_immediately_with_radial_load_progress(tmp_path, monkeypatch):
    _app()
    source = tmp_path / "immediate-source.txt"
    project = tmp_path / "immediate-project.xrdproj"
    _write_scan(source)

    source_window = XRDApp()
    sample = source_window._load_sample_from_path(source)
    sample.status = "complete"
    sample.results = {"current_rfit_percent": None}
    save_project(project, [source_window._sample_to_project_record(sample)])
    sample.project_dirty = False
    source_window.close()

    entered = threading.Event()
    release = threading.Event()
    original_load_project = app_window_module.load_project

    def delayed_load_project(*args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback(18, "读取样品参数与标记")
        entered.set()
        assert release.wait(2.0)
        return original_load_project(*args, **kwargs)

    monkeypatch.setattr(app_window_module, "load_project", delayed_load_project)
    window = XRDApp()
    window.load_files([str(project)])

    assert window.sample_table.rowCount() == 1
    assert window.sample_table.item(0, window.sample_file_col).text() == project.name
    assert window.sample_table.item(0, window.sample_status_col).data(
        SAMPLE_STATUS_PROGRESS_ROLE
    ) == 0

    assert entered.wait(1.0)
    QApplication.processEvents()
    status_item = window.sample_table.item(0, window.sample_status_col)
    assert status_item.data(SAMPLE_STATUS_PROGRESS_ROLE) == 18
    assert "读取样品参数与标记" in status_item.toolTip()

    release.set()
    _wait_for_project_io(window)
    assert len(window.samples) == 1
    assert window.sample_table.item(0, window.sample_file_col).text() == project.name
    assert window.sample_table.item(0, window.sample_status_col).data(
        SAMPLE_STATUS_PROGRESS_ROLE
    ) is None
    assert not window.sample_table.item(0, window.sample_status_col).icon().isNull()
    window.close()


def test_window_project_save_and_restore_parameters(tmp_path):
    _app()
    source = tmp_path / "sample.txt"
    _write_scan(source, offset=float(uuid.uuid4().int % 1000))
    project = tmp_path / "saved.xrdproj"

    window = XRDApp()
    assert window.menuWidget() is None
    assert not hasattr(window, "btn_project")
    assert window.sample_table.selectionMode() == QAbstractItemView.ExtendedSelection
    assert ".xrdproj" in SUPPORTED_SUFFIXES
    window.samples = [window._load_sample_from_path(source)]
    window.select_sample(0)
    window.peak_mu_sliders[0].set(60.2, emit=False)
    window._save_current_peak_states()
    window.slider_alpha.set(2.5, emit=False)
    window._save_current_parameter_state()
    window.samples[0].project_path = str(project)

    assert window.save_project_file()
    _wait_for_project_io(window)
    assert window.progress_bar.value() == 100
    assert window.progress_label.text() == "工程保存完成"
    assert len(load_project(project)["samples"]) == 1
    window.peak_mu_sliders[0].set(60.4, emit=False)
    window._save_current_peak_states()
    window.samples[0].project_dirty = False
    window.close()

    restored_window = XRDApp()
    restored_window.load_files([str(project)])
    _wait_for_project_io(restored_window)
    assert restored_window.progress_bar.value() == 100
    assert restored_window.progress_label.text() == "工程读取完成"
    assert len(restored_window.samples) == 1
    assert restored_window.peak_mu_sliders[0].get() == 60.2
    assert restored_window.slider_alpha.get() == 2.5
    assert restored_window.samples[0].project_dirty is False
    restored_window.close()


def test_sample_context_menu_and_multi_row_delete(tmp_path):
    _app()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    third = tmp_path / "third.txt"
    _write_scan(first, offset=1.0)
    _write_scan(second, offset=2.0)
    _write_scan(third, offset=3.0)

    window = XRDApp()
    window.samples = [window._load_sample_from_path(path) for path in (first, second, third)]
    remaining_id = window.samples[1].sample_id
    window.refresh_sample_table()
    window.select_sample(0)

    single_project = tmp_path / "second.xrdproj"
    window.samples[1].project_path = str(single_project)
    assert window.save_project_file(sample_index=1)
    _wait_for_project_io(window)
    saved_records = load_project(single_project)["samples"]
    assert len(saved_records) == 1
    assert saved_records[0]["sample_id"] == remaining_id

    single_menu = window._build_sample_context_menu([0])
    assert [action.text() for action in single_menu.actions() if not action.isSeparator()] == [
        "删除",
        "保存工程",
        "另存为工程",
    ]
    multi_menu = window._build_sample_context_menu([0, 2])
    assert [action.text() for action in multi_menu.actions() if not action.isSeparator()] == ["删除"]

    selection = window.sample_table.selectionModel()
    selection.clearSelection()
    selection.select(
        window.sample_table.model().index(0, 1),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
    )
    selection.select(
        window.sample_table.model().index(2, 1),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
    )
    assert window._selected_sample_rows() == [0, 2]
    window._remove_sample_rows(window._selected_sample_rows())
    assert len(window.samples) == 1
    assert window.samples[0].sample_id == remaining_id
    assert first.is_file() and second.is_file() and third.is_file()
    window.samples[0].project_dirty = False
    window.close()


def test_project_save_runs_in_background_and_preserves_new_dirty_state(tmp_path, monkeypatch):
    _app()
    source = tmp_path / "async.txt"
    target = tmp_path / "async.xrdproj"
    _write_scan(source, offset=4.0)

    window = XRDApp()
    window.samples = [window._load_sample_from_path(source)]
    window.refresh_sample_table()
    window.select_sample(0)
    window.samples[0].project_path = str(target)

    entered = threading.Event()
    release = threading.Event()
    original_save_project = app_window_module.save_project

    def delayed_save(*args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback(42, "写入工程数据")
        entered.set()
        assert release.wait(2.0)
        return original_save_project(*args, **kwargs)

    monkeypatch.setattr(app_window_module, "save_project", delayed_save)
    started_at = time.monotonic()
    assert window.save_project_file()
    assert time.monotonic() - started_at < 0.5
    assert entered.wait(1.0)
    QApplication.processEvents()
    assert window._project_save_jobs
    status_item = window.sample_table.item(0, window.sample_status_col)
    assert status_item.data(SAMPLE_STATUS_PROGRESS_ROLE) == 42
    assert "保存工程" in status_item.toolTip()
    assert "写入工程数据" in status_item.toolTip()

    window._mark_project_dirty()
    release.set()
    _wait_for_project_io(window)
    assert target.is_file()
    assert window.samples[0].project_dirty is True
    window.samples[0].project_dirty = False
    window.close()


def test_calculation_progress_uses_sample_status_ring_then_complete_icon(tmp_path):
    _app()
    source = tmp_path / "calculation-status.txt"
    _write_scan(source)

    window = XRDApp()
    sample = window._load_sample_from_path(source)
    sample.status = "complete"
    sample.result_is_current = True
    window.samples = [sample]
    window.refresh_sample_table()

    task_token = str(uuid.uuid4())
    window._fit_worker_running = True
    window._fit_task_sample_id = sample.sample_id
    window._fit_task_token = task_token
    window._fit_task_operation = "精细计算"
    window._set_sample_status_progress(
        sample.sample_id,
        0,
        "精细计算",
        "准备计算",
        task_token=task_token,
    )
    window.progress_bar.setValue(47)

    status_item = window.sample_table.item(0, window.sample_status_col)
    assert status_item.data(SAMPLE_STATUS_PROGRESS_ROLE) == 47
    assert "精细计算" in status_item.toolTip()

    window._fit_worker_running = False
    window._finish_sample_calculation_status(sample.sample_id, task_token, True)
    assert status_item.data(SAMPLE_STATUS_PROGRESS_ROLE) == 100
    time.sleep(0.2)
    QApplication.processEvents()
    assert status_item.data(SAMPLE_STATUS_PROGRESS_ROLE) is None
    assert not status_item.icon().isNull()

    sample.project_dirty = False
    window.close()


def test_manual_save_reuses_precompressed_calculation_snapshot(tmp_path, monkeypatch):
    _app()
    source = tmp_path / "prepared.txt"
    target = tmp_path / "prepared.xrdproj"
    _write_scan(source, offset=4.5)

    window = XRDApp()
    sample = window._load_sample_from_path(source)
    sample.status = "complete"
    sample.results = {
        "D_range": np.linspace(1.0, 10.0, 50),
        "best_f_total": np.linspace(0.0, 1.0, 50),
    }
    window.samples = [sample]
    window.refresh_sample_table()
    window._mark_project_dirty(sample)
    window._queue_prepared_project_snapshot(sample)
    _wait_for_project_io(window)
    assert window._prepared_snapshot_for_revision(
        sample.sample_id,
        sample.project_revision,
        sample.project_uuid,
    )

    sample.project_path = str(target)
    original_save_project = app_window_module.save_project

    def unexpected_full_save(*args, **kwargs):
        raise AssertionError("matching prepared snapshot should avoid full recompression")

    monkeypatch.setattr(app_window_module, "save_project", unexpected_full_save)
    assert window.save_project_file(sample_index=0)
    _wait_for_project_io(window)

    restored = load_project(target)["samples"][0]
    assert np.array_equal(restored["results"]["D_range"], sample.results["D_range"])
    assert sample.project_dirty is False

    # Any later state revision makes that snapshot ineligible and falls back
    # to a complete authoritative save.
    second_target = tmp_path / "prepared-after-change.xrdproj"
    full_save_calls = []

    def tracked_full_save(*args, **kwargs):
        full_save_calls.append(args[0])
        return original_save_project(*args, **kwargs)

    monkeypatch.setattr(app_window_module, "save_project", tracked_full_save)
    window._mark_project_dirty(sample)
    sample.project_path = str(second_target)
    assert window.save_project_file(sample_index=0)
    _wait_for_project_io(window)
    assert full_save_calls == [str(second_target.resolve())]
    assert second_target.is_file()
    window.close()


def test_project_record_replaces_basis_matrices_with_compact_display_curves(tmp_path):
    _app()
    source = tmp_path / "compact-source.txt"
    project = tmp_path / "compact.xrdproj"
    _write_scan(source)

    window = XRDApp()
    sample = window._load_sample_from_path(source)
    x = np.linspace(60.0, 61.0, 101)
    D = np.linspace(1.0, 10.0, 50)
    rng = np.random.default_rng(1234)
    basis_k1 = rng.random((x.size, D.size))
    basis_k2 = rng.random((x.size, D.size)) * 0.25
    weights = np.linspace(0.1, 1.0, D.size)
    y_segment = np.linspace(2.0, 5.0, x.size)
    background = np.linspace(0.2, 0.3, x.size)
    details = [
        {
            "center": 5.0,
            "percentage": 100.0,
            "indices": np.arange(D.size, dtype=int),
            "left_boundary": float(D[0]),
            "right_boundary": float(D[-1]),
            "area": 1.0,
            "pct_global": 100.0,
        }
    ]
    sample.x_data = x
    sample.y_data = y_segment + background
    # The synthetic arrays replace the originally imported scan, so let the
    # project serializer calculate a matching fingerprint for these arrays.
    sample.data_fingerprint = ""
    sample.status = "complete"
    sample.result_is_current = True
    sample.size_visibility_state = {0: False}
    sample.size_total_inclusion_state = {0: True}
    sample.analysis_state = {"angle_min": 60.0, "angle_max": 61.0}
    sample.peak_states = [{"checked": True, "value": 60.5, "color": "#FF0000", "visible": True}]
    sample.results = {
        "best_f_total": weights,
        "all_basis_k1": [basis_k1],
        "all_basis_k2": [basis_k2],
        "D_range": D,
        "all_peak_info": [
            {
                "peak_id": 0,
                "f_segment": weights,
                "volume_dist": weights,
                "number_dist": weights,
                "normalized_dist": weights,
                "peak_details": details,
                "color": "#FF0000",
                "basis_k1": basis_k1,
                "basis_k2": basis_k2,
            }
        ],
        "global_max_component_area": 1.0,
        "result_active_peak_indices": [0],
        "result_regularization_method": "l2",
        "result_peak_kernel": "pearson7",
        "x_segment": x,
        "y_segment_raw": y_segment + background,
        "y_segment": y_segment,
        "background": background,
        "fit_quality_history": [],
        "current_rfit_percent": 1.5,
    }

    record = window._sample_to_project_record(sample)
    compact_results = record["results"]
    assert "all_basis_k1" not in compact_results
    assert "all_basis_k2" not in compact_results
    assert "basis_k1" not in compact_results["all_peak_info"][0]
    assert "basis_k2" not in compact_results["all_peak_info"][0]
    snapshot = compact_results["fit_curve_snapshot"]
    expected_signal = (basis_k1.dot(weights) + basis_k2.dot(weights)) * np.max(y_segment)
    np.testing.assert_allclose(snapshot["peaks"][0]["signal"], expected_signal)

    save_project(project, [record])
    restored_record = load_project(project)["samples"][0]
    assert "all_basis_k1" not in restored_record["results"]
    assert restored_record["size_visibility_state"] == {0: False}
    assert restored_record["size_total_inclusion_state"] == {0: True}
    assert project.stat().st_size < basis_k1.nbytes + basis_k2.nbytes

    restored_sample = window._sample_from_project_record(restored_record)
    window.samples = [restored_sample]
    window.refresh_sample_table()
    window.select_sample(0)
    QApplication.processEvents()
    curve_data = window._fit_curve_data_cache([0])
    np.testing.assert_allclose(curve_data["peak_specs"][0]["signal"], expected_signal)
    assert window.all_basis_k1 == []
    assert window.all_basis_k2 == []
    assert window._size_component_visible(0) is False
    assert window._size_component_included(0) is True

    legacy_record = dict(restored_record)
    legacy_record.pop("size_total_inclusion_state")
    legacy_sample = window._sample_from_project_record(legacy_record)
    assert legacy_sample.size_total_inclusion_state == {0: False}

    restored_sample.project_dirty = False
    window.close()


def test_raw_file_load_runs_in_background_with_progress(tmp_path):
    _app()
    source = tmp_path / "background.raw.txt"
    _write_scan(source, offset=5.0)

    window = XRDApp()
    window.load_files([str(source)])
    assert window.progress_label.text().startswith("读取数据：")
    _wait_for_project_io(window)
    assert len(window.samples) == 1
    assert window.progress_bar.value() == 100
    assert window.progress_label.text() == "数据读取完成"
    window.samples[0].project_dirty = False
    window.close()


def test_raw_import_never_restores_previous_results_for_matching_data(tmp_path):
    _app()
    first = tmp_path / "before.txt"
    second = tmp_path / "after.txt"
    _write_scan(first, offset=float(uuid.uuid4().int % 10000))
    second.write_bytes(first.read_bytes())

    window = XRDApp()
    sample = window._load_sample_from_path(first)
    window.samples = [sample]
    window.select_sample(0)
    sample.status = "complete"
    sample.results = {"saved_distribution": np.array([0.2, 0.8])}
    sample.parameter_state = window._current_parameter_state()
    sample.analysis_state = window._current_analysis_state()
    sample.peak_states = window._current_peak_states()
    sample.baseline_state = window._current_manual_baseline_state()
    sample.result_signature = window._result_signature_for_sample(sample)
    sample.result_is_current = True

    restored = window._load_sample_from_path(second)
    assert restored.sample_id != sample.sample_id
    assert restored.path == str(second)
    assert restored.results == {}
    assert restored.status == "pending"
    assert restored.result_is_current is False
    sample.project_dirty = False
    window.close()
