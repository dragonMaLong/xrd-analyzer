import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QPointF, QStandardPaths, Qt
from PyQt5.QtWidgets import QApplication

from xrd_analyzer.core.analysis import calculate_peak_info
from xrd_analyzer.ui.app_window import XRDApp, XRDSample


_QT_APP = None


def _app():
    global _QT_APP
    QStandardPaths.setTestModeEnabled(True)
    _QT_APP = QApplication.instance() or QApplication([])
    return _QT_APP


def test_peak_partition_assigns_every_size_grid_point_once():
    D = np.arange(9.0)
    distribution = np.asarray([0.1, 1.0, 0.9, 0.8, 0.6, 0.05, 0.2, 0.7, 0.2])
    details, percentages = calculate_peak_info(distribution, np.asarray([1, 7]), D)

    assigned = np.concatenate([detail["indices"] for detail in details])
    np.testing.assert_array_equal(np.sort(assigned), np.arange(D.size))
    assert np.unique(assigned).size == D.size
    assert np.isclose(sum(percentages), 100.0)
    np.testing.assert_array_equal(details[0]["indices"], np.arange(6))
    np.testing.assert_array_equal(details[1]["indices"], np.arange(6, D.size))
    assert details[0]["right_boundary"] == D[5]
    assert details[1]["left_boundary"] == D[5]
    assert details[0]["right_boundary"] != 0.5 * (D[1] + D[7])


def test_component_focus_links_fitted_contour_to_particle_size_interval():
    _app()
    window = XRDApp()

    x = np.linspace(60.0, 61.0, 41)
    D = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0])
    centers = np.asarray([60.22, 60.28, 60.50, 60.72, 60.78])
    basis_k1 = np.column_stack(
        [np.exp(-0.5 * ((x - center) / 0.055) ** 2) for center in centers]
    )
    basis_k2 = np.zeros_like(basis_k1)
    weights = np.asarray([0.85, 0.55, 0.08, 0.72, 0.44])
    fitted = basis_k1.dot(weights)
    peak_details, _ = calculate_peak_info(weights, np.asarray([1, 3]), D)

    window.data_loaded = True
    window.results_ready = True
    window.active_peak_indices = [0]
    window.result_active_peak_indices = [0]
    window.x_data = x.copy()
    window.y_data = fitted.copy()
    window.x_segment = x.copy()
    window.y_segment_raw = fitted.copy()
    window.y_segment = fitted / float(np.max(fitted))
    window.background = np.zeros_like(x)
    window.D_range = D
    window.all_peak_info = [
        {
            "f_segment": weights,
            "volume_dist": weights,
            "basis_k1": basis_k1,
            "basis_k2": basis_k2,
            "peak_details": peak_details,
        }
    ]

    window.update_multi_peak_plots()
    first_curve_cache = window._fit_curve_data_cache([0])
    assert window._fit_curve_data_cache([0]) is first_curve_cache

    peak_label = window._fit_peak_labels[0]
    x_range, y_range = window._plot_range(window.fit_plot)
    assert float(peak_label.pos().x()) > window.peak_mu_sliders[0].get()
    assert float(peak_label.pos().y()) > y_range[1] - 0.03 * (y_range[1] - y_range[0])
    window.peak_mu_sliders[0].set(60.4, emit=False)
    window._sync_peak_line_value(0, 60.4)
    expected_offset = (x_range[1] - x_range[0]) * 0.006
    assert np.isclose(float(peak_label.pos().x()), 60.4 + expected_offset)

    first_key = (0, 0)
    second_key = (0, 1)
    assert set(window._fit_component_links) == {first_key, second_key}
    first = window._fit_component_links[first_key]
    second = window._fit_component_links[second_key]
    assert window._current_marker_label_state()["positions"] == {}

    moved_label = first["fit_label"]
    anchor_x, anchor_y = moved_label._xrd_anchor_pos
    moved_label.setPos(anchor_x + 0.18, anchor_y + 0.35)
    moved_label._xrd_connector_active = True
    moved_label._update_connector()
    marker_state = window._current_marker_label_state()
    assert marker_state["position_mode"] == "anchor-offset-v1"
    assert set(marker_state["positions"]) == {"component:0:0"}
    saved_layout = marker_state["positions"]["component:0:0"]
    assert np.isclose(saved_layout["offset_x"], 0.18)
    assert np.isclose(saved_layout["offset_y"], 0.35)
    assert not any(key.startswith("peak:") for key in marker_state["positions"])

    window.marker_label_state = marker_state
    window.update_multi_peak_plots()
    moved_label = window._fit_component_links[first_key]["fit_label"]
    anchor_x, anchor_y = moved_label._xrd_anchor_pos
    assert np.isclose(float(moved_label.pos().x()), anchor_x + 0.18)
    assert np.isclose(float(moved_label.pos().y()), anchor_y + 0.35)

    mismatched_state = {
        **marker_state,
        "positions": {
            "component:0:0": {
                **saved_layout,
                "anchor_x": float(saved_layout["anchor_x"]) + 2.0,
            }
        },
    }
    window._apply_marker_label_state(mismatched_state)
    default_x, default_y = moved_label._xrd_default_pos
    assert np.isclose(float(moved_label.pos().x()), default_x)
    assert np.isclose(float(moved_label.pos().y()), default_y)
    assert moved_label._xrd_connector_active is False

    window._apply_marker_label_state(
        {
            "visible": True,
            "positions": {
                "peak:0": {"x": 19.0, "y": 100.0, "connector": False},
                "component:0:0": {"x": 21.0, "y": 120.0, "connector": True},
            },
        }
    )
    assert window.marker_label_state["positions"] == {}
    assert np.isclose(float(window._fit_peak_labels[0].pos().x()), 60.4 + expected_offset)

    first = window._fit_component_links[first_key]
    second = window._fit_component_links[second_key]
    first_range_x = first["size_line"].getData()[0]
    second_range_x = second["size_line"].getData()[0]
    np.testing.assert_allclose(first_range_x, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(second_range_x, [6.0, 8.0, 10.0])
    assert first_range_x[-1] == second_range_x[0]
    peak_color = window._peak_color(0)
    assert first["size_line"].opts["pen"].color().name() == peak_color.lower()
    assert second["size_line"].opts["pen"].color().name() == peak_color.lower()
    assert window.actual_components[0]["fill"] == {}
    assert "单击锁定" not in first["interaction_label"]

    size_controller = window.size_plot._sample_curve_interaction_controller
    assert len(size_controller.entries) == 2
    size_controller.set_hover_sample(second["interaction_index"], propagate=True)
    assert window._fit_component_effective_key == second_key
    assert second["size_line"].opts["pen"].widthF() >= 3.6
    assert second["fit_line"].opts["pen"].widthF() >= 3.6
    assert first["fit_line"].opacity() < 0.2
    size_controller.clear_hover(propagate=True)
    assert window._fit_component_effective_key is None

    window._apply_fit_component_focus(first_key)
    assert first["fit_line"].opts["pen"].widthF() >= 3.6
    assert first["size_line"].opts["pen"].widthF() >= 3.6
    assert first["fit_line"].opacity() == 1.0
    assert first["size_fill"]["fill"].opacity() == 1.0
    assert second["fit_line"].opacity() < 0.2
    assert second["size_fill"]["fill"].opacity() < 0.2
    assert window.actual_components[0]["line"].opacity() < 0.2

    window._toggle_fit_component_lock(first_key)
    assert window._fit_component_locked_key == first_key
    window._on_fit_component_hover_index(second["interaction_index"])
    assert window._fit_component_effective_key == first_key

    old_size_line = first["size_line"]
    window._set_size_distribution_mode("number")
    first = window._fit_component_links[first_key]
    second = window._fit_component_links[second_key]
    assert first["size_line"] is not old_size_line
    assert first["size_line"].opts["pen"].widthF() >= 3.6
    assert second["size_fill"]["fill"].opacity() < 0.2

    window._toggle_fit_component_lock(first_key)
    assert window._fit_component_locked_key is None
    assert window._fit_component_effective_key is None
    assert second["fit_line"].opacity() == 1.0
    assert second["size_fill"]["fill"].opacity() == 1.0
    window.close()


def test_scatter_display_reduction_preserves_narrow_extrema():
    x = np.linspace(10.0, 20.0, 50000)
    y = np.sin(x)
    y[23457] = 250.0
    y[34567] = -180.0

    display_x, display_y = XRDApp._display_scatter_data(x, y, max_points=2000)

    assert display_x.size <= 2000
    assert np.max(display_y) == 250.0
    assert np.min(display_y) == -180.0
    assert x[23457] in display_x
    assert x[34567] in display_x


def test_size_legend_separates_total_inclusion_from_curve_visibility():
    _app()
    window = XRDApp()
    D = np.asarray([2.0, 4.0, 6.0])
    first_peak = np.asarray([1.0, 2.0, 1.0])
    second_peak = np.asarray([0.0, 1.0, 3.0])
    denominator = float(np.sum(first_peak) + np.sum(second_peak))
    window.D_range = D
    window.all_peak_info = [
        {"peak_id": 0, "volume_dist": first_peak, "f_segment": first_peak, "peak_details": []},
        {"peak_id": 1, "volume_dist": second_peak, "f_segment": second_peak, "peak_details": []},
    ]

    window._redraw_size_distribution_plot([0, 1])

    assert [window.legend_handles[key]["text"] for key in ("global", 0, 1)] == [
        "Total",
        "Peak1",
        "Peak2",
    ]
    original_total = (first_peak + second_peak) / denominator
    np.testing.assert_allclose(window.actual_components["global"]["y"], original_total)

    class _LegendClick:
        def __init__(self, x):
            self._pos = QPointF(float(x), 10.0)

        @staticmethod
        def button():
            return Qt.LeftButton

        def pos(self):
            return self._pos

        def accept(self):
            return None

        def ignore(self):
            return None

    # Hiding only changes the Peak curve; Total and checkbox state stay intact.
    window.legend_handles[0]["sample"].mouseClickEvent(_LegendClick(30))
    assert not window.actual_components[0]["line"].isVisible()
    assert window._size_component_included(0)
    np.testing.assert_allclose(window.actual_components["global"]["y"], original_total)
    window.legend_handles[0]["sample"].mouseClickEvent(_LegendClick(30))

    # The Total-row checkbox provides one-click select-none/select-all.
    window.legend_handles["global"]["sample"].mouseClickEvent(_LegendClick(5))
    assert not window._size_component_included(0)
    assert not window._size_component_included(1)
    assert not window.actual_components[0]["line"].isVisible()
    assert not window.actual_components[1]["line"].isVisible()
    assert window.legend_handles["global"]["sample"].check_state() == Qt.Unchecked
    np.testing.assert_allclose(window.actual_components["global"]["y"], np.zeros_like(D))

    window.legend_handles["global"]["sample"].mouseClickEvent(_LegendClick(5))
    assert window._size_component_included(0)
    assert window._size_component_included(1)
    assert window.actual_components[0]["line"].isVisible()
    assert window.actual_components[1]["line"].isVisible()
    np.testing.assert_allclose(window.actual_components["global"]["y"], original_total)

    window.legend_handles[0]["sample"].mouseClickEvent(_LegendClick(5))
    assert not window.actual_components[0]["line"].isVisible()
    assert window.legend_handles[0]["sample"].check_state() == Qt.Unchecked
    assert window.legend_handles["global"]["sample"].check_state() == Qt.PartiallyChecked
    np.testing.assert_allclose(window.actual_components["global"]["y"], second_peak / denominator)

    # An unchecked Peak cannot be shown from the line swatch alone.
    window.legend_handles[0]["sample"].mouseClickEvent(_LegendClick(30))
    assert not window.actual_components[0]["line"].isVisible()

    # Re-checking restores the curve; it can then be hidden without changing Total.
    window.legend_handles[0]["sample"].mouseClickEvent(_LegendClick(5))
    assert window._size_component_included(0)
    assert window.actual_components[0]["line"].isVisible()
    window.legend_handles[0]["sample"].mouseClickEvent(_LegendClick(30))
    assert not window.actual_components[0]["line"].isVisible()
    assert window._size_component_included(0)
    np.testing.assert_allclose(window.actual_components["global"]["y"], original_total)
    window.close()


def test_comparison_uses_included_single_sample_total_and_translucent_fill():
    _app()
    window = XRDApp()
    D = np.asarray([2.0, 4.0, 6.0])
    first_peak = np.asarray([1.0, 2.0, 1.0])
    excluded_peak = np.asarray([0.0, 1.0, 3.0])
    sample = XRDSample(
        path="sample.txt",
        x_data=np.asarray([60.0, 60.5, 61.0]),
        y_data=np.asarray([1.0, 3.0, 1.0]),
        name="sample",
        metadata={},
        project_dirty=False,
    )
    sample.results = {
        "D_range": D,
        "result_active_peak_indices": [0, 1],
        "all_peak_info": [
            {"peak_id": 0, "volume_dist": first_peak, "f_segment": first_peak},
            {"peak_id": 1, "volume_dist": excluded_peak, "f_segment": excluded_peak},
        ],
    }
    # Visibility no longer affects Total; only the inclusion checkboxes do.
    sample.size_visibility_state = {0: False, 1: True}
    sample.size_total_inclusion_state = {0: True, 1: False}
    window.samples = [sample]

    window.update_comparison_plots()

    controller = window.compare_size_plot._sample_curve_interaction_controller
    assert len(controller.entries) == 1
    entry = controller.entries[0]
    np.testing.assert_allclose(
        entry["item"].getData()[1],
        first_peak / float(np.sum(first_peak) + np.sum(excluded_peak)),
    )
    assert len(entry["associated_items"]) == 3
    fill_item = entry["associated_items"][-1]
    assert np.isclose(fill_item.brush().color().alphaF(), 0.12, atol=0.01)
    window.close()
