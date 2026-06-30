"""
ui/plot_panel_mixin.py
----------------------
PyQt5 right-side plotting mixin.

The XRD plots use pyqtgraph, matching the BET application's plotting stack and
mouse interaction model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from .qt_controls import TextValue


pg.setConfigOptions(antialias=True, useOpenGL=False)

COMPONENT_COLORS = (
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#f97316",
    "#0891b2",
    "#4f46e5",
    "#65a30d",
    "#db2777",
    "#7c3aed",
    "#14b8a6",
)


def _plain_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:,.0f}"
    if abs_value >= 10:
        return f"{value:,.1f}".rstrip("0").rstrip(".")
    if abs_value >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if abs_value >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if abs_value == 0:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


class PlainNumberAxis(pg.AxisItem):
    """BET-style axis labels: compact plain numbers and fewer tick labels."""

    def tickStrings(self, values, scale, spacing):
        labels = []
        axis_length = self.geometry().height() if self.orientation in {"left", "right"} else self.geometry().width()
        max_labels = max(3, int(max(1, axis_length) // 92))
        step = max(1, int(np.ceil(len(values) / max_labels))) if len(values) else 1
        for index, value in enumerate(values):
            if index % step:
                labels.append("")
                continue
            axis_value = float(value) * scale
            if getattr(self, "logMode", False):
                axis_value = 10.0**axis_value
            labels.append(_plain_number(axis_value))
        return labels


class XRDPlotWidget(pg.PlotWidget):
    """Small PlotWidget wrapper so pyqtgraph context menus stay enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackground("w")
        self.setMenuEnabled(True)
        self.showGrid(x=True, y=True, alpha=0.25)


SPLITTER_STYLE = """
QSplitter::handle:horizontal {
    background: #e5e7eb;
    margin: 0 2px;
}
QSplitter::handle:horizontal:hover {
    background: #93c5fd;
}
QSplitter::handle:vertical {
    background: #e5e7eb;
    margin: 2px 0;
}
QSplitter::handle:vertical:hover {
    background: #93c5fd;
}
"""


def _configure_splitter(splitter: QtWidgets.QSplitter, handle_width: int = 8) -> QtWidgets.QSplitter:
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(handle_width)
    splitter.setStyleSheet(SPLITTER_STYLE)
    return splitter


class CollapsibleAnalysisPane(QtWidgets.QFrame):
    """BET-style collapsible analysis-options pane."""

    COLLAPSED_WIDTH = 38
    MIN_EXPANDED_WIDTH = 198
    MAX_EXPANDED_WIDTH = 300
    DEFAULT_EXPANDED_WIDTH = 224

    def __init__(self, content: QtWidgets.QWidget, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.content = content
        self._collapsed = False
        self._expanded_width = self.DEFAULT_EXPANDED_WIDTH
        self.setObjectName("collapsibleAnalysisPane")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumHeight(48)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setObjectName("analysisScrollArea")
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setMinimumHeight(0)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea#analysisScrollArea { background: transparent; border: 0; }
            """
        )
        self.scroll_area.setWidget(content)
        layout.addWidget(self.scroll_area, 1)

        self.toggle_button = QtWidgets.QToolButton(self)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setArrowType(Qt.LeftArrow)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setFixedSize(24, 24)
        self.toggle_button.setToolTip("隐藏参数栏")
        self.toggle_button.clicked.connect(self.toggle_collapsed)

        self.set_expanded_width(self.DEFAULT_EXPANDED_WIDTH, request=False)
        self._sync_content_height()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._position_toggle_button()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._collapsed:
            width = self._clamp_expanded_width(self.width())
            if abs(width - self._expanded_width) > 1:
                self._expanded_width = width
                self.content.setFixedWidth(width)
            self._sync_content_height()
        self._position_toggle_button()

    def expanded_width(self) -> int:
        return int(self._expanded_width)

    def remember_current_width(self) -> None:
        if not self._collapsed:
            self.set_expanded_width(self.width(), request=False)

    def set_expanded_width(self, width: int, *, request: bool = True) -> None:
        self._expanded_width = self._clamp_expanded_width(width)
        if self._collapsed:
            return
        self.content.setVisible(True)
        self.content.setFixedWidth(self._expanded_width)
        self._sync_content_height()
        self.setMinimumWidth(self.MIN_EXPANDED_WIDTH)
        self.setMaximumWidth(self.MAX_EXPANDED_WIDTH)
        self.updateGeometry()
        self._position_toggle_button()
        if request:
            self._request_splitter_width(self._expanded_width)

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        if collapsed:
            self._expanded_width = self._clamp_expanded_width(self.width())
            self.scroll_area.setVisible(False)
            self.setMinimumWidth(self.COLLAPSED_WIDTH)
            self.setMaximumWidth(self.COLLAPSED_WIDTH)
            self.toggle_button.setArrowType(Qt.RightArrow)
            self.toggle_button.setToolTip("展开参数栏")
            target_width = self.COLLAPSED_WIDTH
        else:
            self.scroll_area.setVisible(True)
            self.content.setFixedWidth(self._expanded_width)
            self._sync_content_height()
            self.setMinimumWidth(self.MIN_EXPANDED_WIDTH)
            self.setMaximumWidth(self.MAX_EXPANDED_WIDTH)
            self.toggle_button.setArrowType(Qt.LeftArrow)
            self.toggle_button.setToolTip("隐藏参数栏")
            target_width = self._expanded_width
        self._collapsed = collapsed
        self.updateGeometry()
        self._position_toggle_button()
        self._request_splitter_width(target_width)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _position_toggle_button(self) -> None:
        x = max(7, self.width() - self.toggle_button.width() - 5)
        self.toggle_button.move(x, 5)
        self.toggle_button.raise_()

    def _sync_content_height(self) -> None:
        if self._collapsed:
            return
        self.content.adjustSize()
        height = max(self.content.sizeHint().height(), self.content.minimumSizeHint().height())
        self.content.setMinimumHeight(height)
        self.content.resize(self._expanded_width, height)

    def _request_splitter_width(self, target_width: int) -> None:
        splitter = self.parent()
        while splitter is not None and not isinstance(splitter, QtWidgets.QSplitter):
            splitter = splitter.parent()
        if splitter is None:
            return
        index = splitter.indexOf(self)
        sizes = splitter.sizes()
        if index < 0 or index >= len(sizes) or len(sizes) < 2:
            return
        total = max(sum(sizes), target_width + 80)
        sizes[index] = int(target_width)
        other = 1 if index == 0 else index - 1
        sizes[other] = max(80, total - int(target_width))
        splitter.setSizes(sizes)

    def _clamp_expanded_width(self, width: int) -> int:
        return max(self.MIN_EXPANDED_WIDTH, min(int(width), self.MAX_EXPANDED_WIDTH))


class _LegendToggleButton(QtWidgets.QToolButton):
    """BET-style floating eye button for showing, hiding, and moving legends."""

    def __init__(self, plot: pg.PlotWidget) -> None:
        super().__init__(plot)
        self._plot = plot
        self.setCheckable(True)
        self.setChecked(True)
        self.setAutoRaise(True)
        self.setFixedSize(24, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("隐藏图例")
        self._press_global_pos: QtCore.QPoint | None = None
        self._press_button_pos: QtCore.QPoint | None = None
        self._dragging_button = False
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        _set_plot_legend_visible(self._plot, checked)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._press_global_pos = self._event_global_pos(event)
        self._press_button_pos = QtCore.QPoint(self.pos())
        self._dragging_button = False
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if not (event.buttons() & Qt.LeftButton) or self._press_global_pos is None or self._press_button_pos is None:
            super().mouseMoveEvent(event)
            return
        delta = self._event_global_pos(event) - self._press_global_pos
        if not self._dragging_button and delta.manhattanLength() < QtWidgets.QApplication.startDragDistance():
            event.accept()
            return
        self._dragging_button = True
        self._move_to(self._press_button_pos + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._dragging_button:
            self._dragging_button = False
            self._press_global_pos = None
            self._press_button_pos = None
            event.accept()
            return
        self._press_global_pos = None
        self._press_button_pos = None
        self.setChecked(not self.isChecked())
        event.accept()

    def _move_to(self, position: QtCore.QPoint) -> None:
        margin = 4
        x = max(margin, min(int(position.x()), max(margin, self._plot.width() - self.width() - margin)))
        y = max(margin, min(int(position.y()), max(margin, self._plot.height() - self.height() - margin)))
        self.move(x, y)
        self.raise_()
        legend = getattr(self._plot.plotItem, "legend", None)
        if legend is not None and legend.isVisible():
            _move_legend_to_toggle_anchor(self._plot)
        else:
            setattr(self._plot, "_legend_hidden_toggle_anchor", QtCore.QPoint(self.pos()))

    @staticmethod
    def _event_global_pos(event) -> QtCore.QPoint:
        if hasattr(event, "globalPosition"):
            return event.globalPosition().toPoint()
        return event.globalPos()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        hovered = bool(self.underMouse())
        painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff" if not hovered else "#f8fafc")))
        painter.drawRoundedRect(rect, 5, 5)

        icon_rect = QtCore.QRectF(7, 7, 14, 10)
        pen = QtGui.QPen(QtGui.QColor("#334155"), 1.6)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        path = QtGui.QPainterPath()
        path.moveTo(icon_rect.left(), icon_rect.center().y())
        path.cubicTo(
            icon_rect.left() + 3,
            icon_rect.top(),
            icon_rect.right() - 3,
            icon_rect.top(),
            icon_rect.right(),
            icon_rect.center().y(),
        )
        path.cubicTo(
            icon_rect.right() - 3,
            icon_rect.bottom(),
            icon_rect.left() + 3,
            icon_rect.bottom(),
            icon_rect.left(),
            icon_rect.center().y(),
        )
        painter.drawPath(path)
        if self.isChecked():
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#334155")))
            painter.drawEllipse(QtCore.QPointF(icon_rect.center()), 2.3, 2.3)
        else:
            painter.drawLine(QtCore.QLineF(7, 18, 21, 6))


class _LegendToggleEventFilter(QtCore.QObject):
    def eventFilter(self, obj, event) -> bool:
        plot = self.parent()
        if event.type() in {
            QtCore.QEvent.Resize,
            QtCore.QEvent.Show,
            QtCore.QEvent.MouseMove,
            QtCore.QEvent.MouseButtonRelease,
        }:
            _position_legend_toggle_button(plot)
        return False


def _install_legend_toggle(plot: pg.PlotWidget) -> None:
    if getattr(plot, "_legend_toggle_button", None) is not None:
        return
    setattr(plot, "_legend_visible", True)
    original_clear = plot.clear

    def clear_with_legend_toggle(*args, **kwargs):
        result = original_clear(*args, **kwargs)
        _apply_plot_legend_visibility(plot)
        button = getattr(plot, "_legend_toggle_button", None)
        if button is not None:
            button.show()
            button.raise_()
        return result

    plot.clear = clear_with_legend_toggle
    button = _LegendToggleButton(plot)
    event_filter = _LegendToggleEventFilter(plot)
    plot.installEventFilter(event_filter)
    viewport = getattr(plot, "viewport", lambda: None)()
    if viewport is not None:
        viewport.installEventFilter(event_filter)
    setattr(plot, "_legend_toggle_button", button)
    setattr(plot, "_legend_toggle_event_filter", event_filter)
    _position_legend_toggle_button(plot)
    button.show()
    button.raise_()


def _position_legend_toggle_button(plot) -> None:
    button = getattr(plot, "_legend_toggle_button", None)
    if button is None:
        return
    _refresh_legend_layout(plot)
    margin = 4
    position = _legend_toggle_position(plot, button)
    x = max(margin, min(int(position.x()), max(margin, plot.width() - button.width() - margin)))
    y = max(margin, min(int(position.y()), max(margin, plot.height() - button.height() - margin)))
    button.move(x, y)
    button.raise_()


def _legend_toggle_position(plot, button: QtWidgets.QToolButton) -> QtCore.QPoint:
    legend = getattr(plot.plotItem, "legend", None)
    if legend is not None and legend.isVisible():
        try:
            rect = _legend_rect_in_plot(plot)
            if rect is None:
                raise RuntimeError("legend rect unavailable")
            return QtCore.QPoint(int(rect.right() - button.width() - 4), int(rect.top() + 4))
        except Exception:
            pass
    anchor = getattr(plot, "_legend_hidden_toggle_anchor", None)
    if isinstance(anchor, QtCore.QPoint):
        return QtCore.QPoint(anchor)
    return QtCore.QPoint(max(4, plot.width() - button.width() - 8), 8)


def _move_legend_to_toggle_anchor(plot) -> None:
    legend = getattr(plot.plotItem, "legend", None)
    button = getattr(plot, "_legend_toggle_button", None)
    if legend is None or button is None or not legend.isVisible():
        return
    rect = _legend_rect_in_plot(plot)
    if rect is None:
        return
    try:
        legend_pos = legend.pos()
        base_x = rect.left() - float(legend_pos.x())
        base_y = rect.top() - float(legend_pos.y())
        desired_rect_left = button.x() + button.width() + 4 - rect.width()
        desired_rect_top = button.y() - 4
        desired_offset = (
            float(desired_rect_left) - base_x,
            float(desired_rect_top) - base_y,
        )
        _apply_legend_offset(plot, desired_offset)
        setattr(plot, "_legend_user_offset", (float(desired_offset[0]), float(desired_offset[1])))
    except Exception:
        return
    _position_legend_toggle_button(plot)


def _apply_legend_offset(plot, offset: tuple[float, float]) -> None:
    legend = getattr(plot.plotItem, "legend", None)
    if legend is None:
        return
    legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(float(offset[0]), float(offset[1])))


def _legend_rect_in_plot(plot) -> QtCore.QRect | None:
    legend = getattr(plot.plotItem, "legend", None)
    if legend is None:
        return None
    try:
        _refresh_legend_layout(plot)
        scene_rect = legend.sceneBoundingRect()
        top_left = plot.mapFromScene(scene_rect.topLeft())
        bottom_right = plot.mapFromScene(scene_rect.bottomRight())
        return QtCore.QRect(top_left, bottom_right).normalized()
    except Exception:
        return None


def _refresh_legend_layout(plot) -> None:
    legend = getattr(plot.plotItem, "legend", None)
    if legend is None:
        return
    for method_name in ("updateSize", "adjustSize", "updateGeometry"):
        method = getattr(legend, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _legend_contains_scene_pos(plot, scene_pos: QtCore.QPointF) -> bool:
    legend = getattr(plot.plotItem, "legend", None)
    if legend is None or not legend.isVisible():
        return False
    try:
        return bool(legend.sceneBoundingRect().contains(scene_pos))
    except Exception:
        return False


def _legend_sample_at_scene_pos(plot, scene_pos: QtCore.QPointF) -> int | None:
    if not _legend_contains_scene_pos(plot, scene_pos):
        return None
    for entry in getattr(plot, "_sample_legend_graphics_entries", []):
        sample_index = entry.get("sample_index")
        for key in ("sample_item", "label_item"):
            item = entry.get(key)
            if item is None:
                continue
            try:
                if item.sceneBoundingRect().adjusted(-3, -3, 3, 3).contains(scene_pos):
                    return int(sample_index)
            except Exception:
                continue
    return None


def _set_legend_sample_hover(plot, sample_index: int | None) -> None:
    normalized_index = int(sample_index) if sample_index is not None else None
    current_index = getattr(plot, "_sample_legend_hover_index", None)
    if current_index == normalized_index:
        return
    setattr(plot, "_sample_legend_hover_index", normalized_index)
    changed = False
    for entry in getattr(plot, "_sample_legend_graphics_entries", []):
        label_item = entry.get("label_item")
        if label_item is None:
            continue
        try:
            base_font = entry.get("base_font")
            if not isinstance(base_font, QtGui.QFont):
                item = getattr(label_item, "item", None)
                base_font = item.font() if item is not None and hasattr(item, "font") else label_item.font()
                entry["base_font"] = QtGui.QFont(base_font)
            font = QtGui.QFont(base_font)
            font.setBold(normalized_index is not None and int(entry.get("sample_index", -1)) == normalized_index)
            if hasattr(label_item, "setFont"):
                label_item.setFont(font)
                changed = True
            item = getattr(label_item, "item", None)
            if item is not None and hasattr(item, "setFont"):
                item.setFont(font)
                changed = True
        except Exception:
            pass
    if changed:
        _refresh_legend_layout(plot)


class ClickProjectionCursor:
    """BET-style click-to-show projected coordinates."""

    def __init__(self, plot: pg.PlotWidget) -> None:
        self.plot = plot
        self.plot_item = plot.getPlotItem()
        self.view_box = self.plot_item.getViewBox()
        self.point: tuple[float, float] | None = None
        pen = pg.mkPen("#2563eb", width=1, style=Qt.DashLine)
        self.vertical_line = pg.PlotCurveItem(pen=pen)
        self.horizontal_line = pg.PlotCurveItem(pen=pen)
        self.x_label = pg.TextItem(
            text="",
            color="#111827",
            anchor=(0.5, 1.0),
            fill=pg.mkBrush(255, 255, 255, 235),
            border=pg.mkPen("#2563eb"),
        )
        self.y_label = pg.TextItem(
            text="",
            color="#111827",
            anchor=(0.0, 0.5),
            fill=pg.mkBrush(255, 255, 255, 235),
            border=pg.mkPen("#2563eb"),
        )
        for item in (self.vertical_line, self.horizontal_line, self.x_label, self.y_label):
            item.setZValue(10_000)
            self.view_box.addItem(item, ignoreBounds=True)
            item.hide()
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.view_box.sigRangeChanged.connect(lambda *_args: self.update())

    def _is_disabled(self) -> bool:
        guard = getattr(self.plot, "_click_projection_cursor_disabled", False)
        try:
            return bool(guard()) if callable(guard) else bool(guard)
        except Exception:
            return False

    def reattach(self) -> None:
        added_items = getattr(self.view_box, "addedItems", [])
        for item in (self.vertical_line, self.horizontal_line, self.x_label, self.y_label):
            if item not in added_items:
                self.view_box.addItem(item, ignoreBounds=True)
        self.update()

    def _on_mouse_clicked(self, event) -> None:
        if self._is_disabled():
            self.clear()
            return
        if event.button() != Qt.LeftButton:
            return
        if _legend_contains_scene_pos(self.plot, event.scenePos()):
            if hasattr(event, "accept"):
                event.accept()
            return
        double_click = getattr(event, "double", False)
        is_double_click = double_click() if callable(double_click) else bool(double_click)
        if is_double_click:
            self.clear()
            return
        self.set_scene_position(event.scenePos())

    def set_scene_position(self, scene_pos: QtCore.QPointF) -> bool:
        if not self.view_box.sceneBoundingRect().contains(scene_pos):
            return False
        view_pos = self.view_box.mapSceneToView(scene_pos)
        x = float(view_pos.x())
        y = float(view_pos.y())
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        self.point = (x, y)
        self.update()
        return True

    def clear(self) -> None:
        self.point = None
        self.hide()

    def update(self) -> None:
        if self._is_disabled():
            self.hide()
            return
        if self.point is None:
            self.hide()
            return
        x, y = self.point
        view_range = self.view_box.viewRange()
        if not view_range or len(view_range) != 2:
            self.hide()
            return
        (x_min, x_max), (y_min, y_max) = view_range
        if not all(np.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
            self.hide()
            return
        if (
            x < min(x_min, x_max)
            or x > max(x_min, x_max)
            or y < min(y_min, y_max)
            or y > max(y_min, y_max)
        ):
            self.hide()
            return

        left, right = min(x_min, x_max), max(x_min, x_max)
        bottom, top = min(y_min, y_max), max(y_min, y_max)
        self.vertical_line.setData([x, x], [bottom, top])
        self.horizontal_line.setData([left, right], [y, y])
        self.x_label.setText(self._label_text("bottom", x))
        self.y_label.setText(self._label_text("left", y))
        self.x_label.setPos(x, bottom)
        self.y_label.setPos(left, y)
        for item in (self.vertical_line, self.horizontal_line, self.x_label, self.y_label):
            item.show()

    def hide(self) -> None:
        for item in (self.vertical_line, self.horizontal_line, self.x_label, self.y_label):
            item.hide()

    def _label_text(self, axis_name: str, coordinate: float) -> str:
        axis = self.plot_item.getAxis(axis_name)
        value = coordinate
        if getattr(axis, "logMode", False):
            try:
                value = 10.0**coordinate
            except OverflowError:
                return ""
        return _plain_number(float(value))


def _enable_click_projection_cursor(plot: pg.PlotWidget) -> None:
    if getattr(plot, "_click_projection_cursor", None) is not None:
        return
    cursor = ClickProjectionCursor(plot)
    original_clear = plot.clear

    def clear_with_cursor(*args, **kwargs):
        result = original_clear(*args, **kwargs)
        cursor.point = None
        cursor.reattach()
        return result

    plot.clear = clear_with_cursor
    plot._click_projection_cursor = cursor


class SampleCurveInteractionController(QtCore.QObject):
    """BET-style hover/legend/click interactions for multi-sample curves."""

    HOVER_DELAY_MS = 0
    HIT_DISTANCE_PX = 12.0

    def __init__(self, plot: pg.PlotWidget) -> None:
        super().__init__(plot)
        self.plot = plot
        self.plot_item = plot.getPlotItem()
        self.view_box = self.plot_item.getViewBox()
        self.entries: list[dict[str, object]] = []
        self.hovered_entry: dict[str, object] | None = None
        self.hovered_sample_index: int | None = None
        self.pending_scene_pos: QtCore.QPointF | None = None
        self.hover_timer = QtCore.QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.setInterval(self.HOVER_DELAY_MS)
        self.hover_timer.timeout.connect(self._resolve_hover)
        self.tooltip = self._create_tooltip()
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.plot.installEventFilter(self)
        viewport = getattr(self.plot, "viewport", lambda: None)()
        if viewport is not None:
            viewport.installEventFilter(self)

    def _create_tooltip(self):
        tooltip = pg.TextItem(
            text="",
            color="#111827",
            anchor=(0.0, 1.0),
            fill=pg.mkBrush(255, 255, 255, 238),
            border=pg.mkPen("#2563eb"),
        )
        tooltip.setZValue(20_000)
        tooltip.hide()
        return tooltip

    def _ensure_tooltip(self):
        try:
            self.tooltip.isVisible()
        except RuntimeError:
            self.tooltip = self._create_tooltip()
        return self.tooltip

    def _hide_tooltip(self) -> None:
        try:
            self.tooltip.hide()
        except RuntimeError:
            pass

    def _tooltip_is_visible(self) -> bool:
        try:
            return bool(self.tooltip.isVisible())
        except RuntimeError:
            return False

    def eventFilter(self, obj, event) -> bool:
        if event.type() in {QtCore.QEvent.Leave, QtCore.QEvent.Hide}:
            self.clear_hover()
        return False

    def reset(self) -> None:
        self.hover_timer.stop()
        self.entries = []
        self.pending_scene_pos = None
        self.hovered_entry = None
        self.hovered_sample_index = None
        self._hide_tooltip()
        _set_legend_sample_hover(self.plot, None)

    def register(
        self,
        item,
        *,
        sample_index: int,
        label: str,
        x_values,
        y_values,
    ) -> None:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        if x.size == 0 or y.size == 0:
            return
        count = min(int(x.size), int(y.size))
        x = x[:count]
        y = y[:count]
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return
        x = x[mask]
        y = y[mask]
        entry = {
            "item": item,
            "sample_index": int(sample_index),
            "label": str(label),
            "x": x,
            "y": y,
            "base_pen": _copy_pen_option(item.opts.get("pen")),
            "base_symbol_pen": _copy_pen_option(item.opts.get("symbolPen")),
            "base_shadow_pen": _copy_pen_option(item.opts.get("shadowPen")),
            "base_symbol_size": item.opts.get("symbolSize"),
            "base_z": float(item.zValue()),
        }
        self.entries.append(entry)
        try:
            item.setCurveClickable(True, width=max(10, int(self.HIT_DISTANCE_PX * 1.6)))
            item.sigClicked.connect(
                lambda _item, event, index=int(sample_index), controller=self: controller._on_curve_clicked(index, event)
            )
        except Exception:
            pass

    def _on_mouse_moved(self, scene_pos) -> None:
        legend_sample_index = _legend_sample_at_scene_pos(self.plot, scene_pos)
        if legend_sample_index is not None:
            self.pending_scene_pos = None
            self.hover_timer.stop()
            self._hide_tooltip()
            self.set_hover_sample(int(legend_sample_index), propagate=True)
            return
        if _legend_contains_scene_pos(self.plot, scene_pos):
            self.clear_hover()
            return
        if not self.view_box.sceneBoundingRect().contains(scene_pos):
            self.clear_hover()
            return
        self.pending_scene_pos = QtCore.QPointF(scene_pos)
        if self.HOVER_DELAY_MS <= 0:
            self._resolve_hover()
        else:
            self.hover_timer.start()

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        sample_index = _legend_sample_at_scene_pos(self.plot, event.scenePos())
        if sample_index is None:
            return
        double_click = getattr(event, "double", False)
        is_double_click = double_click() if callable(double_click) else bool(double_click)
        if is_double_click:
            return
        if hasattr(event, "accept"):
            event.accept()
        cursor = getattr(self.plot, "_click_projection_cursor", None)
        if cursor is not None and hasattr(cursor, "clear"):
            cursor.clear()
        self.set_hover_sample(int(sample_index), propagate=True)
        self._select_sample_later(int(sample_index))

    def _resolve_hover(self) -> None:
        scene_pos = self.pending_scene_pos
        if scene_pos is None or not self.entries:
            self.clear_hover()
            return
        if not self.view_box.sceneBoundingRect().contains(scene_pos):
            self.clear_hover()
            return
        transform_context = self._scene_transform_context()
        if transform_context is None:
            self.clear_hover()
            return
        best_entry = None
        best_distance = math.inf
        best_view_point = None
        for entry in self.entries:
            distance, view_point = self._distance_to_entry(scene_pos, entry, transform_context)
            if distance < best_distance:
                best_distance = distance
                best_entry = entry
                best_view_point = view_point
        if best_entry is None or best_distance > self.HIT_DISTANCE_PX:
            self.clear_hover()
            return
        self._set_hover(best_entry, best_view_point)

    def _scene_transform_context(self):
        try:
            (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
            x_min = float(x_min)
            x_max = float(x_max)
            y_min = float(y_min)
            y_max = float(y_max)
            if not all(np.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
                return None
            if abs(x_max - x_min) <= 1e-15 or abs(y_max - y_min) <= 1e-15:
                return None
            origin = self.view_box.mapViewToScene(QtCore.QPointF(x_min, y_min))
            x_ref = self.view_box.mapViewToScene(QtCore.QPointF(x_max, y_min))
            y_ref = self.view_box.mapViewToScene(QtCore.QPointF(x_min, y_max))
            x_scale = 1.0 / (x_max - x_min)
            y_scale = 1.0 / (y_max - y_min)
            return (
                x_min,
                y_min,
                float(origin.x()),
                float(origin.y()),
                (float(x_ref.x()) - float(origin.x())) * x_scale,
                (float(x_ref.y()) - float(origin.y())) * x_scale,
                (float(y_ref.x()) - float(origin.x())) * y_scale,
                (float(y_ref.y()) - float(origin.y())) * y_scale,
            )
        except Exception:
            return None

    @staticmethod
    def _view_to_scene_arrays(
        x_view: np.ndarray,
        y_view: np.ndarray,
        transform_context,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_min, y_min, origin_x, origin_y, x_axis_x, x_axis_y, y_axis_x, y_axis_y = transform_context
        dx = x_view - x_min
        dy = y_view - y_min
        scene_x = origin_x + dx * x_axis_x + dy * y_axis_x
        scene_y = origin_y + dx * x_axis_y + dy * y_axis_y
        return scene_x, scene_y

    def _distance_to_entry(
        self,
        scene_pos: QtCore.QPointF,
        entry: dict[str, object],
        transform_context,
    ) -> tuple[float, QtCore.QPointF | None]:
        x = np.asarray(entry["x"], dtype=float)
        y = np.asarray(entry["y"], dtype=float)
        x_view, y_view = self._data_to_view_coordinates(x, y)
        mask = np.isfinite(x_view) & np.isfinite(y_view)
        if not np.any(mask):
            return (math.inf, None)
        x_view = x_view[mask]
        y_view = y_view[mask]
        scene_x, scene_y = self._view_to_scene_arrays(x_view, y_view, transform_context)
        mask = np.isfinite(scene_x) & np.isfinite(scene_y)
        if not np.any(mask):
            return (math.inf, None)
        x_view = x_view[mask]
        y_view = y_view[mask]
        scene_x = scene_x[mask]
        scene_y = scene_y[mask]
        px = float(scene_pos.x())
        py = float(scene_pos.y())
        if scene_x.size == 1:
            return (
                math.hypot(float(scene_x[0]) - px, float(scene_y[0]) - py),
                QtCore.QPointF(float(x_view[0]), float(y_view[0])),
            )

        x1 = scene_x[:-1]
        y1 = scene_y[:-1]
        x2 = scene_x[1:]
        y2 = scene_y[1:]
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((px - x1) * dx + (py - y1) * dy) / denom
        t = np.where(denom > 1e-12, np.clip(t, 0.0, 1.0), 0.0)
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        distances_sq = (nearest_x - px) ** 2 + (nearest_y - py) ** 2
        if distances_sq.size == 0 or not np.any(np.isfinite(distances_sq)):
            return (math.inf, None)
        idx = int(np.nanargmin(distances_sq))
        distance = math.sqrt(float(distances_sq[idx]))
        segment_t = float(t[idx])
        view_x = float(x_view[idx]) + segment_t * (float(x_view[idx + 1]) - float(x_view[idx]))
        view_y = float(y_view[idx]) + segment_t * (float(y_view[idx + 1]) - float(y_view[idx]))
        return (distance, QtCore.QPointF(view_x, view_y))

    def _data_to_view_coordinates(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_view = np.asarray(x, dtype=float).copy()
        y_view = np.asarray(y, dtype=float).copy()
        if getattr(self.plot_item.getAxis("bottom"), "logMode", False):
            with np.errstate(divide="ignore", invalid="ignore"):
                x_view = np.where(x_view > 0.0, np.log10(x_view), np.nan)
        if getattr(self.plot_item.getAxis("left"), "logMode", False):
            with np.errstate(divide="ignore", invalid="ignore"):
                y_view = np.where(y_view > 0.0, np.log10(y_view), np.nan)
        return x_view, y_view

    def _set_hover(self, entry: dict[str, object], view_point: QtCore.QPointF | None) -> None:
        sample_index = int(entry.get("sample_index", -1))
        if self.hovered_entry is entry and self.hovered_sample_index == sample_index:
            self._move_tooltip(entry, view_point)
            return
        self._restore_all()
        self.hovered_entry = entry
        self.hovered_sample_index = sample_index
        for other in self.entries:
            item = other.get("item")
            if item is None:
                continue
            try:
                item.setOpacity(1.0 if int(other.get("sample_index", -1)) == sample_index else 0.22)
            except Exception:
                pass
        for highlighted in self.entries:
            if int(highlighted.get("sample_index", -1)) == sample_index:
                self._highlight_entry_item(highlighted)
        self._move_tooltip(entry, view_point)
        _set_legend_sample_hover(self.plot, sample_index)
        self._propagate_hover(sample_index)
        self._notify_hover(sample_index)

    def set_linked_hover(self, sample_index: int | None) -> None:
        if sample_index is None:
            self.clear_hover(propagate=False)
            return
        self.set_hover_sample(int(sample_index), propagate=False)

    def set_hover_sample(self, sample_index: int | None, *, propagate: bool = True) -> None:
        if sample_index is None:
            self.clear_hover(propagate=propagate)
            return
        sample_index = int(sample_index)
        if self.hovered_sample_index == sample_index:
            return
        for entry in self.entries:
            if int(entry.get("sample_index", -1)) == sample_index:
                self._set_linked_entry_hover(entry)
                if propagate:
                    self._propagate_hover(sample_index)
                    self._notify_hover(sample_index)
                return
        if any(
            int(entry.get("sample_index", -1)) == sample_index
            for entry in getattr(self.plot, "_sample_legend_graphics_entries", [])
        ):
            self._restore_all()
            self.hovered_entry = None
            self.hovered_sample_index = sample_index
            self._hide_tooltip()
            _set_legend_sample_hover(self.plot, sample_index)
            if propagate:
                self._propagate_hover(sample_index)
                self._notify_hover(sample_index)
            return
        self.clear_hover(propagate=False)
        if propagate:
            self._propagate_hover(None)
            self._notify_hover(None)

    def _set_linked_entry_hover(self, entry: dict[str, object]) -> None:
        sample_index = int(entry.get("sample_index", -1))
        if self.hovered_sample_index == sample_index:
            return
        self._restore_all()
        self.hovered_entry = entry
        self.hovered_sample_index = sample_index
        for other in self.entries:
            item = other.get("item")
            if item is None:
                continue
            try:
                item.setOpacity(1.0 if int(other.get("sample_index", -1)) == sample_index else 0.22)
            except Exception:
                pass
        for highlighted in self.entries:
            if int(highlighted.get("sample_index", -1)) == sample_index:
                self._highlight_entry_item(highlighted)
        self._hide_tooltip()
        _set_legend_sample_hover(self.plot, sample_index)

    def _highlight_entry_item(self, entry: dict[str, object]) -> None:
        item = entry.get("item")
        if item is None:
            return
        base_pen = entry.get("base_pen")
        if isinstance(base_pen, QtGui.QPen):
            highlight_pen = QtGui.QPen(base_pen)
            highlight_pen.setWidthF(max(float(base_pen.widthF()) + 2.5, 5.0))
            item.setPen(highlight_pen)
            glow_pen = QtGui.QPen(base_pen)
            glow_pen.setColor(QtGui.QColor("#fbbf24"))
            glow_pen.setWidthF(max(float(base_pen.widthF()) + 8.0, 10.0))
            try:
                item.setShadowPen(glow_pen)
            except Exception:
                pass
        base_symbol_size = entry.get("base_symbol_size")
        if base_symbol_size is not None:
            try:
                item.setSymbolSize(float(base_symbol_size) + 3.0)
            except Exception:
                pass
        try:
            item.setZValue(15_000)
        except Exception:
            pass

    def _propagate_hover(self, sample_index: int | None) -> None:
        for peer_plot in getattr(self.plot, "_linked_sample_curve_hover_plots", []):
            if peer_plot is self.plot:
                continue
            if sample_index is not None:
                try:
                    if not peer_plot.isVisible():
                        continue
                except Exception:
                    pass
            controller = getattr(peer_plot, "_sample_curve_interaction_controller", None)
            if controller is None:
                continue
            try:
                controller.set_linked_hover(sample_index)
            except Exception:
                pass

    def _notify_hover(self, sample_index: int | None) -> None:
        callback = getattr(self.plot, "_sample_curve_hovered_callback", None)
        if callable(callback):
            try:
                callback(sample_index)
            except Exception:
                pass

    def clear_hover(self, *, propagate: bool = True) -> None:
        self.hover_timer.stop()
        self.pending_scene_pos = None
        if (
            self.hovered_entry is None
            and self.hovered_sample_index is None
            and getattr(self.plot, "_sample_legend_hover_index", None) is None
            and not self._tooltip_is_visible()
        ):
            return
        self._restore_all()
        self.hovered_entry = None
        self.hovered_sample_index = None
        self._hide_tooltip()
        _set_legend_sample_hover(self.plot, None)
        if propagate:
            self._propagate_hover(None)
            self._notify_hover(None)

    def _restore_all(self) -> None:
        for entry in self.entries:
            item = entry.get("item")
            if item is None:
                continue
            try:
                item.setOpacity(1.0)
                if isinstance(entry.get("base_pen"), QtGui.QPen):
                    item.setPen(QtGui.QPen(entry["base_pen"]))
                if isinstance(entry.get("base_symbol_pen"), QtGui.QPen):
                    item.setSymbolPen(QtGui.QPen(entry["base_symbol_pen"]))
                if isinstance(entry.get("base_shadow_pen"), QtGui.QPen):
                    item.setShadowPen(QtGui.QPen(entry["base_shadow_pen"]))
                else:
                    try:
                        item.setShadowPen(None)
                    except Exception:
                        pass
                if entry.get("base_symbol_size") is not None:
                    item.setSymbolSize(entry["base_symbol_size"])
                item.setZValue(float(entry.get("base_z", 0.0)))
            except Exception:
                pass

    def _move_tooltip(self, entry: dict[str, object], view_point: QtCore.QPointF | None) -> None:
        if view_point is None:
            return
        tooltip = self._ensure_tooltip()
        added_items = getattr(self.view_box, "addedItems", [])
        if tooltip not in added_items:
            self.view_box.addItem(tooltip, ignoreBounds=True)
        tooltip.setText(str(entry.get("label", "")))
        tooltip.setPos(float(view_point.x()), float(view_point.y()))
        tooltip.show()

    def _on_curve_clicked(self, sample_index: int, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        double_click = getattr(event, "double", False)
        is_double_click = double_click() if callable(double_click) else bool(double_click)
        cursor = getattr(self.plot, "_click_projection_cursor", None)
        if is_double_click:
            if cursor is not None and hasattr(cursor, "clear"):
                cursor.clear()
            if hasattr(event, "accept"):
                event.accept()
            self._select_sample_later(int(sample_index))
            return
        if cursor is not None and hasattr(cursor, "set_scene_position"):
            cursor.set_scene_position(event.scenePos())

    def _select_sample_later(self, sample_index: int) -> None:
        callback = getattr(self.plot, "_sample_curve_selected_callback", None)
        if not callable(callback):
            return
        QtCore.QTimer.singleShot(
            0,
            lambda index=int(sample_index), selected_callback=callback: selected_callback(index),
        )


def _sample_curve_controller(plot: pg.PlotWidget) -> SampleCurveInteractionController:
    controller = getattr(plot, "_sample_curve_interaction_controller", None)
    if controller is None:
        controller = SampleCurveInteractionController(plot)
        setattr(plot, "_sample_curve_interaction_controller", controller)
        if not getattr(plot, "_sample_curve_clear_patched", False):
            original_clear = plot.clear

            def clear_with_sample_curve_reset(*args, **kwargs):
                result = original_clear(*args, **kwargs)
                controller.reset()
                return result

            plot.clear = clear_with_sample_curve_reset
            setattr(plot, "_sample_curve_clear_patched", True)
    return controller


def link_sample_curve_hover_plots(*plots: pg.PlotWidget) -> None:
    unique_plots = []
    for plot in plots:
        if plot is not None and plot not in unique_plots:
            unique_plots.append(plot)
    for plot in unique_plots:
        setattr(plot, "_linked_sample_curve_hover_plots", [peer for peer in unique_plots if peer is not plot])


def set_sample_curve_hover_plots(sample_index: int | None, *plots: pg.PlotWidget) -> None:
    for plot in plots:
        if plot is None:
            continue
        if sample_index is not None:
            try:
                if not plot.isVisible():
                    continue
            except Exception:
                pass
        controller = getattr(plot, "_sample_curve_interaction_controller", None)
        if controller is None:
            _set_legend_sample_hover(plot, sample_index)
            continue
        try:
            controller.set_linked_hover(sample_index)
        except Exception:
            pass


def _copy_pen_option(value) -> QtGui.QPen | None:
    if value is None:
        return None
    if isinstance(value, QtGui.QPen):
        return QtGui.QPen(value)
    try:
        return QtGui.QPen(pg.mkPen(value))
    except Exception:
        return None


def _reset_sample_curve_interactions(plot: pg.PlotWidget) -> None:
    controller = getattr(plot, "_sample_curve_interaction_controller", None)
    if controller is not None:
        controller.reset()


def _register_sample_curve(
    plot: pg.PlotWidget,
    item,
    *,
    sample_index: int,
    label: str,
    x_values,
    y_values,
) -> None:
    if item is None:
        return
    _sample_curve_controller(plot).register(
        item,
        sample_index=sample_index,
        label=label,
        x_values=x_values,
        y_values=y_values,
    )


def _set_sample_legend_entries(plot: pg.PlotWidget, entries) -> None:
    legend = getattr(plot.plotItem, "legend", None)
    if legend is None:
        return
    legend.clear()
    sorted_entries = sorted(entries, key=lambda entry: entry[0])
    for _index, item, name in sorted_entries:
        legend.addItem(item, name)
    graphics_entries = []
    for source_entry, legend_entry in zip(sorted_entries, list(getattr(legend, "items", []))):
        try:
            index, item, name = source_entry
            sample_item, label_item = legend_entry
            label_graphics_item = getattr(label_item, "item", None)
            base_font = (
                label_graphics_item.font()
                if label_graphics_item is not None and hasattr(label_graphics_item, "font")
                else label_item.font()
            )
            graphics_entries.append(
                {
                    "sample_index": int(index),
                    "curve_item": item,
                    "name": str(name),
                    "sample_item": sample_item,
                    "label_item": label_item,
                    "base_font": QtGui.QFont(base_font),
                }
            )
        except Exception:
            continue
    setattr(plot, "_sample_legend_graphics_entries", graphics_entries)
    setattr(plot, "_sample_legend_hover_index", None)
    _sync_plot_legend_visibility(plot)


def _set_plot_legend_visible(plot: pg.PlotWidget, visible: bool) -> None:
    setattr(plot, "_legend_visible", bool(visible))
    _apply_plot_legend_visibility(plot)


def _apply_plot_legend_visibility(plot: pg.PlotWidget) -> None:
    visible = bool(getattr(plot, "_legend_visible", True))
    legend = getattr(plot.plotItem, "legend", None)
    button = getattr(plot, "_legend_toggle_button", None)
    if legend is not None and button is not None and not visible:
        setattr(plot, "_legend_hidden_toggle_anchor", QtCore.QPoint(button.pos()))
    if legend is not None:
        legend.setVisible(bool(visible))
        if visible:
            hidden_anchor = getattr(plot, "_legend_hidden_toggle_anchor", None)
            if isinstance(hidden_anchor, QtCore.QPoint):
                _move_legend_to_toggle_anchor(plot)
                try:
                    delattr(plot, "_legend_hidden_toggle_anchor")
                except AttributeError:
                    pass
            else:
                user_offset = getattr(plot, "_legend_user_offset", None)
                if user_offset is not None:
                    _apply_legend_offset(plot, user_offset)
    if button is not None:
        button.blockSignals(True)
        button.setChecked(bool(visible))
        button.setToolTip("隐藏图例" if visible else "显示图例")
        button.blockSignals(False)
        _position_legend_toggle_button(plot)
        button.update()


def _sync_plot_legend_visibility(plot: pg.PlotWidget) -> None:
    _apply_plot_legend_visibility(plot)
    QtCore.QTimer.singleShot(0, lambda plot=plot: _position_legend_toggle_button(plot))


class BaselineAnchorItem(pg.TargetItem):
    """Draggable manual-baseline anchor with right-click deletion."""

    def __init__(self, owner, anchor_id, kind: str, pos):
        super().__init__(
            pos=pos,
            size=11,
            symbol="o",
            pen=pg.mkPen("#2563eb", width=1.8),
            hoverPen=pg.mkPen("#60a5fa", width=3.0),
            brush=pg.mkBrush(37, 99, 235, 165),
            hoverBrush=pg.mkBrush(96, 165, 250, 230),
            movable=True,
        )
        self._xrd_owner = owner
        self._xrd_anchor_id = anchor_id
        self._xrd_kind = kind
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setCursor(Qt.ArrowCursor)
        self.setZValue(2500)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            ev.accept()
            self._xrd_owner._delete_manual_baseline_anchor(
                self._xrd_anchor_id,
                self._xrd_kind,
            )
            return
        super().mouseClickEvent(ev)


class DraggableMarkerTextItem(pg.TextItem):
    """Movable fit annotation that can keep a dashed leader to its peak."""

    def __init__(
        self,
        owner,
        plot: pg.PlotWidget,
        text: str,
        color: str,
        anchor_pos: tuple[float, float],
        *,
        marker_key: str,
        fill=None,
    ):
        super().__init__(text=text, color=color, anchor=(0.5, 1.0), fill=fill)
        self._xrd_owner = owner
        self._xrd_plot = plot
        self._xrd_color = color
        self._xrd_marker_key = str(marker_key)
        self._xrd_anchor_pos = (float(anchor_pos[0]), float(anchor_pos[1]))
        self._xrd_drag_offset = QtCore.QPointF(0.0, 0.0)
        self._xrd_connector_active = False
        self._xrd_connector = pg.PlotDataItem(
            [],
            [],
            pen=owner._curve_pen(color, width=1.1, style=Qt.DashLine, alpha=0.85),
        )
        self._xrd_connector.setZValue(39)
        self._xrd_connector.setVisible(False)
        plot.addItem(self._xrd_connector)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.ArrowCursor)

    def _view_pos_from_event(self, ev) -> QtCore.QPointF:
        view_box = self._xrd_plot.getPlotItem().getViewBox()
        return view_box.mapSceneToView(ev.scenePos())

    def _connector_should_show(self) -> bool:
        return bool(
            self._xrd_connector_active
            and getattr(self._xrd_owner, "marker_text_visible", True)
            and self.isVisible()
        )

    def _update_connector(self) -> None:
        pos = self.pos()
        self._xrd_connector.setData(
            [self._xrd_anchor_pos[0], float(pos.x())],
            [self._xrd_anchor_pos[1], float(pos.y())],
        )
        self._xrd_connector.setVisible(self._connector_should_show())

    def set_marker_visible(self, visible: bool) -> None:
        self.setVisible(bool(visible))
        self._update_connector()

    def hoverEvent(self, ev):
        if not ev.isExit():
            ev.acceptDrags(Qt.LeftButton)

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        ev.accept()
        view_pos = self._view_pos_from_event(ev)
        if ev.isStart():
            current = self.pos()
            self._xrd_drag_offset = QtCore.QPointF(
                float(current.x()) - float(view_pos.x()),
                float(current.y()) - float(view_pos.y()),
            )
            self._xrd_connector_active = True
        new_pos = QtCore.QPointF(
            float(view_pos.x()) + self._xrd_drag_offset.x(),
            float(view_pos.y()) + self._xrd_drag_offset.y(),
        )
        self.setPos(new_pos)
        self._update_connector()
        if ev.isFinish():
            self._update_connector()
            if hasattr(self._xrd_owner, "_save_current_marker_label_state"):
                self._xrd_owner._save_current_marker_label_state()


class PlotPanelMixin:
    """Right-side plot panel methods."""

    def setup_plots(self):
        self.progress_label_widget = QLabel("")
        self.progress_var = TextValue(self.progress_label_widget)
        self.progress_label = self.progress_label_widget
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(260)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #c7d2ce;
                border-radius: 7px;
                background: #edf2f0;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #34d399,
                    stop: 0.55 #22c55e,
                    stop: 1 #16a34a
                );
            }
            """
        )
        self.progress_label.hide()
        self.progress_bar.hide()

        outer_right_layout = self.right_layout
        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.setDocumentMode(False)
        self.right_tabs.tabBar().setDrawBase(False)
        self.right_tabs.setStyleSheet(
            """
            QTabWidget { background: transparent; border: 0; }
            QTabWidget::pane { border: 0; background: transparent; margin: 0; padding: 0; }
            QTabWidget::tab-bar { left: 0; }
            QTabBar { border: 0; qproperty-drawBase: 0; }
            QTabBar::tab {
                background: #eef1f3;
                border: 1px solid #d1d5db;
                border-bottom: 0;
                padding: 5px 14px;
                font: 8pt 'Microsoft YaHei';
            }
            QTabBar::tab:selected { background: #ffffff; color: #111827; }
            """
        )
        self.progress_corner_widget = QWidget(self.right_tabs)
        progress_corner_layout = QHBoxLayout(self.progress_corner_widget)
        progress_corner_layout.setContentsMargins(0, 0, 4, 0)
        progress_corner_layout.setSpacing(6)
        self.progress_label.setStyleSheet("font: 8pt 'Microsoft YaHei UI'; color: #166534;")
        progress_corner_layout.addWidget(self.progress_label)
        progress_corner_layout.addWidget(self.progress_bar)
        self.right_tabs.setCornerWidget(self.progress_corner_widget, Qt.TopRightCorner)

        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(6)
        self.right_tabs.addTab(self.analysis_tab, "单样品分析")
        outer_right_layout.addWidget(self.right_tabs, 1)
        self.right_layout = analysis_layout

        analysis_controls = self._build_analysis_controls(self.analysis_tab)
        self.analysis_options_pane = CollapsibleAnalysisPane(analysis_controls)

        preview_panel = QWidget()
        preview_panel.setMinimumHeight(48)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        self.preview_title_label = QLabel("完整数据预览")
        self.preview_title_label.setStyleSheet("font: 8pt 'Microsoft YaHei UI'; color: #111827;")
        self.preview_title_label.hide()

        self.preview_plot = self._make_plot("", "Intensity", "2θ (°)")
        self.preview_plot.setMinimumHeight(48)
        self.preview_plot.setTitle("完整数据预览", color="#111827", size="10pt")
        self.preview_ax = self.preview_plot
        self.preview_canvas = self.preview_plot
        self.preview_plot.scene().sigMouseClicked.connect(self._on_preview_plot_mouse_clicked)
        preview_layout.addWidget(self.preview_plot, 1)

        self.top_plot_splitter = _configure_splitter(QtWidgets.QSplitter(Qt.Horizontal), handle_width=7)
        self.top_plot_splitter.setMinimumHeight(64)
        self.top_plot_splitter.addWidget(self.analysis_options_pane)
        self.top_plot_splitter.addWidget(preview_panel)
        self.top_plot_splitter.setStretchFactor(0, 0)
        self.top_plot_splitter.setStretchFactor(1, 1)
        self.top_plot_splitter.setSizes([self.analysis_options_pane.expanded_width(), 900])
        self.top_plot_splitter.splitterMoved.connect(
            lambda _pos, _index: self.analysis_options_pane.remember_current_width()
        )

        self.fit_plot = self._make_plot("拟合范围预览", "Intensity", "2θ (°)")
        self.size_plot = self._make_plot("粒径分布 (计算后显示)", "Volume Density", "Particle size (nm)")
        self.fit_plot._click_projection_cursor_disabled = lambda: bool(
            getattr(self, "manual_baseline_enabled", False)
        )
        self.fit_plot.setMinimumHeight(80)
        self.size_plot.setMinimumHeight(80)
        setattr(self.size_plot, "_legend_default_position", "right")
        self.axes0 = self.fit_plot
        self.axes1 = self.size_plot
        self._install_fit_view_all_handler()
        self.fit_plot.scene().sigMouseClicked.connect(self._on_fit_plot_mouse_clicked)
        self.bottom_plot_splitter = _configure_splitter(QtWidgets.QSplitter(Qt.Horizontal), handle_width=7)
        self.bottom_plot_splitter.setMinimumHeight(110)
        self.bottom_plot_splitter.addWidget(self.fit_plot)
        self.bottom_plot_splitter.addWidget(self.size_plot)
        self.bottom_plot_splitter.setStretchFactor(0, 1)
        self.bottom_plot_splitter.setStretchFactor(1, 1)
        self.bottom_plot_splitter.setSizes([700, 700])

        self.plot_vertical_splitter = _configure_splitter(QtWidgets.QSplitter(Qt.Vertical), handle_width=8)
        self.plot_vertical_splitter.setMinimumHeight(220)
        self.plot_vertical_splitter.addWidget(self.top_plot_splitter)
        self.plot_vertical_splitter.addWidget(self.bottom_plot_splitter)
        self.plot_vertical_splitter.setStretchFactor(0, 1)
        self.plot_vertical_splitter.setStretchFactor(1, 1)
        self.plot_vertical_splitter.setSizes([470, 480])

        self.canvas = self.plot_vertical_splitter
        self.canvas_widget = self.plot_vertical_splitter
        self.toolbar = None
        self.right_layout.addWidget(self.plot_vertical_splitter, 1)
        self._initial_preview_height_applied = False
        QtCore.QTimer.singleShot(0, self._apply_initial_preview_height)
        QtCore.QTimer.singleShot(120, self._apply_initial_preview_height)
        self._setup_comparison_tab()
        self.right_tabs.currentChanged.connect(self._on_right_tab_changed)

        self.preview_range_region = None
        self.preview_range_span = None
        self.line_min = None
        self.line_max = None
        self.peak_mu_lines_preview = []
        self.peak_mu_lines_axes0 = []
        self.peak_mu_rects_preview = []
        self.peak_mu_rects_axes0 = []
        self.axes0_context_artists = []
        self._fit_autorange_sources = []
        self._syncing_range_region = False
        self._syncing_peak_line = False
        self._manual_baseline_curve_item = None
        self._manual_baseline_anchor_items = []
        self._syncing_manual_baseline_anchor = False
        self.marker_text_visible = True
        self.fit_marker_text_items = []
        self._fit_marker_text_legend = None
        self._size_visibility = {}

    def _apply_initial_preview_height(self) -> None:
        if getattr(self, "_initial_preview_height_applied", False):
            return
        splitter = getattr(self, "plot_vertical_splitter", None)
        button = getattr(self, "btn_fast", None)
        if splitter is None or button is None or not button.isVisible():
            return

        sizes = splitter.sizes()
        total = sum(sizes) if sizes else splitter.height()
        if total <= 0:
            return

        button_bottom = button.mapTo(splitter, QtCore.QPoint(0, button.height())).y()
        target_top = max(180, min(int(button_bottom), int(total - 160)))
        if target_top <= 0:
            return

        splitter.setSizes([target_top, max(160, int(total - target_top))])
        self._initial_preview_height_applied = True

    def _setup_comparison_tab(self) -> None:
        self.compare_tab = QWidget()
        compare_layout = QVBoxLayout(self.compare_tab)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        compare_layout.setSpacing(6)
        self.compare_preview_plot = self._make_plot("完整数据预览", "Intensity", "2θ (°)")
        self.compare_size_plot = self._make_plot("粒径分布", "Volume Density", "Particle size (nm)")
        setattr(self.compare_size_plot, "_legend_default_position", "right")
        for plot in (self.compare_preview_plot, self.compare_size_plot):
            setattr(plot, "_sample_curve_selected_callback", self._select_sample_from_curve)
            setattr(plot, "_sample_curve_hovered_callback", self._set_hovered_sample_row)
        link_sample_curve_hover_plots(self.compare_preview_plot, self.compare_size_plot)
        compare_layout.addWidget(self.compare_preview_plot, 1)
        compare_layout.addWidget(self.compare_size_plot, 1)
        self.right_tabs.addTab(self.compare_tab, "对比分析")

    def _on_right_tab_changed(self, index: int) -> None:
        if hasattr(self, "compare_tab") and index == self.right_tabs.indexOf(self.compare_tab):
            self.update_comparison_plots()

    def _make_plot(self, title: str, left_label: str, bottom_label: str) -> XRDPlotWidget:
        bottom_axis = PlainNumberAxis(orientation="bottom")
        left_axis = PlainNumberAxis(orientation="left")
        bottom_axis.setStyle(tickTextWidth=86, autoExpandTextSpace=True)
        left_axis.setStyle(tickTextWidth=96, autoExpandTextSpace=True)
        for axis in (bottom_axis, left_axis):
            axis.enableAutoSIPrefix(False)
        plot = XRDPlotWidget(axisItems={"bottom": bottom_axis, "left": left_axis})
        plot.setTitle(title, color="#111827", size="10pt")
        plot.setLabel("left", left_label)
        plot.setLabel("bottom", bottom_label)
        _install_legend_toggle(plot)
        _enable_click_projection_cursor(plot)
        return plot

    def _sample_display_name(self, sample) -> str:
        try:
            name = Path(str(getattr(sample, "path", ""))).stem
        except Exception:
            name = ""
        if not name:
            metadata = getattr(sample, "metadata", {}) or {}
            name = str(metadata.get("file_name") or getattr(sample, "name", "") or "样品")
        return name

    def _comparison_color(self, index: int) -> str:
        colors = (
            "#2563eb",
            "#16a34a",
            "#f97316",
            "#9333ea",
            "#0891b2",
            "#db2777",
            "#65a30d",
            "#4f46e5",
            "#14b8a6",
            "#7c3aed",
        )
        return colors[int(index) % len(colors)]

    def set_sample_curve_hover_plots(self, sample_index: int | None, *plots: pg.PlotWidget) -> None:
        set_sample_curve_hover_plots(sample_index, *plots)

    def _add_pending_legend_item(self, plot: pg.PlotWidget, label: str, color: str) -> None:
        legend = self._ensure_plot_legend(plot)
        dummy = pg.PlotDataItem(
            [0, 1],
            [0, 0],
            pen=self._curve_pen(color, width=1.8, style=Qt.DashLine, alpha=0.45),
        )
        legend.addItem(dummy, label)
        _sync_plot_legend_visibility(plot)
        return dummy

    def update_comparison_plots(self) -> None:
        if not hasattr(self, "compare_preview_plot") or not hasattr(self, "compare_size_plot"):
            return

        samples = list(getattr(self, "samples", []) or [])
        visible_samples = [
            (index, sample)
            for index, sample in enumerate(samples)
            if bool(getattr(sample, "compare_visible", True))
        ]

        self._clear_plot(self.compare_preview_plot, title="完整数据预览")
        self._clear_plot(self.compare_size_plot, title="粒径分布")
        self._ensure_plot_legend(self.compare_preview_plot)
        self._ensure_plot_legend(self.compare_size_plot)

        preview_sources = []
        preview_legend_entries = []
        size_legend_entries = []
        size_y_max = 0.0
        size_has_curve = False
        for index, sample in visible_samples:
            color = self._comparison_color(index)
            label = self._sample_display_name(sample)
            x_data = np.asarray(getattr(sample, "x_data", []), dtype=float)
            y_data = np.asarray(getattr(sample, "y_data", []), dtype=float)
            if x_data.size and y_data.size and x_data.size == y_data.size:
                item = self._line(self.compare_preview_plot, x_data, y_data, color, width=1.4, alpha=0.86, name=label)
                preview_legend_entries.append((index, item, label))
                _register_sample_curve(
                    self.compare_preview_plot,
                    item,
                    sample_index=index,
                    label=label,
                    x_values=x_data,
                    y_values=y_data,
                )
                preview_sources.append((x_data, y_data))

            results = getattr(sample, "results", {}) or {}
            D_range = np.asarray(results.get("D_range", []), dtype=float)
            peak_infos = list(results.get("all_peak_info", []) or [])
            if D_range.size and peak_infos:
                eps = 1e-12
                area_total = 0.0
                f_segments = []
                for info in peak_infos:
                    f_segment = np.asarray(info.get("f_segment", []), dtype=float)
                    if f_segment.size == D_range.size:
                        f_segments.append(f_segment)
                    area_total += sum(float(det.get("area", 0.0)) for det in info.get("peak_details", []))
                if f_segments:
                    y_pdf = np.sum(f_segments, axis=0) / max(area_total, eps)
                    item = self._line(self.compare_size_plot, D_range, y_pdf, color, width=2.0, alpha=0.9, name=label)
                    size_legend_entries.append((index, item, label))
                    _register_sample_curve(
                        self.compare_size_plot,
                        item,
                        sample_index=index,
                        label=label,
                        x_values=D_range,
                        y_values=y_pdf,
                    )
                    size_y_max = max(size_y_max, float(np.nanmax(y_pdf)) if y_pdf.size else 0.0)
                    size_has_curve = True
                else:
                    pending_label = f"{label}(待计算)"
                    dummy = self._add_pending_legend_item(self.compare_size_plot, pending_label, color)
                    size_legend_entries.append((index, dummy, pending_label))
            else:
                pending_label = f"{label}(待计算)"
                dummy = self._add_pending_legend_item(self.compare_size_plot, pending_label, color)
                size_legend_entries.append((index, dummy, pending_label))

        _set_sample_legend_entries(self.compare_preview_plot, preview_legend_entries)
        _set_sample_legend_entries(self.compare_size_plot, size_legend_entries)

        if preview_sources:
            x_min = min(float(np.nanmin(x)) for x, _ in preview_sources if x.size)
            x_max = max(float(np.nanmax(x)) for x, _ in preview_sources if x.size)
            y_min = min(float(np.nanmin(y)) for _, y in preview_sources if y.size)
            y_max = max(float(np.nanmax(y)) for _, y in preview_sources if y.size)
            self._set_plot_xrange(self.compare_preview_plot, x_min, x_max)
            pad = max((y_max - y_min) * 0.08, abs(y_max) * 0.02, 1.0)
            self.compare_preview_plot.setYRange(y_min - pad, y_max + pad, padding=0)
        else:
            self.compare_preview_plot.setXRange(0, 1, padding=0)
            self.compare_preview_plot.setYRange(0, 1, padding=0)

        if size_has_curve:
            d_ranges = [
                np.asarray(getattr(sample, "results", {}).get("D_range", []), dtype=float)
                for _, sample in visible_samples
                if getattr(sample, "results", {}).get("D_range", None) is not None
            ]
            d_ranges = [arr for arr in d_ranges if arr.size]
            if d_ranges:
                self._set_plot_xrange(
                    self.compare_size_plot,
                    min(float(np.nanmin(arr)) for arr in d_ranges),
                    max(float(np.nanmax(arr)) for arr in d_ranges),
                )
            self.compare_size_plot.setYRange(0.0, size_y_max * 1.18 if size_y_max > 0 else 1.0, padding=0)
        else:
            self._set_plot_xrange(self.compare_size_plot, 0.5, 100.0)
            self.compare_size_plot.setYRange(0, 1, padding=0)

        self.compare_preview_plot.setLabel("left", "Intensity")
        self.compare_preview_plot.setLabel("bottom", "2θ (°)")
        self.compare_size_plot.setLabel("left", "Volume Density")
        self.compare_size_plot.setLabel("bottom", "Particle size (nm)")
        hovered = getattr(self, "_hovered_sample_row", -1)
        self.set_sample_curve_hover_plots(
            hovered if 0 <= int(hovered) < len(samples) else None,
            self.compare_preview_plot,
            self.compare_size_plot,
        )
        self._safe_draw_idle()

    def _clear_plot(self, plot: pg.PlotWidget, *, title: str | None = None) -> None:
        _reset_sample_curve_interactions(plot)
        setattr(plot, "_manual_sample_legend_entries", [])
        setattr(plot, "_sample_legend_graphics_entries", [])
        setattr(plot, "_sample_legend_hover_index", None)
        if hasattr(self, "_clear_plot_coordinate_artifacts"):
            self._clear_plot_coordinate_artifacts(plot)
        plot.clear()
        if plot is getattr(self, "fit_plot", None):
            self._manual_baseline_curve_item = None
            self._manual_baseline_anchor_items = []
            self.fit_marker_text_items = []
            self._fit_marker_text_legend = None
        legend = getattr(plot.getPlotItem(), "legend", None)
        if legend is not None:
            legend.clear()
        if title is not None:
            plot.setTitle(title, color="#111827", size="10pt")

    def _ensure_plot_legend(self, plot: pg.PlotWidget):
        legend = getattr(plot.getPlotItem(), "legend", None)
        if legend is None:
            legend = plot.addLegend(
                offset=(10, 10),
                labelTextColor="#111827",
                brush=pg.mkBrush(255, 255, 255, 220),
                pen=pg.mkPen("#d1d5db"),
            )
        self._apply_plot_legend_default_position(plot)
        _sync_plot_legend_visibility(plot)
        return legend

    @staticmethod
    def _apply_plot_legend_default_position(plot: pg.PlotWidget) -> None:
        legend = getattr(plot.getPlotItem(), "legend", None)
        if legend is None or getattr(plot, "_legend_user_offset", None) is not None:
            return
        if getattr(plot, "_legend_default_position", "left") == "right":
            legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))
        else:
            legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(10, 10))

    def _toggle_peak_visible_from_legend(self, peak_idx: int) -> None:
        if hasattr(self, "_set_peak_visible"):
            self._set_peak_visible(peak_idx, not self._peak_visible(peak_idx))
        else:
            if not hasattr(self, "peak_visible_states"):
                self.peak_visible_states = []
            while peak_idx >= len(self.peak_visible_states):
                self.peak_visible_states.append(True)
            self.peak_visible_states[peak_idx] = not bool(self.peak_visible_states[peak_idx])
        if hasattr(self, "_save_current_peak_states"):
            self._save_current_peak_states()
        if hasattr(self, "_refresh_plots_after_peak_display_change"):
            self._refresh_plots_after_peak_display_change()

    def _add_peak_legend_toggle(self, plot: pg.PlotWidget, peak_idx: int) -> None:
        legend = self._ensure_plot_legend(plot)
        visible = self._peak_visible(peak_idx)
        color = self._peak_color(peak_idx)
        alpha = 1.0 if visible else 0.25
        dummy = pg.PlotDataItem(
            [0, 1],
            [0, 0],
            pen=self._curve_pen(color, width=2.0, alpha=alpha),
        )
        label_text = f"Peak{peak_idx + 1}" if visible else f"Peak{peak_idx + 1} (隐藏)"
        legend.addItem(dummy, label_text)
        try:
            sample, label = legend.items[-1]
        except Exception:
            return
        try:
            label.setText(label_text, color="#111827" if visible else "#9ca3af", size="9pt")
        except Exception:
            pass
        for obj in (sample, label):
            try:
                obj.setOpacity(alpha)
                obj.setCursor(Qt.PointingHandCursor)
                obj.setAcceptedMouseButtons(Qt.LeftButton)
            except Exception:
                pass

            def on_click(event, idx=peak_idx):
                try:
                    if event.button() == Qt.LeftButton:
                        event.accept()
                    else:
                        return
                except Exception:
                    pass
                self._toggle_peak_visible_from_legend(idx)

            try:
                obj.mouseClickEvent = on_click
            except Exception:
                pass

    def _add_peak_legend_toggles(self, plot: pg.PlotWidget, peak_indices) -> None:
        for peak_idx in list(peak_indices or []):
            self._add_peak_legend_toggle(plot, int(peak_idx))

    def _set_fit_marker_text_visible(self, visible: bool) -> None:
        self.marker_text_visible = bool(visible)
        for item in getattr(self, "fit_marker_text_items", []) or []:
            try:
                if hasattr(item, "set_marker_visible"):
                    item.set_marker_visible(self.marker_text_visible)
                else:
                    item.setVisible(self.marker_text_visible)
            except Exception:
                pass
        self._update_marker_text_legend_style()
        self._save_current_marker_label_state()
        self._safe_draw_idle()

    def _toggle_marker_text_visible_from_legend(self) -> None:
        self._set_fit_marker_text_visible(not bool(getattr(self, "marker_text_visible", True)))

    def _update_marker_text_legend_style(self) -> None:
        legend_bits = getattr(self, "_fit_marker_text_legend", None)
        if not legend_bits:
            return
        visible = bool(getattr(self, "marker_text_visible", True))
        label_text = "标记文本" if visible else "标记文本 (隐藏)"
        sample, label = legend_bits
        try:
            label.setText(label_text, color="#111827" if visible else "#9ca3af", size="9pt")
        except Exception:
            pass
        try:
            sample.setOpacity(1.0 if visible else 0.25)
            label.setOpacity(1.0 if visible else 0.55)
        except Exception:
            pass

    def _add_marker_text_legend_toggle(self, plot: pg.PlotWidget) -> None:
        legend = self._ensure_plot_legend(plot)
        visible = bool(getattr(self, "marker_text_visible", True))
        alpha = 1.0 if visible else 0.25
        dummy = pg.PlotDataItem(
            [0, 1],
            [0, 0],
            pen=self._curve_pen("#374151", width=1.5, style=Qt.DashLine, alpha=alpha),
        )
        legend.addItem(dummy, "标记文本" if visible else "标记文本 (隐藏)")
        try:
            sample, label = legend.items[-1]
        except Exception:
            return
        self._fit_marker_text_legend = (sample, label)
        for obj in (sample, label):
            try:
                obj.setCursor(Qt.PointingHandCursor)
                obj.setAcceptedMouseButtons(Qt.LeftButton)
            except Exception:
                pass

            def on_click(event):
                try:
                    if event.button() == Qt.LeftButton:
                        event.accept()
                    else:
                        return
                except Exception:
                    pass
                self._toggle_marker_text_visible_from_legend()

            try:
                obj.mouseClickEvent = on_click
            except Exception:
                pass
        self._update_marker_text_legend_style()

    def _current_marker_label_state(self) -> dict:
        positions = dict((getattr(self, "marker_label_state", {}) or {}).get("positions") or {})
        for item in getattr(self, "fit_marker_text_items", []) or []:
            key = getattr(item, "_xrd_marker_key", None)
            if not key:
                continue
            try:
                pos = item.pos()
                positions[str(key)] = {
                    "x": float(pos.x()),
                    "y": float(pos.y()),
                    "connector": bool(getattr(item, "_xrd_connector_active", False)),
                }
            except Exception:
                pass
        return {
            "visible": bool(getattr(self, "marker_text_visible", True)),
            "positions": positions,
        }

    def _apply_marker_label_state(self, state: dict | None) -> None:
        state = dict(state or {})
        self.marker_label_state = {
            "visible": bool(state.get("visible", True)),
            "positions": dict(state.get("positions") or {}),
        }
        self.marker_text_visible = bool(self.marker_label_state.get("visible", True))
        self._apply_marker_label_state_to_items()

    def _save_current_marker_label_state(self) -> None:
        self.marker_label_state = self._current_marker_label_state()
        index = getattr(self, "active_sample_index", -1)
        samples = getattr(self, "samples", [])
        if 0 <= index < len(samples):
            samples[index].marker_label_state = self.marker_label_state

    def _apply_marker_label_state_to_items(self) -> None:
        state = getattr(self, "marker_label_state", {}) or {}
        positions = dict(state.get("positions") or {})
        self.marker_text_visible = bool(state.get("visible", getattr(self, "marker_text_visible", True)))
        for item in getattr(self, "fit_marker_text_items", []) or []:
            key = getattr(item, "_xrd_marker_key", None)
            saved = positions.get(str(key)) if key is not None else None
            if saved:
                try:
                    item.setPos(float(saved["x"]), float(saved["y"]))
                    if hasattr(item, "_xrd_connector_active"):
                        item._xrd_connector_active = bool(saved.get("connector", True))
                        item._update_connector()
                except Exception:
                    pass
            try:
                if hasattr(item, "set_marker_visible"):
                    item.set_marker_visible(self.marker_text_visible)
                else:
                    item.setVisible(self.marker_text_visible)
            except Exception:
                pass
        self._update_marker_text_legend_style()

    @staticmethod
    def _rgba(color: str, alpha: float) -> QtGui.QColor:
        qcolor = QtGui.QColor(str(color))
        if not qcolor.isValid():
            qcolor = QtGui.QColor("#000000")
        qcolor.setAlphaF(max(0.0, min(1.0, float(alpha))))
        return qcolor

    @staticmethod
    def _curve_pen(color: str, width: float = 1.5, style=Qt.SolidLine, alpha: float = 1.0) -> QtGui.QPen:
        return pg.mkPen(PlotPanelMixin._rgba(color, alpha), width=width, style=style)

    def _scatter(self, plot: pg.PlotWidget, x, y, color: str, *, alpha: float = 0.5, size: float = 4, name: str | None = None):
        return plot.plot(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            pen=None,
            symbol="o",
            symbolSize=size,
            symbolBrush=pg.mkBrush(self._rgba(color, alpha)),
            symbolPen=None,
            name=name,
        )

    def _line(self, plot: pg.PlotWidget, x, y, color: str, *, width: float = 1.5, style=Qt.SolidLine, alpha: float = 1.0, name: str | None = None):
        return plot.plot(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            pen=self._curve_pen(color, width=width, style=style, alpha=alpha),
            name=name,
        )

    def _add_fill(self, plot: pg.PlotWidget, x, y_top, y_bottom, color: str, *, alpha: float = 0.2):
        top_curve = pg.PlotCurveItem(np.asarray(x, dtype=float), np.asarray(y_top, dtype=float), pen=None)
        bottom_curve = pg.PlotCurveItem(np.asarray(x, dtype=float), np.asarray(y_bottom, dtype=float), pen=None)
        fill = pg.FillBetweenItem(top_curve, bottom_curve, brush=pg.mkBrush(self._rgba(color, alpha)))
        plot.addItem(top_curve)
        plot.addItem(bottom_curve)
        plot.addItem(fill)
        return {"top": top_curve, "bottom": bottom_curve, "fill": fill}

    @staticmethod
    def _set_items_visible(items, visible: bool) -> None:
        if not isinstance(items, (list, tuple)):
            items = [items]
        for item in items:
            if item is None:
                continue
            try:
                item.setVisible(bool(visible))
            except Exception:
                pass

    def _save_canvas_image(self, canvas):
        path, _ = QFileDialog.getSaveFileName(
            canvas,
            "保存图片",
            "",
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;所有文件 (*)",
        )
        if not path:
            return
        canvas.grab().save(path)

    def _safe_draw_idle(self):
        for widget in (
            getattr(self, "preview_plot", None),
            getattr(self, "fit_plot", None),
            getattr(self, "size_plot", None),
            getattr(self, "compare_preview_plot", None),
            getattr(self, "compare_size_plot", None),
        ):
            if widget is not None:
                widget.update()

    def _safe_hide(self, artist):
        self._set_items_visible(artist, False)

    def _current_manual_baseline_state(self) -> dict:
        endpoint_y = getattr(self, "manual_baseline_endpoint_y", {"left": None, "right": None}) or {}
        user_points = []
        for point in getattr(self, "manual_baseline_user_points", []) or []:
            try:
                x_val = float(point.get("x"))
                y_val = float(point.get("y"))
            except Exception:
                continue
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue
            user_points.append(
                {
                    "id": int(point.get("id", len(user_points) + 1)),
                    "x": x_val,
                    "y": y_val,
                }
            )
        return {
            "enabled": bool(getattr(self, "manual_baseline_enabled", False)),
            "edited": bool(getattr(self, "manual_baseline_edited", False)),
            "user_points": user_points,
            "endpoint_y": {
                "left": endpoint_y.get("left"),
                "right": endpoint_y.get("right"),
            },
            "endpoint_deleted": sorted(str(v) for v in getattr(self, "manual_baseline_endpoint_deleted", set())),
        }

    def _apply_manual_baseline_state(self, state: dict | None) -> None:
        state = dict(state or {})
        self.manual_baseline_enabled = bool(state.get("enabled", False))
        self.manual_baseline_edited = bool(state.get("edited", False))
        endpoint_y = dict(state.get("endpoint_y") or {})
        self.manual_baseline_endpoint_y = {
            "left": endpoint_y.get("left"),
            "right": endpoint_y.get("right"),
        }
        self.manual_baseline_endpoint_deleted = set(state.get("endpoint_deleted") or [])
        self.manual_baseline_user_points = []
        max_id = 0
        for point in state.get("user_points") or []:
            try:
                point_id = int(point.get("id", max_id + 1))
                x_val = float(point.get("x"))
                y_val = float(point.get("y"))
            except Exception:
                continue
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue
            self.manual_baseline_user_points.append({"id": point_id, "x": x_val, "y": y_val})
            max_id = max(max_id, point_id)
        self._manual_baseline_next_anchor_id = max(1, max_id + 1)
        button = getattr(self, "btn_manual_baseline", None)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(self.manual_baseline_enabled)
            button.blockSignals(False)
        if self.manual_baseline_enabled:
            self._draw_manual_baseline_overlay()
        else:
            self._clear_manual_baseline_overlay()

    def _save_current_manual_baseline_state(self) -> None:
        index = getattr(self, "active_sample_index", -1)
        samples = getattr(self, "samples", [])
        if 0 <= index < len(samples):
            samples[index].baseline_state = self._current_manual_baseline_state()

    def _on_manual_baseline_toggled(self, checked: bool) -> None:
        self.manual_baseline_enabled = bool(checked)
        if self.manual_baseline_enabled:
            cursor = getattr(getattr(self, "fit_plot", None), "_click_projection_cursor", None)
            if cursor is not None and hasattr(cursor, "clear"):
                cursor.clear()
        self._save_current_manual_baseline_state()
        if self.manual_baseline_enabled:
            self._draw_manual_baseline_overlay()
        else:
            self._clear_manual_baseline_overlay()
        self._safe_draw_idle()

    @staticmethod
    def _auto_background_for_segment(x_values, y_values, angle_min: float, angle_max: float):
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        if x_arr.size == 0:
            return np.asarray([], dtype=float), 0.0, 0.0
        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
        if np.count_nonzero(finite) < 2:
            fill = float(y_arr[finite][0]) if np.any(finite) else 0.0
            return np.full_like(x_arr, fill, dtype=float), fill, fill
        bg_indices = np.where(finite & ((x_arr < angle_min + 0.5) | (x_arr > angle_max - 0.5)))[0]
        if len(bg_indices) < 2:
            finite_indices = np.where(finite)[0]
            bg_indices = np.array([finite_indices[0], finite_indices[-1]])
        slope_bg, intercept_bg = np.polyfit(x_arr[bg_indices], y_arr[bg_indices], 1)
        background = slope_bg * x_arr + intercept_bg
        return (
            background,
            float(slope_bg * float(angle_min) + intercept_bg),
            float(slope_bg * float(angle_max) + intercept_bg),
        )

    @staticmethod
    def _baseline_xy_arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        cleaned = []
        for point in points:
            try:
                x_val = float(point["x"])
                y_val = float(point["y"])
            except Exception:
                continue
            if np.isfinite(x_val) and np.isfinite(y_val):
                cleaned.append((x_val, y_val))
        cleaned.sort(key=lambda pair: pair[0])
        unique = []
        for x_val, y_val in cleaned:
            if unique and abs(x_val - unique[-1][0]) <= 1e-9:
                unique[-1] = (x_val, y_val)
            else:
                unique.append((x_val, y_val))
        if not unique:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        x_points, y_points = zip(*unique)
        return np.asarray(x_points, dtype=float), np.asarray(y_points, dtype=float)

    def _manual_baseline_anchor_points_for_segment(
        self,
        x_values,
        y_values,
        angle_min: float,
        angle_max: float,
        state: dict | None = None,
    ) -> list[dict]:
        state = dict(state or self._current_manual_baseline_state())
        _, left_auto, right_auto = self._auto_background_for_segment(x_values, y_values, angle_min, angle_max)
        endpoint_y = dict(state.get("endpoint_y") or {})
        deleted = set(state.get("endpoint_deleted") or [])
        low, high = sorted((float(angle_min), float(angle_max)))
        points: list[dict] = []
        if "left" not in deleted:
            y_val = endpoint_y.get("left")
            points.append(
                {
                    "id": "left",
                    "kind": "left",
                    "x": low,
                    "y": float(y_val) if y_val is not None and np.isfinite(float(y_val)) else left_auto,
                }
            )
        for point in state.get("user_points") or []:
            try:
                x_val = float(point.get("x"))
                y_val = float(point.get("y"))
                point_id = int(point.get("id"))
            except Exception:
                continue
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue
            if low <= x_val <= high:
                points.append({"id": point_id, "kind": "user", "x": x_val, "y": y_val})
        if "right" not in deleted:
            y_val = endpoint_y.get("right")
            points.append(
                {
                    "id": "right",
                    "kind": "right",
                    "x": high,
                    "y": float(y_val) if y_val is not None and np.isfinite(float(y_val)) else right_auto,
                }
            )
        points.sort(key=lambda point: float(point["x"]))
        return points

    def _evaluate_manual_baseline(self, x_values, points: list[dict]) -> np.ndarray:
        x_arr = np.asarray(x_values, dtype=float)
        x_points, y_points = self._baseline_xy_arrays(points)
        if x_points.size == 0:
            return np.zeros_like(x_arr, dtype=float)
        if x_points.size == 1:
            return np.full_like(x_arr, float(y_points[0]), dtype=float)
        if x_points.size == 2:
            return np.interp(x_arr, x_points, y_points, left=y_points[0], right=y_points[-1])
        try:
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(x_points, y_points, bc_type="natural", extrapolate=False)
            baseline = np.asarray(spline(x_arr), dtype=float)
            baseline[x_arr < x_points[0]] = y_points[0]
            baseline[x_arr > x_points[-1]] = y_points[-1]
            nan_mask = ~np.isfinite(baseline)
            if np.any(nan_mask):
                baseline[nan_mask] = np.interp(
                    x_arr[nan_mask],
                    x_points,
                    y_points,
                    left=y_points[0],
                    right=y_points[-1],
                )
            return baseline
        except Exception:
            return np.interp(x_arr, x_points, y_points, left=y_points[0], right=y_points[-1])

    def _compute_background_for_segment(
        self,
        x_values,
        y_values,
        angle_min: float,
        angle_max: float,
        state: dict | None = None,
    ) -> np.ndarray:
        auto_background, _, _ = self._auto_background_for_segment(x_values, y_values, angle_min, angle_max)
        state = dict(state or self._current_manual_baseline_state())
        if not (state.get("enabled") and state.get("edited")):
            return auto_background
        points = self._manual_baseline_anchor_points_for_segment(
            x_values,
            y_values,
            angle_min,
            angle_max,
            state,
        )
        x_points, _ = self._baseline_xy_arrays(points)
        if x_points.size < 2:
            return auto_background
        return self._evaluate_manual_baseline(x_values, points)

    def _clear_manual_baseline_overlay(self) -> None:
        items = []
        if getattr(self, "_manual_baseline_curve_item", None) is not None:
            items.append(self._manual_baseline_curve_item)
        items.extend(getattr(self, "_manual_baseline_anchor_items", []) or [])
        for item in items:
            try:
                self.fit_plot.removeItem(item)
            except Exception:
                pass
        self._manual_baseline_curve_item = None
        self._manual_baseline_anchor_items = []

    def _draw_manual_baseline_overlay(self) -> None:
        self._clear_manual_baseline_overlay()
        if not getattr(self, "manual_baseline_enabled", False):
            return
        if not getattr(self, "data_loaded", False) or not hasattr(self, "fit_plot"):
            return
        angle_min, angle_max = self._normalize_angle_range()
        low, high = sorted((float(angle_min), float(angle_max)))
        x_data = np.asarray(getattr(self, "x_data", []), dtype=float)
        y_data = np.asarray(getattr(self, "y_data", []), dtype=float)
        mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data >= low) & (x_data <= high)
        if not np.any(mask):
            return
        x_segment = x_data[mask]
        y_segment = y_data[mask]
        state = self._current_manual_baseline_state()
        background = self._compute_background_for_segment(x_segment, y_segment, low, high, state)
        line = self._line(self.fit_plot, x_segment, background, "#2563eb", width=2.0, alpha=0.95)
        line.setZValue(2200)
        self._manual_baseline_curve_item = line

        points = self._manual_baseline_anchor_points_for_segment(x_segment, y_segment, low, high, state)
        for point in points:
            item = BaselineAnchorItem(
                self,
                point["id"],
                str(point["kind"]),
                (float(point["x"]), float(point["y"])),
            )
            item.sigPositionChanged.connect(self._on_manual_baseline_anchor_position_changed)
            item.sigPositionChangeFinished.connect(self._on_manual_baseline_anchor_change_finished)
            self.fit_plot.addItem(item, ignoreBounds=True)
            self._manual_baseline_anchor_items.append(item)

    def _refresh_manual_baseline_curve(self) -> None:
        if not getattr(self, "manual_baseline_enabled", False):
            return
        curve = getattr(self, "_manual_baseline_curve_item", None)
        if curve is None:
            self._draw_manual_baseline_overlay()
            return
        angle_min, angle_max = self._normalize_angle_range()
        low, high = sorted((float(angle_min), float(angle_max)))
        x_data = np.asarray(getattr(self, "x_data", []), dtype=float)
        y_data = np.asarray(getattr(self, "y_data", []), dtype=float)
        mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data >= low) & (x_data <= high)
        if not np.any(mask):
            self._clear_manual_baseline_overlay()
            return
        x_segment = x_data[mask]
        y_segment = y_data[mask]
        background = self._compute_background_for_segment(
            x_segment,
            y_segment,
            low,
            high,
            self._current_manual_baseline_state(),
        )
        curve.setData(x_segment, background)
        self._safe_draw_idle()

    def _manual_baseline_anchor_is_near(self, scene_pos, radius: float = 12.0) -> bool:
        for item in getattr(self, "_manual_baseline_anchor_items", []) or []:
            try:
                anchor_pos = item.mapToScene(QtCore.QPointF(0, 0))
                dx = anchor_pos.x() - scene_pos.x()
                dy = anchor_pos.y() - scene_pos.y()
                if (dx * dx + dy * dy) ** 0.5 <= radius:
                    return True
            except Exception:
                continue
        return False

    def _on_fit_plot_mouse_clicked(self, event) -> None:
        if not getattr(self, "manual_baseline_enabled", False):
            return
        try:
            if event.button() != Qt.LeftButton or event.isAccepted():
                return
        except Exception:
            return
        view_box = self.fit_plot.getPlotItem().getViewBox()
        scene_pos = event.scenePos()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return
        if self._manual_baseline_anchor_is_near(scene_pos):
            return
        view_pos = view_box.mapSceneToView(scene_pos)
        x_val = float(view_pos.x())
        y_val = float(view_pos.y())
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            return
        angle_min, angle_max = self._normalize_angle_range()
        low, high = sorted((float(angle_min), float(angle_max)))
        if x_val < low or x_val > high:
            return
        for line in getattr(self, "peak_mu_lines_axes0", []) or []:
            if line is None:
                continue
            try:
                if abs(x_val - float(line.value())) <= self._peak_hit_tolerance(self.fit_plot):
                    return
            except Exception:
                continue
        self._add_manual_baseline_anchor(x_val, y_val)
        event.accept()

    def _add_manual_baseline_anchor(self, x_val: float, y_val: float) -> None:
        point_id = int(getattr(self, "_manual_baseline_next_anchor_id", 1))
        self._manual_baseline_next_anchor_id = point_id + 1
        self.manual_baseline_user_points.append({"id": point_id, "x": float(x_val), "y": float(y_val)})
        self.manual_baseline_edited = True
        self._save_current_manual_baseline_state()
        self._draw_manual_baseline_overlay()
        self._safe_draw_idle()

    def _on_manual_baseline_anchor_position_changed(self, item) -> None:
        if getattr(self, "_syncing_manual_baseline_anchor", False):
            return
        if not getattr(self, "manual_baseline_enabled", False):
            return
        pos = item.pos()
        try:
            x_val = float(pos.x())
            y_val = float(pos.y())
        except Exception:
            return
        if not np.isfinite(y_val):
            return
        angle_min, angle_max = self._normalize_angle_range()
        low, high = sorted((float(angle_min), float(angle_max)))
        kind = getattr(item, "_xrd_kind", "user")
        anchor_id = getattr(item, "_xrd_anchor_id", None)
        if kind in {"left", "right"}:
            x_target = low if kind == "left" else high
            self.manual_baseline_endpoint_y[kind] = y_val
            self.manual_baseline_endpoint_deleted.discard(kind)
        else:
            x_target = min(high, max(low, x_val))
            for point in self.manual_baseline_user_points:
                if int(point.get("id", -1)) == int(anchor_id):
                    point["x"] = x_target
                    point["y"] = y_val
                    break
        if abs(x_target - x_val) > 1e-9:
            self._syncing_manual_baseline_anchor = True
            try:
                item.setPos(x_target, y_val)
            finally:
                self._syncing_manual_baseline_anchor = False
        self.manual_baseline_edited = True
        self._save_current_manual_baseline_state()
        self._refresh_manual_baseline_curve()

    def _on_manual_baseline_anchor_change_finished(self, _item) -> None:
        self._save_current_manual_baseline_state()
        self._draw_manual_baseline_overlay()
        self._safe_draw_idle()

    def _delete_manual_baseline_anchor(self, anchor_id, kind: str) -> None:
        if kind in {"left", "right"}:
            self.manual_baseline_endpoint_deleted.add(kind)
            self.manual_baseline_endpoint_y[kind] = None
        else:
            try:
                anchor_id = int(anchor_id)
            except Exception:
                return
            self.manual_baseline_user_points = [
                point for point in self.manual_baseline_user_points
                if int(point.get("id", -1)) != anchor_id
            ]
        self.manual_baseline_edited = True
        self._save_current_manual_baseline_state()
        self._draw_manual_baseline_overlay()
        self._safe_draw_idle()

    def _enforce_clipping(self, _ax):
        return

    def _style_axis(self, _ax):
        return

    @staticmethod
    def _plot_range(plot: pg.PlotWidget):
        view_range = plot.getPlotItem().getViewBox().viewRange()
        return view_range[0], view_range[1]

    @staticmethod
    def _format_x_coordinate(value: float) -> str:
        try:
            return f"{float(value):.2f}"
        except Exception:
            return ""

    def _line_coordinate_label(self, plot: pg.PlotWidget, line) -> pg.TextItem | None:
        if plot is None or line is None:
            return None
        label = getattr(line, "_xrd_coordinate_label", None)
        if label is not None:
            return label
        label = pg.TextItem(
            text="",
            color="#111827",
            anchor=(0.0, 1.0),
            fill=pg.mkBrush(255, 255, 255, 235),
            border=pg.mkPen("#2563eb", width=1.0),
        )
        label.setZValue(6000)
        try:
            label.setAcceptedMouseButtons(Qt.LeftButton)
            label.setCursor(Qt.IBeamCursor)
        except Exception:
            pass

        def label_mouse_click(event, coord_label=label):
            try:
                if event.button() == Qt.LeftButton:
                    event.accept()
                    self._edit_coordinate_label_inline(coord_label)
                    return
            except Exception:
                pass

        label.mouseClickEvent = label_mouse_click
        try:
            plot.addItem(label, ignoreBounds=True)
        except Exception:
            return None
        line._xrd_coordinate_label = label
        label.hide()
        return label

    def _discard_coordinate_label(self, line) -> None:
        if line is None:
            return
        label = getattr(line, "_xrd_coordinate_label", None)
        if label is None:
            return
        editor = getattr(label, "_xrd_editor", None)
        if editor is not None:
            try:
                editor.hide()
                editor.deleteLater()
            except Exception:
                pass
            label._xrd_editor = None
            label._xrd_editing = False
        plot = getattr(label, "_xrd_plot", None) or getattr(line, "_xrd_plot", None)
        if plot is not None:
            try:
                plot.removeItem(label)
            except Exception:
                pass
        try:
            label.hide()
        except Exception:
            pass
        line._xrd_coordinate_label = None
        line._xrd_coordinate_label_token = None

    def _clear_peak_coordinate_artifacts(self, peak_idx: int | None = None) -> None:
        for lines in (
            getattr(self, "peak_mu_lines_preview", []),
            getattr(self, "peak_mu_lines_axes0", []),
        ):
            for line in list(lines or []):
                if line is None:
                    continue
                if peak_idx is not None and getattr(line, "_xrd_peak_idx", None) != peak_idx:
                    continue
                self._discard_coordinate_label(line)

    def _clear_plot_coordinate_artifacts(self, plot: pg.PlotWidget) -> None:
        for line in (getattr(self, "line_min", None), getattr(self, "line_max", None)):
            if getattr(line, "_xrd_plot", None) is plot:
                self._discard_coordinate_label(line)
        for lines in (
            getattr(self, "peak_mu_lines_preview", []),
            getattr(self, "peak_mu_lines_axes0", []),
        ):
            for line in list(lines or []):
                if line is not None and getattr(line, "_xrd_plot", None) is plot:
                    self._discard_coordinate_label(line)
        try:
            for editor in plot.findChildren(QtWidgets.QLineEdit, "xrdCoordinateInlineEditor"):
                editor.hide()
                editor.deleteLater()
        except Exception:
            pass

    def _show_line_coordinate_label(
        self,
        plot: pg.PlotWidget,
        line,
        preferred_side: str = "right",
        *,
        timeout_ms: int = 1000,
    ) -> None:
        label = self._line_coordinate_label(plot, line)
        if label is None:
            return
        self._cancel_coordinate_editor_if_unchanged(label)
        try:
            x_value = float(line.value())
            x_range, y_range = self._plot_range(plot)
            x_left, x_right = sorted((float(x_range[0]), float(x_range[1])))
            y_bottom = min(float(y_range[0]), float(y_range[1]))
        except Exception:
            return
        if not all(np.isfinite(v) for v in (x_value, x_left, x_right, y_bottom)):
            return

        span = max(abs(x_right - x_left), 1e-9)
        edge_margin = span * 0.045
        side = preferred_side if preferred_side in {"left", "right"} else "right"
        if side == "left" and x_value - x_left <= edge_margin:
            side = "right"
        elif side == "right" and x_right - x_value <= edge_margin:
            side = "left"
        label.setAnchor((1.0, 1.0) if side == "left" else (0.0, 1.0))
        label.setText(self._format_x_coordinate(x_value))
        label.setPos(x_value, y_bottom)
        label._xrd_line = line
        label._xrd_plot = plot
        label._xrd_side = side
        label.show()

        token = object()
        line._xrd_coordinate_label_token = token

        def hide_if_current() -> None:
            if getattr(label, "_xrd_editing", False):
                return
            if getattr(line, "_xrd_coordinate_label_token", None) is token:
                try:
                    label.hide()
                except Exception:
                    pass

        if int(timeout_ms) > 0:
            QtCore.QTimer.singleShot(int(timeout_ms), hide_if_current)

    def _edit_coordinate_label_inline(self, label) -> None:
        line = getattr(label, "_xrd_line", None)
        plot = getattr(label, "_xrd_plot", None)
        if line is None or plot is None:
            return
        old_editor = getattr(label, "_xrd_editor", None)
        if old_editor is not None:
            try:
                old_editor.setFocus()
                old_editor.selectAll()
                return
            except Exception:
                pass
        label._xrd_editing = True
        try:
            current = self._format_x_coordinate(float(line.value()))
        except Exception:
            current = str(label.toPlainText() if hasattr(label, "toPlainText") else "")
        editor = QtWidgets.QLineEdit(str(current), plot)
        editor.setObjectName("xrdCoordinateInlineEditor")
        editor.setAlignment(Qt.AlignCenter)
        editor.setFixedSize(70, 22)
        editor.setStyleSheet(
            """
            QLineEdit#xrdCoordinateInlineEditor {
                background: rgba(255, 255, 255, 235);
                border: 1px solid #2563eb;
                border-radius: 0;
                color: #111827;
                font: 9pt 'Microsoft YaHei UI';
                padding: 0 2px;
            }
            """
        )
        label._xrd_editor = editor
        label._xrd_editor_original_text = str(current)
        self._position_coordinate_editor(editor, label)
        editor.show()
        editor.raise_()
        editor.setFocus(Qt.MouseFocusReason)
        editor.selectAll()
        label.hide()
        done = {"active": False}

        def cancel() -> None:
            if done["active"]:
                return
            done["active"] = True
            try:
                editor.hide()
                editor.deleteLater()
            except Exception:
                pass
            label._xrd_editor = None
            label._xrd_editing = False
            label.show()

        def finish() -> None:
            if done["active"]:
                return
            done["active"] = True
            text = editor.text().strip()
            try:
                value = float(text)
            except Exception:
                done["active"] = False
                editor.setFocus(Qt.OtherFocusReason)
                editor.selectAll()
                return
            try:
                editor.hide()
                editor.deleteLater()
            except Exception:
                pass
            label._xrd_editor = None
            label._xrd_editing = False
            self._apply_coordinate_label_value(label, value)

        editor._xrd_cancel = cancel
        editor._xrd_finish = finish
        editor.returnPressed.connect(finish)
        editor.editingFinished.connect(finish)

    def _cancel_coordinate_editor_if_unchanged(self, label) -> None:
        editor = getattr(label, "_xrd_editor", None)
        if editor is None:
            return
        original = str(getattr(label, "_xrd_editor_original_text", ""))
        current = str(editor.text()).strip()
        if current != original:
            finish = getattr(editor, "_xrd_finish", None)
            if callable(finish):
                finish()
            return
        cancel = getattr(editor, "_xrd_cancel", None)
        if callable(cancel):
            cancel()

    def _position_coordinate_editor(self, editor: QtWidgets.QLineEdit, label) -> None:
        plot = getattr(label, "_xrd_plot", None)
        line = getattr(label, "_xrd_line", None)
        if plot is None or line is None:
            return
        try:
            rect = label.sceneBoundingRect()
            top_left = plot.mapFromScene(rect.topLeft())
            bottom_right = plot.mapFromScene(rect.bottomRight())
            if hasattr(top_left, "toPoint"):
                top_left = top_left.toPoint()
            if hasattr(bottom_right, "toPoint"):
                bottom_right = bottom_right.toPoint()
            x1, y1 = int(top_left.x()), int(top_left.y())
            x2, y2 = int(bottom_right.x()), int(bottom_right.y())
            x, y = min(x1, x2), min(y1, y2)
            width = max(46, abs(x2 - x1))
            height = max(20, abs(y2 - y1))
            editor.setFixedSize(width, height)
        except Exception:
            x, y = 8, max(4, plot.height() - editor.height() - 28)
        x = max(4, min(x, max(4, plot.width() - editor.width() - 4)))
        y = max(4, min(y, max(4, plot.height() - editor.height() - 4)))
        editor.move(x, y)

    def _apply_coordinate_label_value(self, label, value: float) -> None:
        line = getattr(label, "_xrd_line", None)
        if line is None:
            return
        kind = getattr(line, "_xrd_coordinate_type", "peak")
        if kind == "range":
            self._apply_range_boundary_value(line, value)
            self._show_range_line_coordinate(line, timeout_ms=2200)
            return
        peak_idx = getattr(line, "_xrd_peak_idx", None)
        if peak_idx is None:
            return
        self._apply_peak_line_value(int(peak_idx), line, value)
        self._show_peak_line_coordinate(line, timeout_ms=2200)

    def _range_line_preferred_label_side(self, line) -> str:
        region = getattr(self, "preview_range_region", None)
        if region is None or line is None:
            return "right"
        try:
            left, right = sorted(float(v) for v in region.getRegion())
            value = float(line.value())
            return "left" if abs(value - left) <= abs(value - right) else "right"
        except Exception:
            return getattr(line, "_xrd_range_label_side", "right")

    def _install_range_line_interaction(self, line, preferred_side: str) -> None:
        if line is None or getattr(line, "_xrd_range_interaction_installed", False):
            return
        line._xrd_range_label_side = preferred_side
        try:
            line.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        except Exception:
            pass
        original_mouse_click = line.mouseClickEvent

        def mouse_click_event(event, *, range_line=line, original=original_mouse_click):
            try:
                if event.button() == Qt.RightButton:
                    event.accept()
                    self._show_range_line_coordinate(range_line, timeout_ms=2200)
                    return
            except Exception:
                pass
            return original(event)

        line.mouseClickEvent = mouse_click_event
        try:
            line.sigPositionChanged.connect(
                lambda item, range_line=line: self._show_range_line_coordinate(range_line)
            )
            line.sigPositionChangeFinished.connect(
                lambda item, range_line=line: self._show_range_line_coordinate(range_line, timeout_ms=1200)
            )
        except Exception:
            pass
        line._xrd_range_interaction_installed = True
        line._xrd_coordinate_type = "range"
        line._xrd_plot = self.preview_plot

    def _install_range_region_context_menu(self, region) -> None:
        if region is None or getattr(region, "_xrd_context_menu_installed", False):
            return
        original_mouse_click = region.mouseClickEvent

        def mouse_click_event(event, *, original=original_mouse_click):
            try:
                if event.button() == Qt.RightButton:
                    event.accept()
                    self._show_preview_region_context_menu()
                    return
            except Exception:
                pass
            return original(event)

        region.mouseClickEvent = mouse_click_event
        region._xrd_context_menu_installed = True

    def _show_range_line_coordinate(self, line, *, timeout_ms: int = 1800) -> None:
        self._show_line_coordinate_label(
            self.preview_plot,
            line,
            self._range_line_preferred_label_side(line),
            timeout_ms=timeout_ms,
        )

    def _show_peak_line_coordinate(self, line, *, timeout_ms: int = 1800) -> None:
        plot = getattr(line, "_xrd_plot", None)
        if plot is None:
            source = getattr(line, "_xrd_source", "")
            plot = self.preview_plot if source == "preview" else self.fit_plot
        self._show_line_coordinate_label(plot, line, "right", timeout_ms=timeout_ms)

    def _apply_range_boundary_value(self, line, value: float) -> None:
        region = getattr(self, "preview_range_region", None)
        if region is None or line is None:
            return
        try:
            left, right = sorted(float(v) for v in region.getRegion())
        except Exception:
            return
        low_bound, high_bound = self._peak_value_bounds()
        value = max(float(low_bound), min(float(high_bound), float(value)))
        role = self._range_line_preferred_label_side(line)
        if role == "left":
            new_region = (value, right)
        else:
            new_region = (left, value)
        try:
            region.setRegion(tuple(sorted(new_region)))
        except Exception:
            return
        self._on_range_region_finished()

    def _apply_peak_line_value(self, peak_idx: int, line, value: float) -> None:
        if peak_idx < 0 or peak_idx >= len(getattr(self, "peak_mu_sliders", [])):
            return
        low_bound, high_bound = self._peak_value_bounds()
        value = max(float(low_bound), min(float(high_bound), float(value)))
        self._syncing_peak_line = True
        try:
            self.peak_mu_sliders[peak_idx].set(value, emit=False)
            self._sync_peak_line_value(peak_idx, value)
        finally:
            self._syncing_peak_line = False
        self._save_current_peak_states()

    def _on_preview_plot_mouse_clicked(self, event) -> None:
        try:
            if event.button() != Qt.RightButton or event.isAccepted():
                return
        except Exception:
            return
        region = getattr(self, "preview_range_region", None)
        if region is None or not getattr(self, "data_loaded", False):
            return
        view_box = self.preview_plot.getPlotItem().getViewBox()
        scene_pos = event.scenePos()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return
        try:
            x_value = float(view_box.mapSceneToView(scene_pos).x())
            left, right = sorted(float(v) for v in region.getRegion())
        except Exception:
            return
        if x_value < left or x_value > right:
            return
        event.accept()
        self._show_preview_region_context_menu()

    def _show_preview_region_context_menu(self) -> None:
        menu = QtWidgets.QMenu(self.preview_plot)
        apply_action = menu.addAction("应用全部")
        menu.addSeparator()
        view_all_action = menu.addAction("View All")
        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen == apply_action:
            self._apply_analysis_range_to_all_samples()
        elif chosen == view_all_action:
            try:
                self.preview_plot.getPlotItem().getViewBox().autoRange()
            except Exception:
                pass

    def _current_fit_default_range(self) -> tuple[float, float]:
        region = getattr(self, "preview_range_region", None)
        if region is not None:
            try:
                return tuple(sorted(float(v) for v in region.getRegion()))
            except Exception:
                pass
        if hasattr(self, "slider_min") and hasattr(self, "slider_max"):
            return tuple(sorted((float(self.slider_min.get()), float(self.slider_max.get()))))
        if hasattr(self, "x_segment"):
            x = np.asarray(self.x_segment, dtype=float)
            if x.size:
                return float(np.nanmin(x)), float(np.nanmax(x))
        return 0.0, 1.0

    def _install_fit_view_all_handler(self) -> None:
        vb = self.fit_plot.getPlotItem().getViewBox()
        if getattr(vb, "_xrd_view_all_wrapped", False):
            return
        original_auto_range = vb.autoRange

        def auto_range(padding=None, items=None, item=None):
            if getattr(self, "data_loaded", False):
                try:
                    left, right = self._current_fit_default_range()
                    self._set_fit_view_range(left, right)
                    return
                except Exception:
                    pass
            return original_auto_range(padding=padding, items=items, item=item)

        vb.autoRange = auto_range
        vb._xrd_view_all_wrapped = True

    def _capture_plot_views(self) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        views = {}
        for name in ("preview_plot", "fit_plot", "size_plot"):
            plot = getattr(self, name, None)
            if plot is None:
                continue
            try:
                x_range, y_range = self._plot_range(plot)
                views[name] = (
                    (float(x_range[0]), float(x_range[1])),
                    (float(y_range[0]), float(y_range[1])),
                )
            except Exception:
                pass
        return views

    def _restore_plot_views(self, views: dict[str, tuple[tuple[float, float], tuple[float, float]]]) -> None:
        for name, ranges in (views or {}).items():
            plot = getattr(self, name, None)
            if plot is None:
                continue
            try:
                x_range, y_range = ranges
                plot.getPlotItem().getViewBox().setRange(
                    xRange=x_range,
                    yRange=y_range,
                    padding=0,
                    disableAutoRange=True,
                )
            except Exception:
                pass

    def _current_plot_view_state(self) -> dict:
        views = {}
        for name in ("fit_plot", "size_plot"):
            plot = getattr(self, name, None)
            if plot is None:
                continue
            try:
                x_range, y_range = self._plot_range(plot)
                views[name] = {
                    "x": [float(x_range[0]), float(x_range[1])],
                    "y": [float(y_range[0]), float(y_range[1])],
                }
            except Exception:
                pass
        return views

    def _apply_plot_view_state(self, state: dict | None) -> None:
        self.plot_view_state = dict(state or {})

    def _save_current_plot_view_state(self) -> None:
        if not getattr(self, "results_ready", False):
            return
        self.plot_view_state = self._current_plot_view_state()
        index = getattr(self, "active_sample_index", -1)
        samples = getattr(self, "samples", [])
        if 0 <= index < len(samples):
            samples[index].plot_view_state = self.plot_view_state

    def _restore_sample_plot_view_state(self) -> None:
        state = getattr(self, "plot_view_state", {}) or {}
        views = {}
        for name in ("fit_plot", "size_plot"):
            ranges = state.get(name)
            if not ranges:
                continue
            try:
                views[name] = (
                    (float(ranges["x"][0]), float(ranges["x"][1])),
                    (float(ranges["y"][0]), float(ranges["y"][1])),
                )
            except Exception:
                continue
        if views:
            self._restore_plot_views(views)

    @staticmethod
    def _set_plot_xrange(plot: pg.PlotWidget, left: float, right: float, *, padding: float = 0.0) -> None:
        if not np.isfinite(left) or not np.isfinite(right) or left == right:
            return
        plot.setXRange(float(min(left, right)), float(max(left, right)), padding=padding)

    @staticmethod
    def _set_resize_cursor(item) -> None:
        if item is None:
            return
        try:
            item.setCursor(Qt.SizeHorCursor)
        except Exception:
            pass

    def _set_fit_autorange_sources(self, *sources) -> None:
        cleaned = []
        for x_values, y_values in sources:
            try:
                x_arr = np.asarray(x_values, dtype=float)
                y_arr = np.asarray(y_values, dtype=float)
            except Exception:
                continue
            if x_arr.size and y_arr.size and x_arr.size == y_arr.size:
                cleaned.append((x_arr, y_arr))
        self._fit_autorange_sources = cleaned

    def _auto_fit_y_range(self, left: float, right: float) -> None:
        low, high = sorted((float(left), float(right)))
        y_parts = []
        for x_arr, y_arr in getattr(self, "_fit_autorange_sources", []):
            mask = (x_arr >= low) & (x_arr <= high) & np.isfinite(y_arr)
            if np.any(mask):
                y_parts.append(y_arr[mask])
        if not y_parts:
            return
        y_values = np.concatenate(y_parts)
        if y_values.size == 0:
            return
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        pad = max((y_max - y_min) * 0.08, abs(y_max) * 0.02, 1.0)
        self.fit_plot.setYRange(y_min - pad, y_max + pad, padding=0)

    def _set_fit_view_range(self, left: float, right: float) -> None:
        self._set_plot_xrange(self.fit_plot, left, right)
        self._auto_fit_y_range(left, right)

    def _peak_hit_tolerance(self, plot: pg.PlotWidget, minimum: float = 0.12, pixels: float = 8.0) -> float:
        try:
            x_range, _ = self._plot_range(plot)
            width = max(float(plot.getPlotItem().vb.width()), 1.0)
            return max(minimum, abs(x_range[1] - x_range[0]) * pixels / width)
        except Exception:
            return minimum

    def _hide_axes0_overlays(self):
        self._clear_plot(self.fit_plot, title="XRD多峰拟合及粒径分解")
        self.peak_mu_rects_axes0 = []
        self.peak_mu_lines_axes0 = []
        self.axes0_context_artists = []
        self._ensure_plot_legend(self.fit_plot)
        self._safe_draw_idle()

    def _sync_axes0_peak_lines(self):
        for line, peak_idx in zip(getattr(self, "peak_mu_lines_axes0", []), getattr(self, "active_peak_indices", [])):
            if line is None or peak_idx >= len(self.peak_mu_sliders):
                continue
            mu = self.peak_mu_sliders[peak_idx].get()
            try:
                line.setValue(mu)
                line.setPen(self._curve_pen(self._peak_color(peak_idx), width=1.6, alpha=0.85))
                line.setVisible(self._peak_visible(peak_idx))
            except Exception:
                pass

    @staticmethod
    def _set_span_bounds(span, _ax, left: float, right: float) -> None:
        if span is None:
            return
        try:
            span.setRegion((float(left), float(right)))
        except Exception:
            try:
                span.setValue((float(left) + float(right)) / 2.0)
            except Exception:
                pass

    def _clear_axes0_peak_markers(self) -> None:
        for line in getattr(self, "peak_mu_lines_axes0", []):
            try:
                self.fit_plot.removeItem(line)
            except Exception:
                pass
        self.peak_mu_lines_axes0 = []

    def _make_peak_line(self, peak_idx: int, value: float, *, movable: bool, source: str):
        color = self._peak_color(peak_idx)
        line = pg.InfiniteLine(
            pos=float(value),
            angle=90,
            movable=movable,
            pen=self._curve_pen(color, width=1.6, alpha=0.88),
            hoverPen=self._curve_pen(color, width=2.2, alpha=1.0),
        )
        self._set_resize_cursor(line)
        try:
            line.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        except Exception:
            pass
        original_mouse_click = line.mouseClickEvent

        def mouse_click_event(event, *, idx=peak_idx, peak_line=line, original=original_mouse_click):
            try:
                if event.button() == Qt.RightButton:
                    event.accept()
                    if hasattr(self, "_delete_peak_index"):
                        self._delete_peak_index(idx)
                    return
            except Exception:
                pass
            return original(event)

        line.mouseClickEvent = mouse_click_event
        line.setZValue(1000)
        try:
            low, high = self._peak_value_bounds()
            line.setBounds((float(low), float(high)))
        except Exception:
            pass
        if movable:
            line.sigPositionChanged.connect(
                lambda item, idx=peak_idx, src=source: self._on_peak_line_position_changed(idx, item, src)
            )
            line.sigPositionChangeFinished.connect(
                lambda item, idx=peak_idx, src=source: self._on_peak_line_change_finished(idx, item, src)
            )
        line._xrd_peak_idx = peak_idx
        line._xrd_source = source
        line._xrd_plot = self.preview_plot if source == "preview" else self.fit_plot
        line._xrd_coordinate_type = "peak"
        return line

    def _redraw_axes0_peak_markers(self) -> None:
        self._clear_axes0_peak_markers()
        if not hasattr(self, "fit_plot"):
            return
        for peak_idx in getattr(self, "active_peak_indices", []):
            if peak_idx >= len(getattr(self, "peak_mu_sliders", [])) or not self._peak_visible(peak_idx):
                self.peak_mu_lines_axes0.append(None)
                continue
            line = self._make_peak_line(
                peak_idx,
                self.peak_mu_sliders[peak_idx].get(),
                movable=True,
                source="fit",
            )
            self.fit_plot.addItem(line, ignoreBounds=True)
            self.peak_mu_lines_axes0.append(line)

    def _refresh_axes0_context_for_view(self) -> None:
        if not getattr(self, "results_ready", False) or not getattr(self, "data_loaded", False):
            return
        x_range, y_range = self._plot_range(self.fit_plot)
        self._update_axes0_context_data(float(x_range[0]), float(x_range[1]), autoscale=False)
        self.fit_plot.setXRange(x_range[0], x_range[1], padding=0)
        self.fit_plot.setYRange(y_range[0], y_range[1], padding=0)

    def _redraw_axes0_range_preview(self, angle_min: float, angle_max: float) -> None:
        self._clear_plot(self.fit_plot, title="拟合范围预览")
        self.peak_mu_rects_axes0 = []
        self.peak_mu_lines_axes0 = []
        self.axes0_context_artists = []

        if getattr(self, "data_loaded", False):
            self._set_fit_autorange_sources((self.x_data, self.y_data))
            self._scatter(self.fit_plot, self.x_data, self.y_data, "#9ca3af", alpha=0.22, size=3)
            mask = (self.x_data >= angle_min) & (self.x_data <= angle_max)
            if np.any(mask):
                self._scatter(self.fit_plot, self.x_data[mask], self.y_data[mask], "#6b7280", alpha=0.62, size=4)
            for peak_idx in self.active_peak_indices:
                if peak_idx >= len(self.peak_mu_sliders) or not self._peak_visible(peak_idx):
                    self.peak_mu_rects_axes0.append(None)
                    self.peak_mu_lines_axes0.append(None)
                    continue
                line = self._make_peak_line(
                    peak_idx,
                    self.peak_mu_sliders[peak_idx].get(),
                    movable=True,
                    source="fit",
                )
                self.fit_plot.addItem(line, ignoreBounds=True)
                self.peak_mu_rects_axes0.append(line)
                self.peak_mu_lines_axes0.append(line)
            self._add_peak_legend_toggles(self.fit_plot, self.active_peak_indices)
            if getattr(self, "manual_baseline_enabled", False):
                self._draw_manual_baseline_overlay()
            self._set_fit_view_range(angle_min, angle_max)

    def _update_axes0_context_data(self, angle_min: float, angle_max: float, *, autoscale: bool = True) -> None:
        for item in getattr(self, "axes0_context_artists", []):
            try:
                self.fit_plot.removeItem(item)
            except Exception:
                pass
        self.axes0_context_artists = []
        if not getattr(self, "data_loaded", False):
            return
        mask = (self.x_data >= angle_min) & (self.x_data <= angle_max)
        if np.any(mask):
            item = self._scatter(self.fit_plot, self.x_data[mask], self.y_data[mask], "#9ca3af", alpha=0.24, size=3)
            item.setZValue(-10)
            self.axes0_context_artists.append(item)
        self._set_fit_view_range(angle_min, angle_max)
        if autoscale:
            self._auto_fit_y_range(angle_min, angle_max)

    def _on_range_region_changed(self):
        if self._syncing_range_region or self.preview_range_region is None:
            return
        if not getattr(self, "data_loaded", False):
            return
        left, right = sorted(float(v) for v in self.preview_range_region.getRegion())
        self._syncing_range_region = True
        try:
            self.slider_min.set(left, emit=False)
            self.slider_max.set(right, emit=False)
        finally:
            self._syncing_range_region = False
        self._set_fit_view_range(left, right)

    def _on_range_region_finished(self):
        if not getattr(self, "data_loaded", False):
            return
        left, right = sorted(float(v) for v in self.preview_range_region.getRegion())
        self._save_current_analysis_state()
        self._refresh_peak_slider_bounds()
        self._sync_all_peak_lines()
        if getattr(self, "results_ready", False):
            self._update_axes0_context_data(left, right, autoscale=False)
            self._sync_axes0_peak_lines()
            self._set_fit_view_range(left, right)
        else:
            self._redraw_axes0_range_preview(left, right)
        if getattr(self, "manual_baseline_enabled", False):
            self._draw_manual_baseline_overlay()

    def _sync_range_region(self, angle_min: float, angle_max: float) -> None:
        if self.preview_range_region is None:
            return
        self._syncing_range_region = True
        try:
            self.preview_range_region.setRegion((float(angle_min), float(angle_max)))
        finally:
            self._syncing_range_region = False

    def _sync_all_peak_lines(self):
        for peak_idx in range(len(getattr(self, "peak_mu_sliders", []))):
            self._sync_peak_line_value(peak_idx, self.peak_mu_sliders[peak_idx].get())

    def _sync_peak_line_value(self, peak_idx: int, value: float, *, exclude=None) -> None:
        for lines in (getattr(self, "peak_mu_lines_preview", []), getattr(self, "peak_mu_lines_axes0", [])):
            for line in lines:
                if line is None or line is exclude:
                    continue
                if getattr(line, "_xrd_peak_idx", None) != peak_idx:
                    continue
                try:
                    line.setValue(float(value))
                except Exception:
                    pass

    def _on_peak_line_position_changed(self, peak_idx: int, line, _source: str) -> None:
        if self._syncing_peak_line or not getattr(self, "data_loaded", False):
            return
        self._syncing_peak_line = True
        try:
            value = float(line.value())
            if 0 <= peak_idx < len(self.peak_mu_sliders):
                self.peak_mu_sliders[peak_idx].set(value, emit=False)
            self._sync_peak_line_value(peak_idx, value, exclude=line)
            self._show_peak_line_coordinate(line)
        finally:
            self._syncing_peak_line = False

    def _on_peak_line_change_finished(self, peak_idx: int, line, _source: str) -> None:
        if 0 <= peak_idx < len(self.peak_mu_sliders):
            self.peak_mu_sliders[peak_idx].set(float(line.value()), emit=False)
        self._save_current_peak_states()
        self._show_peak_line_coordinate(line, timeout_ms=1200)

    def update_preview(self, val=None):
        if not self.data_loaded:
            return

        angle_min, angle_max = self._normalize_angle_range()
        self._refresh_peak_slider_bounds()

        self._clear_plot(self.preview_plot, title="完整数据预览")
        self._ensure_plot_legend(self.preview_plot)
        self.peak_mu_rects_preview = []
        self.peak_mu_lines_preview = []

        self._scatter(
            self.preview_plot,
            self.x_data,
            self.y_data,
            "#6b7280",
            alpha=0.52,
            size=3.5,
            name=self.current_file_name,
        )

        self.preview_range_region = pg.LinearRegionItem(
            values=sorted((float(angle_min), float(angle_max))),
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(self._rgba("#2563eb", 0.10)),
            pen=self._curve_pen("#2563eb", width=2.0, style=Qt.DashLine, alpha=0.95),
            hoverBrush=pg.mkBrush(self._rgba("#2563eb", 0.14)),
            hoverPen=self._curve_pen("#2563eb", width=2.4, style=Qt.DashLine, alpha=1.0),
            movable=True,
        )
        self.preview_range_region.setZValue(20)
        self.preview_range_region.sigRegionChanged.connect(self._on_range_region_changed)
        self.preview_range_region.sigRegionChangeFinished.connect(self._on_range_region_finished)
        self.preview_plot.addItem(self.preview_range_region)
        self._install_range_region_context_menu(self.preview_range_region)
        self.preview_range_span = self.preview_range_region
        try:
            range_lines = list(self.preview_range_region.lines)
            range_lines.sort(key=lambda item: float(item.value()))
            self.line_min, self.line_max = range_lines
        except Exception:
            self.line_min = self.line_max = None
        range_pen = self._curve_pen("#2563eb", width=2.0, style=Qt.DashLine, alpha=0.95)
        range_hover_pen = self._curve_pen("#2563eb", width=2.4, style=Qt.DashLine, alpha=1.0)
        for line in (self.line_min, self.line_max):
            try:
                line.setPen(range_pen)
                line.setHoverPen(range_hover_pen)
            except Exception:
                pass
            self._set_resize_cursor(line)
        self._install_range_line_interaction(self.line_min, "left")
        self._install_range_line_interaction(self.line_max, "right")

        for peak_idx in self.active_peak_indices:
            if peak_idx >= len(self.peak_mu_sliders) or not self._peak_visible(peak_idx):
                self.peak_mu_rects_preview.append(None)
                self.peak_mu_lines_preview.append(None)
                continue
            line = self._make_peak_line(
                peak_idx,
                self.peak_mu_sliders[peak_idx].get(),
                movable=True,
                source="preview",
            )
            line._xrd_peak_idx = peak_idx
            self.preview_plot.addItem(line, ignoreBounds=True)
            self.peak_mu_rects_preview.append(line)
            self.peak_mu_lines_preview.append(line)
        self._add_peak_legend_toggles(self.preview_plot, self.active_peak_indices)

        self._set_plot_xrange(self.preview_plot, float(np.nanmin(self.x_data)), float(np.nanmax(self.x_data)))
        self.preview_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        if not self.results_ready:
            self._redraw_axes0_range_preview(angle_min, angle_max)
            self._clear_plot(self.size_plot, title="粒径分布 (计算后显示)")
            self.size_plot.setLabel("left", "Volume Density")
            self.size_plot.setLabel("bottom", "Particle size (nm)")
        else:
            self._update_axes0_context_data(angle_min, angle_max, autoscale=False)
            self._redraw_axes0_peak_markers()
            self._set_fit_view_range(angle_min, angle_max)
        if getattr(self, "manual_baseline_enabled", False):
            self._draw_manual_baseline_overlay()
        self._safe_draw_idle()

    def update_multi_peak_plots(self):
        """Update the fitted XRD plot and the particle-size distribution plot."""
        active_indices = list(getattr(self, "result_active_peak_indices", self.active_peak_indices))

        self._clear_plot(self.fit_plot, title="XRD多峰拟合及粒径分解")
        self.fit_plot.addLegend(
            offset=(10, 10),
            labelTextColor="#111827",
            brush=pg.mkBrush(255, 255, 255, 220),
            pen=pg.mkPen("#d1d5db"),
        )
        _sync_plot_legend_visibility(self.fit_plot)
        self.peak_mu_rects_axes0 = []
        self.peak_mu_lines_axes0 = []
        self.axes0_context_artists = []
        self.fit_marker_text_items = []

        x = np.asarray(self.x_segment, dtype=float)
        y_raw = np.asarray(self.y_segment_raw, dtype=float)
        y = np.asarray(self.y_segment, dtype=float)
        bg = np.asarray(self.background, dtype=float)
        fit_sources = []

        if getattr(self, "data_loaded", False):
            self._scatter(self.fit_plot, self.x_data, self.y_data, "#9ca3af", alpha=0.18, size=3)
            fit_sources.append((self.x_data, self.y_data))
        self._scatter(self.fit_plot, x, y_raw, "#6b7280", alpha=0.68, size=4, name="原始数据")
        self._line(self.fit_plot, x, bg, "#111827", width=1.4, style=Qt.DashLine, alpha=0.72, name="背景")

        fit_sources.extend([(x, y_raw), (x, bg)])
        total_fit_curve = np.zeros_like(x, dtype=float)

        for i, info in enumerate(self.all_peak_info):
            peak_id = active_indices[i] if i < len(active_indices) else i
            f_segment = np.asarray(info["f_segment"], dtype=float)
            basis_k1 = np.asarray(info["basis_k1"], dtype=float)
            basis_k2 = np.asarray(info["basis_k2"], dtype=float)
            peak_color = self._peak_color(peak_id)

            peak_fit = (basis_k1.dot(f_segment) + basis_k2.dot(f_segment)) * y.max() + bg
            total_fit_curve += peak_fit - bg
            fit_sources.append((x, peak_fit))

            for j, detail in enumerate(info.get("peak_details", [])):
                idx = detail.get("indices", None)
                if idx is None or len(idx) == 0:
                    continue
                idx = np.asarray(idx, dtype=int)
                f_component = np.zeros_like(f_segment)
                f_component[idx] = f_segment[idx]
                comp_fit = (basis_k1[:, idx].dot(f_component[idx]) + basis_k2[:, idx].dot(f_component[idx])) * y.max() + bg
                fit_sources.append((x, comp_fit))
                color = COMPONENT_COLORS[j % len(COMPONENT_COLORS)]
                line = self._line(self.fit_plot, x, comp_fit, color, width=1.5, alpha=0.95)
                fill_items = self._add_fill(self.fit_plot, x, comp_fit, bg, color, alpha=0.24)
                line.setZValue(15)
                pk_idx = int(np.nanargmax(comp_fit - bg))
                peak_anchor = (float(x[pk_idx]), float(comp_fit[pk_idx]))
                label = DraggableMarkerTextItem(
                    self,
                    self.fit_plot,
                    f"{float(detail['center']):.2f}nm({float(detail['percentage']):.0f}%)",
                    color,
                    peak_anchor,
                    marker_key=f"component:{peak_id}:{j}",
                    fill=pg.mkBrush(255, 255, 255, 205),
                )
                label.setZValue(40)
                self.fit_plot.addItem(label)
                label.setPos(float(x[pk_idx]), float(comp_fit[pk_idx] * 1.05))
                label.set_marker_visible(bool(getattr(self, "marker_text_visible", True)))
                self.fit_marker_text_items.append(label)

            pk_idx = int(np.nanargmax(peak_fit - bg))
            peak_label = pg.TextItem(
                text=f"Peak{peak_id + 1}",
                color=peak_color,
                anchor=(0.5, 1.0),
                fill=pg.mkBrush(255, 255, 255, 215),
            )
            peak_label._xrd_marker_key = f"peak:{peak_id}"
            peak_label.setZValue(45)
            self.fit_plot.addItem(peak_label)
            peak_label.setPos(float(x[pk_idx]), float(peak_fit[pk_idx]))
            peak_label.setVisible(bool(getattr(self, "marker_text_visible", True)))
            self.fit_marker_text_items.append(peak_label)

        self._line(self.fit_plot, x, total_fit_curve + bg, "#111111", width=2.4, alpha=0.76, name="总拟合")
        self.fit_plot.setLabel("left", "Intensity")
        self.fit_plot.setLabel("bottom", "2θ (°)")
        fit_sources.append((x, total_fit_curve + bg))
        self._set_fit_autorange_sources(*fit_sources)
        self._set_fit_view_range(float(np.nanmin(x)), float(np.nanmax(x)))
        self._redraw_axes0_peak_markers()
        self._add_peak_legend_toggles(self.fit_plot, active_indices)
        self._add_marker_text_legend_toggle(self.fit_plot)
        self._apply_marker_label_state_to_items()
        if getattr(self, "manual_baseline_enabled", False):
            self._draw_manual_baseline_overlay()

        self._redraw_size_distribution_plot(active_indices)
        self._safe_draw_idle()

    def _redraw_size_distribution_plot(self, active_indices: list[int]) -> None:
        self._clear_plot(self.size_plot, title="晶粒尺寸分布 (总分布 vs 分峰)")
        self._ensure_plot_legend(self.size_plot)
        self.legend_handles = {}
        self.actual_components = {}
        self.dist_texts = {}
        self._size_visibility = dict(getattr(self, "_size_visibility", {}))

        eps = 1e-12
        D_range = np.asarray(self.D_range, dtype=float)
        A_total_sum = 0.0
        for info in self.all_peak_info:
            A_total_sum += sum(float(det.get("area", 0.0)) for det in info.get("peak_details", []))
        A_total_sum = max(A_total_sum, eps)

        global_f_sum = sum(np.asarray(info["f_segment"], dtype=float) for info in self.all_peak_info)
        global_y_pdf = global_f_sum / A_total_sum

        global_line = self._line(
            self.size_plot,
            D_range,
            global_y_pdf,
            "#111111",
            width=2.4,
            style=Qt.DashLine,
            name="Total Distribution",
        )
        global_fill = self._add_fill(self.size_plot, D_range, global_y_pdf, np.zeros_like(global_y_pdf), "#6b7280", alpha=0.18)
        self.actual_components["global"] = {"items": [global_line, *global_fill.values()]}
        self.dist_texts["global"] = []

        y_max = float(np.nanmax(global_y_pdf)) if len(global_y_pdf) else 1.0

        for i, info in enumerate(self.all_peak_info):
            peak_id = active_indices[i] if i < len(active_indices) else i
            color = self._peak_color(peak_id)
            f_total = np.asarray(info["f_segment"], dtype=float)
            line_y_pdf = f_total / A_total_sum
            label = f"Peak{peak_id + 1} Particle Size Distribution"
            line = self._line(self.size_plot, D_range, line_y_pdf, color, width=2.0, name=label)
            fill = self._add_fill(self.size_plot, D_range, line_y_pdf, np.zeros_like(line_y_pdf), color, alpha=0.12)
            items = [line, *fill.values()]
            texts = []
            for det in info.get("peak_details", []):
                idx = det.get("indices", [])
                if len(idx) == 0:
                    continue
                idx = np.asarray(idx, dtype=int)
                local_max_idx = int(np.nanargmax(line_y_pdf[idx]))
                real_idx = int(idx[local_max_idx])
                cx = float(D_range[real_idx])
                cy = float(line_y_pdf[real_idx])
                y_max = max(y_max, cy)
                txt = pg.TextItem(text=f"{float(det['center']):.2f}nm", color=color, anchor=(0.5, 1.0))
                txt.setZValue(30)
                self.size_plot.addItem(txt)
                txt.setPos(cx, cy * 1.05)
                texts.append(txt)
                items.append(txt)
            self.actual_components[peak_id] = {"items": items}
            self.dist_texts[peak_id] = texts

        self.size_plot.setLabel("left", "Volume Density")
        self.size_plot.setLabel("bottom", "Particle size (nm)")
        self._set_plot_xrange(self.size_plot, float(np.nanmin(D_range)), float(np.nanmax(D_range)))
        self.size_plot.setYRange(0.0, y_max * 1.2 if y_max > 0 else 1.0, padding=0)

    def on_pick_legend(self, event):
        return

    def bind_events(self):
        # pyqtgraph's ViewBox supplies the BET-like left/right/wheel interaction.
        return

    def on_press(self, event):
        return

    def on_motion(self, event):
        return

    def on_release(self, event):
        return
