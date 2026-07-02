"""
ui/control_panel_mixin.py
--------------------------
BET-style PyQt5 panels:
  - left sidebar: import/update, sample table, sample parameters
  - analysis controls: compact numeric controls placed left of preview plot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .qt_controls import QtLabeledSpin, QtSpinSlider

SUPPORTED_DROP_SUFFIXES = {".txt", ".raw"}


def make_update_available_icon(size: int = 28) -> QtGui.QIcon:
    """Blue cloud download icon copied in spirit from the BET UI."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    scale = size / 28.0

    def s(value: float) -> float:
        return float(value) * scale

    cloud = QtGui.QPainterPath()
    cloud.moveTo(s(6.4), s(22.0))
    cloud.cubicTo(s(3.3), s(22.0), s(1.4), s(19.8), s(1.4), s(17.1))
    cloud.cubicTo(s(1.4), s(14.4), s(3.4), s(12.3), s(6.0), s(12.2))
    cloud.cubicTo(s(6.8), s(8.6), s(9.9), s(5.9), s(13.8), s(5.9))
    cloud.cubicTo(s(17.0), s(5.9), s(19.7), s(7.8), s(21.0), s(10.8))
    cloud.cubicTo(s(24.2), s(11.1), s(26.6), s(13.5), s(26.6), s(16.6))
    cloud.cubicTo(s(26.6), s(19.6), s(24.2), s(22.0), s(21.0), s(22.0))
    cloud.lineTo(s(6.4), s(22.0))
    cloud.closeSubpath()
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor("#2563eb"))
    painter.drawPath(cloud)

    arrow = QtGui.QPainterPath()
    arrow.setFillRule(QtCore.Qt.WindingFill)
    arrow.moveTo(s(14.0), s(8.6))
    arrow.cubicTo(s(12.9), s(8.6), s(12.0), s(9.5), s(12.0), s(10.6))
    arrow.lineTo(s(12.0), s(15.4))
    arrow.lineTo(s(8.4), s(15.4))
    arrow.lineTo(s(14.0), s(21.0))
    arrow.lineTo(s(19.6), s(15.4))
    arrow.lineTo(s(16.0), s(15.4))
    arrow.lineTo(s(16.0), s(10.6))
    arrow.cubicTo(s(16.0), s(9.5), s(15.1), s(8.6), s(14.0), s(8.6))
    arrow.closeSubpath()
    painter.setBrush(QtGui.QColor("#ffffff"))
    painter.drawPath(arrow)
    painter.end()
    return QtGui.QIcon(pixmap)


class EyeToggleButton(QtWidgets.QToolButton):
    """Small painted eye toggle used by the peak controls."""

    def __init__(self, visible: bool = True, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(bool(visible))
        self.setAutoRaise(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("显示/隐藏峰线")
        self.setFixedSize(22, 22)
        self.toggled.connect(lambda _checked: self.update())

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect().adjusted(3, 5, -3, -5)
        center = QtCore.QPointF(rect.center())
        width = float(max(12, rect.width()))
        height = float(max(7, rect.height()))
        color = QtGui.QColor("#2563eb" if self.isChecked() else "#9ca3af")

        painter.setPen(QtGui.QPen(color, 1.35))
        painter.setBrush(QtCore.Qt.NoBrush)
        path = QtGui.QPainterPath()
        path.moveTo(center.x() - width / 2.0, center.y())
        path.cubicTo(
            center.x() - width * 0.24,
            center.y() - height * 0.66,
            center.x() + width * 0.24,
            center.y() - height * 0.66,
            center.x() + width / 2.0,
            center.y(),
        )
        path.cubicTo(
            center.x() + width * 0.24,
            center.y() + height * 0.66,
            center.x() - width * 0.24,
            center.y() + height * 0.66,
            center.x() - width / 2.0,
            center.y(),
        )
        painter.drawPath(path)

        if self.isChecked():
            painter.setBrush(color)
            painter.drawEllipse(center, 2.2, 2.2)
        else:
            painter.drawLine(
                QtCore.QPointF(rect.left() + 1, rect.bottom()),
                QtCore.QPointF(rect.right() - 1, rect.top()),
            )
        painter.end()


class _ComboValue:
    """Small adapter so computation code can keep using source_var.get()."""

    def __init__(self, combo: QtWidgets.QComboBox):
        self.combo = combo

    def get(self):
        return self.combo.currentText()

    def set(self, value):
        idx = self.combo.findText(str(value))
        if idx >= 0:
            self.combo.setCurrentIndex(idx)


class _NoFocusDelegate(QtWidgets.QStyledItemDelegate):
    """Match the BET sample table selection and text rendering."""

    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        option.state &= ~QtWidgets.QStyle.State_HasFocus
        foreground = index.data(QtCore.Qt.ForegroundRole)
        color = foreground.color() if isinstance(foreground, QtGui.QBrush) else QtGui.QColor("#111827")
        option.palette.setColor(QtGui.QPalette.HighlightedText, color)


class _CenteredStatusIconDelegate(QtWidgets.QStyledItemDelegate):
    """Paint status icons as table items so resizing columns updates live."""

    def paint(self, painter, option, index) -> None:
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.state &= ~QtWidgets.QStyle.State_HasFocus
        icon = QtGui.QIcon(opt.icon)
        opt.icon = QtGui.QIcon()
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        if icon.isNull():
            return
        pixmap = icon.pixmap(QtCore.QSize(18, 18))
        rect = QtCore.QRect(
            opt.rect.center().x() - pixmap.width() // 2,
            opt.rect.center().y() - pixmap.height() // 2,
            pixmap.width(),
            pixmap.height(),
        )
        painter.drawPixmap(rect, pixmap)


class _CenteredCompareDotDelegate(QtWidgets.QStyledItemDelegate):
    """BET-style centered blue dot for sample comparison visibility."""

    def paint(self, painter, option, index) -> None:
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.state &= ~QtWidgets.QStyle.State_HasFocus
        style = opt.widget.style() if opt.widget is not None else QtWidgets.QApplication.style()
        style.drawPrimitive(QtWidgets.QStyle.PE_PanelItemViewItem, opt, painter, opt.widget)

        checked = index.data(QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked
        center = QtCore.QPointF(opt.rect.center())
        rect = QtCore.QRectF(center.x() - 5.5, center.y() - 5.5, 11.0, 11.0)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2563eb" if checked else "#6b7280"), 1.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#2563eb" if checked else "#ffffff")))
        painter.drawEllipse(rect)
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if not (index.flags() & QtCore.Qt.ItemIsUserCheckable):
            return False
        if not (index.flags() & QtCore.Qt.ItemIsEnabled):
            return False

        if event.type() in (QtCore.QEvent.MouseButtonRelease, QtCore.QEvent.MouseButtonDblClick):
            if event.button() != QtCore.Qt.LeftButton or not option.rect.contains(event.pos()):
                return False
            if event.type() == QtCore.QEvent.MouseButtonDblClick:
                return True
        elif event.type() == QtCore.QEvent.KeyPress:
            if event.key() not in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Select):
                return False
        else:
            return False

        checked = index.data(QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked
        model.setData(
            index,
            QtCore.Qt.Unchecked if checked else QtCore.Qt.Checked,
            QtCore.Qt.CheckStateRole,
        )
        return True


class _SelectAllCompareCheckBox(QtWidgets.QCheckBox):
    def nextCheckState(self) -> None:
        if self.checkState() == QtCore.Qt.Checked:
            self.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.setCheckState(QtCore.Qt.Checked)


class _TitleActionGroupBox(QtWidgets.QGroupBox):
    """GroupBox with a compact action button aligned to the title row."""

    def __init__(self, title: str, action_text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._action_right_margin = 8
        self.action_button = QtWidgets.QPushButton(action_text, self)
        self.action_button.setFixedSize(72, 22)
        self.action_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.action_button.raise_()

    def set_action_right_margin(self, margin: int) -> None:
        self._action_right_margin = max(8, int(margin))
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.action_button.move(
            max(8, self.width() - self.action_button.width() - self._action_right_margin),
            0,
        )
        self.action_button.raise_()


class _SampleTableWidget(QtWidgets.QTableWidget):
    filesDropped = QtCore.pyqtSignal(list)
    rowHovered = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._hovered_row = -1
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.viewport():
            if event.type() == QtCore.QEvent.MouseMove:
                self._set_hovered_row(self.rowAt(event.pos().y()))
            elif event.type() in (QtCore.QEvent.Leave, QtCore.QEvent.Hide):
                self._set_hovered_row(-1)
        return super().eventFilter(obj, event)

    def _set_hovered_row(self, row: int) -> None:
        row = int(row) if row is not None else -1
        if row < 0 or row >= self.rowCount():
            row = -1
        if row == self._hovered_row:
            return
        self._hovered_row = row
        self.rowHovered.emit(row)

    def dragEnterEvent(self, event) -> None:
        if self._accept_file_drag_event(event):
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._accept_file_drag_event(event):
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = self._paths_from_mime_data(event.mimeData())
        if not paths:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self.filesDropped.emit(paths)

    def _accept_file_drag_event(self, event) -> bool:
        if self._paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return True
        return False

    @staticmethod
    def _paths_from_mime_data(mime_data) -> list[str]:
        if mime_data is None or not mime_data.hasUrls():
            return []
        paths: list[str] = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_DROP_SUFFIXES:
                        paths.append(str(child))
            elif path.is_file() and path.suffix.lower() in SUPPORTED_DROP_SUFFIXES:
                paths.append(str(path))
        return paths


class ControlPanelMixin:
    """Left sample sidebar and analysis controls."""

    def _panel_title(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("font: 700 9pt 'Microsoft YaHei UI'; color: #111827;")
        return label

    def _param_row(self, widget: QtWidgets.QWidget, index: int, *, striped: bool = True) -> QtWidgets.QFrame:
        row = QtWidgets.QFrame()
        row.setMinimumHeight(28)
        if striped:
            row.setObjectName("paramRowEven" if index % 2 == 0 else "paramRowOdd")
            row.setStyleSheet(
                """
                #paramRowEven { background: #ffffff; border: 0; }
                #paramRowOdd { background: #eef1f4; border: 0; }
                """
            )
        else:
            row.setObjectName("paramRowPlain")
            row.setStyleSheet("#paramRowPlain { background: transparent; border: 0; }")
        layout = QtWidgets.QVBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(0)
        layout.addWidget(widget)
        return row

    def _param_hrow(self, index: int) -> tuple[QtWidgets.QFrame, QtWidgets.QHBoxLayout]:
        row = QtWidgets.QFrame()
        row.setFixedHeight(28)
        row.setObjectName("paramRowEven" if index % 2 == 0 else "paramRowOdd")
        row.setStyleSheet(
            """
            #paramRowEven { background: #ffffff; border: 0; }
            #paramRowOdd { background: #eef1f4; border: 0; }
            """
        )
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        return row, layout

    def _build_left_sidebar(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)

        self.btn_import = QtWidgets.QPushButton("导入文件")
        self.btn_import.clicked.connect(self.load_file)
        self.btn_import.setFixedSize(96, 32)

        self.update_available_button = QtWidgets.QToolButton()
        self.update_available_button.setIcon(make_update_available_icon(28))
        self.update_available_button.setIconSize(QtCore.QSize(24, 24))
        self.update_available_button.setFixedSize(32, 32)
        self.update_available_button.setAutoRaise(True)
        self.update_available_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.update_available_button.setToolTip("发现新版本，点击更新")
        self.update_available_button.clicked.connect(self._show_pending_update_dialog)
        self.update_available_button.hide()

        self.update_button = QtWidgets.QPushButton("软件更新")
        self.update_button.setToolTip("联网检查是否有新版")
        self.update_button.clicked.connect(self.check_for_updates)
        self.update_button.setFixedSize(96, 32)

        button_row.addWidget(self.btn_import)
        button_row.addStretch(1)
        button_row.addWidget(self.update_available_button)
        button_row.addWidget(self.update_button)
        parent_layout.addLayout(button_row)

        self.sample_compare_col = 0
        self.sample_file_col = 1
        self.sample_status_col = 2
        self.sample_table = _SampleTableWidget(0, 3)
        self.sample_table.setHorizontalHeaderLabels(["", "文件名", "状态"])
        self.sample_table.verticalHeader().hide()
        self.sample_table.verticalHeader().setDefaultSectionSize(28)
        self.sample_table.setAlternatingRowColors(True)
        self.sample_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sample_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sample_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sample_table.setShowGrid(False)
        self.sample_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.sample_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.sample_table.setMinimumHeight(120)
        self.sample_table.setIconSize(QtCore.QSize(18, 18))
        self.sample_table.setColumnWidth(self.sample_compare_col, 30)
        self.sample_table.setColumnWidth(self.sample_file_col, 170)
        self.sample_table.setColumnWidth(self.sample_status_col, 48)
        self.sample_table.setItemDelegate(_NoFocusDelegate(self.sample_table))
        sample_header = self.sample_table.horizontalHeader()
        sample_header.setStretchLastSection(False)
        sample_header.setSectionsMovable(False)
        sample_header.setHighlightSections(False)
        sample_header.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        sample_header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        sample_header.sectionResized.connect(self._position_compare_select_all_check)
        sample_header.geometriesChanged.connect(self._position_compare_select_all_check)
        self.sample_table.horizontalScrollBar().valueChanged.connect(self._position_compare_select_all_check)
        self.sample_table.filesDropped.connect(self.load_files)
        self.sample_table.setItemDelegateForColumn(
            self.sample_compare_col,
            _CenteredCompareDotDelegate(self.sample_table),
        )
        self.sample_table.setItemDelegateForColumn(
            self.sample_status_col,
            _CenteredStatusIconDelegate(self.sample_table),
        )
        self.sample_table.currentCellChanged.connect(self._on_sample_table_current_changed)
        self.sample_table.itemChanged.connect(self._on_sample_table_item_changed)
        self.sample_table.rowHovered.connect(self._on_sample_table_row_hovered)
        self.sample_table.setStyleSheet(
            """
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                color: #111827;
            }
            QTableWidget::item:selected { background: #e0ecff; color: #111827; }
            QTableWidget::item:focus { outline: none; }
            QHeaderView::section {
                background: #f9fafb;
                border: 0;
                border-right: 1px solid #d1d5db;
                border-bottom: 1px solid #d1d5db;
                color: #374151;
                font-weight: 600;
                padding: 4px 8px 4px 6px;
            }
            QTableWidget::indicator {
                width: 11px;
                height: 11px;
                border-radius: 6px;
                border: 1px solid #6b7280;
                background: white;
            }
            QTableWidget::indicator:checked {
                border: 1px solid #2563eb;
                background: #2563eb;
            }
            """
        )
        self.select_all_compare_check = _SelectAllCompareCheckBox(sample_header)
        self.select_all_compare_check.setTristate(True)
        self.select_all_compare_check.setFocusPolicy(QtCore.Qt.NoFocus)
        self.select_all_compare_check.setFixedSize(16, 16)
        self.select_all_compare_check.setCursor(QtCore.Qt.PointingHandCursor)
        self.select_all_compare_check.setToolTip("在对比分析中显示或隐藏全部样品")
        self.select_all_compare_check.stateChanged.connect(self._on_compare_select_all_changed)
        self.select_all_compare_check.setStyleSheet(
            """
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border-radius: 7px;
                border: 1px solid #6b7280;
                background: white;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2563eb;
                background: #2563eb;
            }
            QCheckBox::indicator:indeterminate {
                border: 1px solid #2563eb;
                background: #93c5fd;
            }
            """
        )
        self.select_all_compare_check.show()
        parent_layout.addWidget(self.sample_table, 1)
        QtCore.QTimer.singleShot(0, self._position_compare_select_all_check)

        self.sample_detail_tabs = QtWidgets.QTabWidget()
        self.sample_detail_tabs.setDocumentMode(False)
        self.sample_detail_tabs.tabBar().setDrawBase(False)
        self.sample_param_table = self._make_detail_table(["参数", "值"])
        self.sample_param_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_param_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.sample_result_table = self._make_detail_table(["Peak", "峰面积", "峰高", "尺寸", "面积百分比", "高度百分比"])
        self.sample_result_table.horizontalHeader().setStretchLastSection(False)
        self.sample_result_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_result_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_result_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_result_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_result_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_result_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_stats_table = self._make_detail_table(["粒径区间", "百分比"])
        self.sample_stats_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.sample_stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.sample_stats_table.horizontalHeader().setStretchLastSection(True)
        self.sample_stats_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.sample_stats_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self._install_table_copy_menu(self.sample_stats_table)
        self.sample_detail_tabs.addTab(self.sample_param_table, "样品")
        self.sample_detail_tabs.addTab(self.sample_result_table, "计算结果")
        self.sample_detail_tabs.addTab(self.sample_stats_table, "统计结果")
        parent_layout.addWidget(self.sample_detail_tabs, 2)
        self.update_statistics_table()

    @staticmethod
    def _make_detail_table(headers: list[str]) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(
            """
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                color: #111827;
            }
            QTableWidget::item:selected { background: #e0ecff; color: #111827; }
            QTableWidget::item:focus { outline: none; }
            QHeaderView::section {
                background: #f9fafb;
                border: 0;
                border-right: 1px solid #d1d5db;
                border-bottom: 1px solid #d1d5db;
                color: #374151;
                font-weight: 600;
                padding: 4px 8px 4px 6px;
            }
            """
        )
        return table

    @staticmethod
    def _detail_table_item(text: str, *, alignment=None) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(str(text))
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        item.setForeground(QtGui.QBrush(QtGui.QColor("#111827")))
        if alignment is not None:
            item.setTextAlignment(alignment)
        return item

    def _install_table_copy_menu(self, table: QtWidgets.QTableWidget) -> None:
        table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        copy_action = QtWidgets.QAction("复制", table)
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        copy_action.triggered.connect(lambda _checked=False, t=table: self._copy_table_selection_to_clipboard(t))
        table.addAction(copy_action)
        table.customContextMenuRequested.connect(
            lambda pos, t=table: self._show_table_copy_menu(t, pos)
        )

    def _show_table_copy_menu(self, table: QtWidgets.QTableWidget, pos: QtCore.QPoint) -> None:
        if not table.selectedIndexes():
            item = table.itemAt(pos)
            if item is not None:
                table.setCurrentItem(item)
                item.setSelected(True)
        if not table.selectedIndexes():
            return
        menu = QtWidgets.QMenu(table)
        copy_action = menu.addAction("复制")
        copy_action.triggered.connect(lambda _checked=False, t=table: self._copy_table_selection_to_clipboard(t))
        menu.exec_(table.viewport().mapToGlobal(pos))

    def _copy_table_selection_to_clipboard(self, table: QtWidgets.QTableWidget) -> None:
        indexes = table.selectedIndexes()
        if not indexes:
            return
        rows = sorted({idx.row() for idx in indexes})
        cols = sorted({idx.column() for idx in indexes})
        selected = {(idx.row(), idx.column()) for idx in indexes}
        lines = []
        for row in rows:
            values = []
            for col in cols:
                item = table.item(row, col)
                values.append(item.text() if item is not None and (row, col) in selected else "")
            lines.append("\t".join(values))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def _build_analysis_controls(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame(parent)
        panel.setObjectName("analysisControls")
        panel.setMinimumWidth(198)
        panel.setMaximumWidth(300)
        panel.resize(224, panel.height())
        panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        panel.setStyleSheet(
            """
            #analysisControls { background: #f3f4f5; border: 1px solid #d1d5db; }
            QPushButton {
                background: #f8f9fa;
                color: #2c3135;
                font: 8pt 'Microsoft YaHei';
                border: 1px solid #c9ced3;
                border-radius: 3px;
                padding: 2px 6px;
                min-height: 22px;
            }
            QPushButton:hover { background: #eef1f3; border-color: #aeb6bd; }
            QPushButton:checked {
                background: #e0ecff;
                border-color: #2563eb;
                color: #1d4ed8;
            }
            QCheckBox { color: #111827; spacing: 4px; font: 9pt 'Microsoft YaHei UI'; }
            QComboBox {
                min-height: 22px;
                padding: 1px 6px;
                border: 1px solid #c6cbd0;
                border-radius: 3px;
                background: white;
                color: #111827;
                font: 9pt 'Microsoft YaHei UI';
            }
            QLabel { color: #111827; font: 9pt 'Microsoft YaHei UI'; }
            QGroupBox {
                border: 1px solid #d1d5db;
                border-radius: 0;
                margin-top: 10px;
                background: #f3f4f5;
                color: #111827;
                font: 9pt 'Microsoft YaHei UI';
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 3px;
                background: #f3f4f5;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        row_index = 0
        self.slider_min = QtLabeledSpin("起始角", 0, 100, 60, decimals=2, step=0.01, parent=panel)
        self.slider_min.valueChanged.connect(lambda v: self._on_angle_slider_changed("min", v))
        self.slider_min.hide()

        self.slider_max = QtLabeledSpin("结束角", 0, 100, 74.6, decimals=2, step=0.01, parent=panel)
        self.slider_max.valueChanged.connect(lambda v: self._on_angle_slider_changed("max", v))
        self.slider_max.hide()

        peak_group = QtWidgets.QGroupBox("晶面选择", panel)
        peak_layout = QtWidgets.QVBoxLayout(peak_group)
        peak_layout.setContentsMargins(8, 14, 8, 8)
        peak_layout.setSpacing(4)

        peak_actions = QtWidgets.QHBoxLayout()
        peak_actions.setContentsMargins(0, 0, 0, 0)
        peak_actions.setSpacing(4)
        peak_actions.addStretch(1)
        self.peaks_detail_button = QtWidgets.QPushButton("详情", peak_group)
        self.peaks_detail_button.setCheckable(True)
        self.peaks_detail_button.setFixedSize(48, 22)
        self.peaks_detail_button.setToolTip("展开或收起默认峰列表")
        add_peak_button = QtWidgets.QPushButton("添加", peak_group)
        add_peak_button.setFixedSize(56, 22)
        peak_button_style = (
            """
            QPushButton {
                background: #f8f9fa;
                color: #2c3135;
                font: 9pt 'Microsoft YaHei UI';
                border: 1px solid #c9ced3;
                border-radius: 3px;
                padding: 0 6px;
                min-height: 0;
            }
            QPushButton:hover {
                background: #eef1f3;
                border-color: #aeb6bd;
            }
            QPushButton:pressed {
                background: #e5e9ed;
            }
            """
        )
        self.peaks_detail_button.setStyleSheet(peak_button_style)
        add_peak_button.setStyleSheet(peak_button_style)
        peak_actions.addWidget(self.peaks_detail_button)
        peak_actions.addWidget(add_peak_button)
        peak_layout.addLayout(peak_actions)
        self.peaks_detail_button.toggled.connect(self._set_peak_details_visible)
        add_peak_button.clicked.connect(self.add_peak_control)
        row_index += 1

        self.peaks_frame = QtWidgets.QWidget(panel)
        self.peaks_layout = QtWidgets.QVBoxLayout(self.peaks_frame)
        self.peaks_layout.setContentsMargins(0, 0, 0, 0)
        self.peaks_layout.setSpacing(2)
        self.peaks_scroll = QtWidgets.QScrollArea(panel)
        self.peaks_scroll.setWidgetResizable(False)
        self.peaks_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.peaks_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.peaks_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.peaks_scroll.setMaximumHeight(162)
        self.peaks_scroll.setWidget(self.peaks_frame)
        self.peaks_scroll.hide()
        peak_layout.addWidget(self.peaks_scroll)
        layout.addWidget(peak_group)

        if not self.peak_check_vars:
            self._building_peak_controls = True
            for i in range(self.max_peaks):
                self._create_peak_control(i, checked=(i == 0))
            self._building_peak_controls = False
            self.update_ui_for_peaks()

        self.slider_alpha = QtSpinSlider("平滑因子 (α)", 0.01, 100, 1, 0.01, decimals=2, parent=panel)
        self.slider_alpha.valueChanged.connect(self._on_alpha_value_changed)
        self.slider_alpha.slider.sliderReleased.connect(lambda: self._alpha_fast_timer.start(1))
        layout.addWidget(self._param_row(self.slider_alpha, row_index))
        row_index += 1

        source_frame, source_row = self._param_hrow(row_index)
        source_row.addWidget(QtWidgets.QLabel("X射线源"))
        self.source_menu = QtWidgets.QComboBox()
        self.source_menu.addItems(["Cu", "Co", "Fe", "Mo"])
        self.source_menu.setCurrentText("Cu")
        source_row.addWidget(self.source_menu, 1)
        layout.addWidget(source_frame)
        self.source_var = _ComboValue(self.source_menu)

        action_grid = QtWidgets.QGridLayout()
        action_grid.setContentsMargins(0, 2, 0, 0)
        action_grid.setHorizontalSpacing(4)
        action_grid.setVerticalSpacing(4)
        self.btn_fast = QtWidgets.QPushButton("极速计算")
        self.btn_fast.clicked.connect(lambda: self.compute_thread("fast"))
        self.btn_fast.setStyleSheet(
            """
            QPushButton {
                background: #22c55e;
                color: #ffffff;
                font: 700 8pt 'Microsoft YaHei';
                border: 1px solid #16a34a;
                border-radius: 3px;
                padding: 2px 6px;
                min-height: 22px;
            }
            QPushButton:hover { background: #16a34a; border-color: #15803d; }
            QPushButton:pressed { background: #15803d; }
            QPushButton:disabled { background: #a7d9b8; border-color: #9acda9; color: #f8fafc; }
            """
        )
        self.btn_fine = QtWidgets.QPushButton("精细计算")
        self.btn_fine.clicked.connect(lambda: self.compute_thread("fine"))
        self.btn_fine.setStyleSheet(
            """
            QPushButton {
                background: #f97316;
                color: #ffffff;
                font: 700 8pt 'Microsoft YaHei';
                border: 1px solid #ea580c;
                border-radius: 3px;
                padding: 2px 6px;
                min-height: 22px;
            }
            QPushButton:hover { background: #ea580c; border-color: #c2410c; }
            QPushButton:pressed { background: #c2410c; }
            QPushButton:disabled { background: #f2b183; border-color: #e8a170; color: #fff7ed; }
            """
        )
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_compute)
        self.btn_advanced = QtWidgets.QPushButton("高级")
        self.btn_advanced.clicked.connect(self._show_advanced_dialog)
        self.btn_lcurve = self.btn_advanced
        self.btn_manual_baseline = QtWidgets.QPushButton("手动基线")
        self.btn_manual_baseline.setCheckable(True)
        self.btn_manual_baseline.setToolTip("在拟合图中显示并编辑计算基线")
        self.btn_manual_baseline.toggled.connect(self._on_manual_baseline_toggled)
        self.btn_save = QtWidgets.QPushButton("保存")
        self.btn_save.clicked.connect(self.save_results)
        action_grid.addWidget(self.btn_advanced, 0, 0)
        action_grid.addWidget(self.btn_manual_baseline, 0, 1)
        action_grid.addWidget(self.btn_save, 1, 0)
        action_grid.addWidget(self.btn_stop, 1, 1)
        action_grid.addWidget(self.btn_fast, 2, 0)
        action_grid.addWidget(self.btn_fine, 2, 1)
        layout.addLayout(action_grid)
        layout.addStretch(1)
        return panel

    def _show_advanced_dialog(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("高级")
        dialog.setModal(True)
        dialog.setMinimumWidth(300)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        d_min_spin = QtWidgets.QDoubleSpinBox(dialog)
        d_min_spin.setRange(0.01, 1000.0)
        d_min_spin.setDecimals(2)
        d_min_spin.setSingleStep(0.1)
        d_min_spin.setFixedWidth(108)
        d_min_spin.setValue(float(getattr(self, "particle_size_min", 0.1)))

        d_max_spin = QtWidgets.QDoubleSpinBox(dialog)
        d_max_spin.setRange(0.02, 1000.0)
        d_max_spin.setDecimals(2)
        d_max_spin.setSingleStep(1.0)
        d_max_spin.setFixedWidth(108)
        d_max_spin.setValue(float(getattr(self, "particle_size_max", 100.0)))

        def unit_field(spin: QtWidgets.QDoubleSpinBox, unit: str) -> QtWidgets.QWidget:
            field = QtWidgets.QWidget(dialog)
            row = QtWidgets.QHBoxLayout(field)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            unit_label = QtWidgets.QLabel(unit, field)
            unit_label.setStyleSheet("font: 8pt 'Microsoft YaHei'; color: #4b5563;")
            row.addWidget(spin)
            row.addWidget(unit_label)
            row.addStretch(1)
            return field

        d_min_field = unit_field(d_min_spin, "nm")
        d_max_field = unit_field(d_max_spin, "nm")

        d_step_combo = QtWidgets.QComboBox(dialog)
        d_step_combo.addItem("0.10 nm（标准）", 0.10)
        d_step_combo.addItem("0.05 nm（精细）", 0.05)
        d_step_combo.addItem("0.02 nm（高精度）", 0.02)
        d_step_combo.addItem("0.01 nm（超高精度）", 0.01)
        current_step = float(getattr(self, "particle_size_step", 0.1) or 0.1)
        step_index = min(
            range(d_step_combo.count()),
            key=lambda i: abs(float(d_step_combo.itemData(i)) - current_step),
        )
        d_step_combo.setCurrentIndex(step_index)

        fwhm_spin = QtWidgets.QDoubleSpinBox(dialog)
        fwhm_spin.setRange(0.0, 5.0)
        fwhm_spin.setDecimals(4)
        fwhm_spin.setSingleStep(0.001)
        fwhm_spin.setSuffix(" °")
        fwhm_spin.setValue(float(getattr(self, "instrument_fwhm", 0.0)))

        algorithm_combo = QtWidgets.QComboBox(dialog)
        algorithm_combo.addItem("平滑 L2（当前默认）", "l2")
        algorithm_combo.addItem("TV 高分辨（实验）", "tv")
        algorithm_combo.addItem("L2+TV 混合（实验）", "hybrid")
        algorithm_combo.addItem("DL 超分辨（实验）", "dl_sr")
        current_algorithm = str(getattr(self, "regularization_method", "l2") or "l2").lower()
        index = algorithm_combo.findData(current_algorithm)
        algorithm_combo.setCurrentIndex(index if index >= 0 else 0)

        for spin in (d_min_spin, d_max_spin, fwhm_spin):
            spin.setKeyboardTracking(False)
            spin.setLocale(QtCore.QLocale.c())
            spin.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
            spin.lineEdit().editingFinished.connect(spin.interpretText)
            spin.setStyleSheet(
                "QDoubleSpinBox { min-height: 22px; padding: 1px 6px;"
                " border: 1px solid #c6cbd0; border-radius: 3px;"
                " background: white; font: 8pt 'Microsoft YaHei'; }"
            )
        algorithm_combo.setStyleSheet(
            "QComboBox { min-height: 22px; padding: 1px 6px;"
            " border: 1px solid #c6cbd0; border-radius: 3px;"
            " background: white; font: 8pt 'Microsoft YaHei'; }"
        )
        d_step_combo.setStyleSheet(algorithm_combo.styleSheet())

        form.addRow("正则算法", algorithm_combo)
        form.addRow("最小粒径", d_min_field)
        form.addRow("最大粒径", d_max_field)
        form.addRow("粒径步长", d_step_combo)
        form.addRow("仪器展宽矫正", fwhm_spin)
        layout.addLayout(form)

        lcurve_button = QtWidgets.QPushButton("运行 L-Curve")
        lcurve_button.setStyleSheet(
            "QPushButton { background: #f8f9fa; color: #2c3135; font: 8pt 'Microsoft YaHei';"
            " border: 1px solid #c9ced3; border-radius: 3px; padding: 3px 8px; min-height: 22px; }"
            "QPushButton:hover { background: #eef1f3; border-color: #aeb6bd; }"
        )
        layout.addWidget(lcurve_button)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("确定")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("取消")
        layout.addWidget(buttons)

        def commit_spinbox_edits() -> None:
            for spin in (d_min_spin, d_max_spin, fwhm_spin):
                spin.interpretText()

        def apply_values() -> bool:
            commit_spinbox_edits()
            d_min = float(d_min_spin.value())
            d_max = float(d_max_spin.value())
            if d_max <= d_min:
                QtWidgets.QMessageBox.warning(dialog, "提示", "最大粒径必须大于最小粒径。")
                return False
            self.particle_size_min = d_min
            self.particle_size_max = d_max
            self.particle_size_step = float(d_step_combo.currentData() or 0.1)
            self.instrument_fwhm = float(fwhm_spin.value())
            self.regularization_method = str(algorithm_combo.currentData() or "l2")
            return True

        def accept_dialog() -> None:
            if apply_values():
                dialog.accept()

        def run_lcurve() -> None:
            if apply_values():
                dialog.accept()
                self.run_l_curve_thread()

        buttons.accepted.connect(accept_dialog)
        buttons.rejected.connect(dialog.reject)
        lcurve_button.clicked.connect(run_lcurve)
        dialog.exec_()

    def _create_peak_control(
        self,
        peak_idx: int,
        checked: bool = False,
        value: float | None = None,
        color: str | None = None,
        visible: bool = True,
    ) -> None:
        row, row_layout = self._param_hrow(peak_idx + 3)
        row.setToolTip("右键删除这个峰")
        peak_color = self._set_peak_color_value(peak_idx, color or self._peak_color(peak_idx))

        chk = QtWidgets.QCheckBox()
        chk.setChecked(checked)
        chk.stateChanged.connect(lambda _state: self.update_ui_for_peaks())
        row_layout.addWidget(chk)

        peak_min, peak_max = self._peak_value_bounds()
        spin = QtLabeledSpin(
            f"Peak{peak_idx + 1}",
            peak_min,
            peak_max,
            60 + peak_idx if value is None else value,
            decimals=2,
            step=0.01,
            parent=row,
        )
        spin.label.setMinimumWidth(52)
        spin.label.setMaximumWidth(56)
        spin.spin.setFixedWidth(82)

        color_button = QtWidgets.QToolButton(row)
        color_button.setCursor(QtCore.Qt.PointingHandCursor)
        color_button.setToolTip("选择峰线颜色")
        color_button.setFixedSize(16, 16)
        self._set_peak_color_button_style(color_button, peak_color)
        color_button.clicked.connect(lambda _checked=False, idx=peak_idx: self._choose_peak_color(idx))
        spin.layout().insertWidget(1, color_button)

        spin.valueChanged.connect(lambda _v: self._on_peak_value_changed())
        row_layout.addWidget(spin, 1)

        self._attach_peak_context_menu(row, row)
        self._attach_peak_context_menu(chk, row)
        self._attach_peak_context_menu(spin.label, row)
        self._attach_peak_context_menu(color_button, row)
        self.peaks_layout.addWidget(row)

        if not hasattr(self, "peak_rows"):
            self.peak_rows = []
        if not hasattr(self, "peak_color_buttons"):
            self.peak_color_buttons = []
        if not hasattr(self, "peak_visible_buttons"):
            self.peak_visible_buttons = []
        if not hasattr(self, "peak_visible_states"):
            self.peak_visible_states = []
        self.peak_rows.append(row)
        self.peak_check_vars.append(chk)
        self.peak_mu_sliders.append(spin)
        self.peak_color_buttons.append(color_button)
        self.peak_visible_states.append(bool(visible))
        self._update_peak_scroll_extent()
        if not getattr(self, "_building_peak_controls", False):
            self.update_ui_for_peaks()

    def _update_peak_scroll_extent(self) -> None:
        if not hasattr(self, "peaks_scroll"):
            return
        width = max(198, self.peaks_scroll.viewport().width())
        row_count = len(getattr(self, "peak_rows", []))
        spacing = self.peaks_layout.spacing()
        height = max(1, row_count * 28 + max(0, row_count - 1) * spacing)
        self.peaks_frame.setFixedWidth(width)
        self.peaks_frame.setMinimumHeight(height)
        self.peaks_frame.resize(width, height)

    def _set_peak_details_visible(self, visible: bool) -> None:
        if not hasattr(self, "peaks_scroll"):
            return
        self.peaks_scroll.setVisible(bool(visible))
        button = getattr(self, "peaks_detail_button", None)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(bool(visible))
            button.setText("收起" if visible else "详情")
            button.blockSignals(False)
        self._update_peak_scroll_extent()
        pane = getattr(self, "analysis_options_pane", None)
        if pane is not None and hasattr(pane, "_sync_content_height"):
            pane._sync_content_height()

    def add_peak_control(self):
        self.max_peaks = len(self.peak_mu_sliders) + 1
        region = getattr(self, "preview_range_region", None)
        if region is not None:
            try:
                angle_min, angle_max = sorted(float(value) for value in region.getRegion())
            except Exception:
                angle_min, angle_max = self._normalize_angle_range()
        else:
            angle_min, angle_max = self._normalize_angle_range()
        center = (float(angle_min) + float(angle_max)) / 2.0
        new_index = self.max_peaks - 1
        self._create_peak_control(
            new_index,
            checked=True,
            value=center,
            color=self._default_peak_color(new_index),
        )
        self._set_peak_details_visible(True)

    def _attach_peak_context_menu(self, widget: QtWidgets.QWidget, row: QtWidgets.QWidget) -> None:
        widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos, source=widget, peak_row=row: self._show_peak_context_menu(
                peak_row,
                source.mapToGlobal(pos),
            )
        )

    def _show_peak_context_menu(self, row: QtWidgets.QWidget, global_pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(row)
        delete_action = menu.addAction("删除")
        if menu.exec_(global_pos) == delete_action:
            self._delete_peak_row(row)

    def _delete_peak_row(self, row: QtWidgets.QWidget) -> None:
        if not hasattr(self, "peak_rows") or row not in self.peak_rows:
            return
        index = self.peak_rows.index(row)
        if hasattr(self, "_clear_peak_coordinate_artifacts"):
            self._clear_peak_coordinate_artifacts(index)
        states = self._current_peak_states()
        del states[index]
        for i, state in enumerate(states):
            state["color"] = self._default_peak_color(i)
        self._apply_peak_states(states)
        self._save_current_peak_states()
        self.update_preview()

    def _delete_peak_index(self, peak_idx: int) -> None:
        rows = getattr(self, "peak_rows", [])
        try:
            index = int(peak_idx)
        except Exception:
            return
        if 0 <= index < len(rows):
            self._delete_peak_row(rows[index])

    @staticmethod
    def _default_peak_color(peak_idx: int) -> str:
        palette = [
            "#FF0000", "#0077FF", "#00C853", "#FFAB00", "#00E5FF",
            "#8E24AA", "#D81B60", "#43A047", "#F4511E", "#3949AB",
        ]
        return palette[int(peak_idx) % len(palette)]

    def _peak_color(self, peak_idx: int) -> str:
        while peak_idx >= len(self.peak_colors):
            self.peak_colors.append(self._default_peak_color(len(self.peak_colors)))
        return self.peak_colors[peak_idx]

    def _set_peak_color_value(self, peak_idx: int, color: str | None) -> str:
        self._peak_color(peak_idx)
        qcolor = QtGui.QColor(str(color or self.peak_colors[peak_idx]))
        if not qcolor.isValid():
            qcolor = QtGui.QColor(self.peak_colors[peak_idx])
        self.peak_colors[peak_idx] = qcolor.name().upper()
        return self.peak_colors[peak_idx]

    @staticmethod
    def _set_peak_color_button_style(button: QtWidgets.QToolButton, color: str) -> None:
        button.setStyleSheet(
            f"""
            QToolButton {{
                background: {color};
                border: 1px solid #6b7280;
                border-radius: 2px;
                padding: 0;
            }}
            QToolButton:hover {{ border-color: #111827; }}
            """
        )

    def _choose_peak_color(self, peak_idx: int) -> None:
        current = QtGui.QColor(self._peak_color(peak_idx))
        color = QtWidgets.QColorDialog.getColor(current, self, "选择峰线颜色")
        if not color.isValid():
            return
        new_color = self._set_peak_color_value(peak_idx, color.name())
        if peak_idx < len(getattr(self, "peak_color_buttons", [])):
            self._set_peak_color_button_style(self.peak_color_buttons[peak_idx], new_color)
        if not getattr(self, "_building_peak_controls", False):
            self._save_current_peak_states()
        self._refresh_plots_after_peak_display_change()

    def _peak_visible(self, peak_idx: int) -> bool:
        states = getattr(self, "peak_visible_states", [])
        if 0 <= peak_idx < len(states):
            return bool(states[peak_idx])
        buttons = getattr(self, "peak_visible_buttons", [])
        if 0 <= peak_idx < len(buttons):
            return bool(buttons[peak_idx].isChecked())
        return True

    def _set_peak_visible(self, peak_idx: int, visible: bool) -> None:
        if not hasattr(self, "peak_visible_states"):
            self.peak_visible_states = []
        while peak_idx >= len(self.peak_visible_states):
            self.peak_visible_states.append(True)
        self.peak_visible_states[peak_idx] = bool(visible)
        buttons = getattr(self, "peak_visible_buttons", [])
        if 0 <= peak_idx < len(buttons):
            button = buttons[peak_idx]
            if button is not None:
                button.blockSignals(True)
                button.setChecked(bool(visible))
                button.blockSignals(False)

    def _on_peak_visible_changed(self, _peak_idx: int) -> None:
        if not getattr(self, "_building_peak_controls", False):
            self._save_current_peak_states()
        self._refresh_plots_after_peak_display_change()

    def _refresh_plots_after_peak_display_change(self) -> None:
        if not getattr(self, "data_loaded", False):
            return
        if hasattr(self, "_save_current_marker_label_state"):
            self._save_current_marker_label_state()
        if hasattr(self, "_save_current_plot_view_state"):
            self._save_current_plot_view_state()
        views = self._capture_plot_views() if hasattr(self, "_capture_plot_views") else {}
        self.update_preview()
        if getattr(self, "results_ready", False):
            self.update_multi_peak_plots()
        if views and hasattr(self, "_restore_plot_views"):
            self._restore_plot_views(views)

    def update_ui_for_peaks(self):
        self.active_peak_indices = [
            i for i, chk in enumerate(self.peak_check_vars) if chk.isChecked()
        ]
        self._refresh_peak_slider_bounds()
        if not getattr(self, "_building_peak_controls", False):
            self._save_current_peak_states()
        self.update_preview()

    def _on_peak_value_changed(self):
        if not getattr(self, "_building_peak_controls", False):
            self._save_current_peak_states()
        self.update_preview()

    def _default_peak_states(self, count: int = 1) -> list[dict[str, float | bool | str]]:
        return [
            {
                "checked": True,
                "value": float(60 + i),
                "color": self._default_peak_color(i),
                "visible": True,
            }
            for i in range(int(count))
        ]

    def _current_peak_states(self) -> list[dict[str, float | bool | str]]:
        states = []
        for i, (chk, spin) in enumerate(zip(self.peak_check_vars, self.peak_mu_sliders)):
            states.append(
                {
                    "checked": bool(chk.isChecked()),
                    "value": float(spin.get()),
                    "color": self._peak_color(i),
                    "visible": self._peak_visible(i),
                }
            )
        return states

    def _save_current_peak_states(self) -> None:
        if getattr(self, "_building_peak_controls", False):
            return
        index = getattr(self, "active_sample_index", -1)
        samples = getattr(self, "samples", [])
        if 0 <= index < len(samples):
            samples[index].peak_states = self._current_peak_states()

    def _current_analysis_state(self) -> dict[str, float]:
        if not hasattr(self, "slider_min") or not hasattr(self, "slider_max"):
            return {}
        return {
            "angle_min": float(self.slider_min.get()),
            "angle_max": float(self.slider_max.get()),
        }

    def _save_current_analysis_state(self) -> None:
        index = getattr(self, "active_sample_index", -1)
        samples = getattr(self, "samples", [])
        if 0 <= index < len(samples):
            samples[index].analysis_state = self._current_analysis_state()

    def _apply_analysis_range_to_all_samples(self) -> None:
        samples = getattr(self, "samples", [])
        if not samples or not hasattr(self, "slider_min") or not hasattr(self, "slider_max"):
            return
        angle_min, angle_max = self._normalize_angle_range()
        state = {"angle_min": float(angle_min), "angle_max": float(angle_max)}
        for sample in samples:
            sample.analysis_state = dict(state)
        self._save_current_analysis_state()
        self.statusBar().showMessage(
            f"已将分析区间 {angle_min:.2f} - {angle_max:.2f} 应用到全部样品",
            4000,
        )

    def _peak_value_bounds(self) -> tuple[float, float]:
        if getattr(self, "data_loaded", False) and hasattr(self, "x_data"):
            try:
                x = np.asarray(self.x_data, dtype=float)
                finite = x[np.isfinite(x)]
                if finite.size:
                    return float(np.nanmin(finite)), float(np.nanmax(finite))
            except Exception:
                pass
        if hasattr(self, "slider_min") and hasattr(self, "slider_max"):
            return float(self.slider_min.spin.minimum()), float(self.slider_max.spin.maximum())
        return 0.0, 100.0

    def _selected_peak_indices_in_fit_range(
        self,
        angle_min: float | None = None,
        angle_max: float | None = None,
    ) -> list[int]:
        if angle_min is None or angle_max is None:
            angle_min, angle_max = self._normalize_angle_range()
        low, high = sorted((float(angle_min), float(angle_max)))
        eps = 1e-9
        selected = []
        for peak_idx in getattr(self, "active_peak_indices", []):
            if peak_idx >= len(getattr(self, "peak_mu_sliders", [])):
                continue
            value = float(self.peak_mu_sliders[peak_idx].get())
            if low - eps <= value <= high + eps:
                selected.append(peak_idx)
        return selected

    def _apply_analysis_state(self, state: dict | None) -> None:
        if not state or not hasattr(self, "slider_min") or not hasattr(self, "slider_max"):
            return
        angle_min = float(state.get("angle_min", self.slider_min.get()))
        angle_max = float(state.get("angle_max", self.slider_max.get()))
        self.slider_min.set(angle_min, emit=False)
        self.slider_max.set(angle_max, emit=False)
        self._normalize_angle_range()

    def _apply_peak_states(self, states: list[dict[str, float | bool | str]]) -> None:
        states = self._default_peak_states() if states is None else list(states)
        self._building_peak_controls = True
        try:
            for i, state in enumerate(states):
                self._set_peak_color_value(i, str(state.get("color", self._default_peak_color(i))))
            while self.peaks_layout.count():
                item = self.peaks_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
            self.peak_rows = []
            self.peak_check_vars = []
            self.peak_mu_sliders = []
            self.peak_color_buttons = []
            self.peak_visible_buttons = []
            self.peak_visible_states = []
            self.max_peaks = len(states)
            for i, state in enumerate(states):
                self._create_peak_control(
                    i,
                    checked=bool(state.get("checked", i == 0)),
                    value=float(state.get("value", 60 + i)),
                    color=str(state.get("color", self._default_peak_color(i))),
                    visible=bool(state.get("visible", True)),
                )
        finally:
            self._building_peak_controls = False
        self._update_peak_scroll_extent()
        self.update_ui_for_peaks()

    def _refresh_peak_slider_bounds(self):
        low, high = self._peak_value_bounds()
        for s in self.peak_mu_sliders:
            s.config(from_=low, to=high)

    def _on_angle_slider_changed(self, changed: str, _value=None):
        self._normalize_angle_range(changed)
        self._refresh_peak_slider_bounds()
        self._save_current_analysis_state()
        self.update_preview()

    def _normalize_angle_range(self, changed: str | None = None):
        angle_min = self.slider_min.get()
        angle_max = self.slider_max.get()
        eps = 0.01
        low_limit = self.slider_min.spin.minimum()
        high_limit = self.slider_max.spin.maximum()
        if angle_min + eps <= angle_max:
            return angle_min, angle_max
        if changed == "max":
            new_max = max(low_limit + eps, min(high_limit, angle_max))
            new_min = max(low_limit, new_max - eps)
        else:
            new_min = min(high_limit - eps, max(low_limit, angle_min))
            new_max = min(high_limit, new_min + eps)
        self.slider_min.set(new_min, emit=False)
        self.slider_max.set(new_max, emit=False)
        return new_min, new_max

    def update_info_panel(self, metadata: dict):
        rows = self._metadata_rows(metadata)
        self.sample_param_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            key_item = self._detail_table_item(key)
            key_item.setForeground(QtGui.QBrush(QtGui.QColor("#4b5563")))
            value_item = self._detail_table_item(value if value else "—")
            self.sample_param_table.setItem(row, 0, key_item)
            self.sample_param_table.setItem(row, 1, value_item)

    def clear_result_table(self) -> None:
        if hasattr(self, "sample_result_table"):
            self.sample_result_table.setRowCount(0)
        self.update_statistics_table()

    def update_result_table(self) -> None:
        if not hasattr(self, "sample_result_table"):
            return
        rows = self._result_rows()
        self.sample_result_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                alignment = None
                if col in {1, 2, 4, 5}:
                    alignment = QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                elif col == 3:
                    alignment = QtCore.Qt.AlignCenter
                item = self._detail_table_item(value, alignment=alignment)
                if "-" not in values[0]:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.sample_result_table.setItem(row, col, item)
        self.update_statistics_table()

    def update_statistics_table(self) -> None:
        if not hasattr(self, "sample_stats_table"):
            return
        rows = self._statistics_rows()
        self.sample_stats_table.setRowCount(len(rows))
        for row, (interval, percentage) in enumerate(rows):
            self.sample_stats_table.setItem(row, 0, self._detail_table_item(interval))
            self.sample_stats_table.setItem(
                row,
                1,
                self._detail_table_item(percentage, alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter),
            )

    def _statistics_rows(self) -> list[tuple[str, str]]:
        intervals = self._particle_size_intervals()
        percentages = self._particle_size_interval_percentages()
        if not percentages:
            return [(label, "") for label, _left, _right in intervals]
        return [(label, percentages.get(label, "0.00%")) for label, _left, _right in intervals]

    @staticmethod
    def _particle_size_intervals() -> list[tuple[str, float, float]]:
        intervals = [("＜1", 0.0, 1.0)]
        intervals.extend((f"{i}-{i + 1}", float(i), float(i + 1)) for i in range(1, 100))
        return intervals

    def _particle_size_interval_percentages(self) -> dict[str, str]:
        if not getattr(self, "results_ready", False):
            return {}
        D_range = np.asarray(getattr(self, "D_range", []), dtype=float)
        peak_infos = list(getattr(self, "all_peak_info", []) or [])
        if D_range.size < 2 or not peak_infos:
            return {}
        global_y = np.zeros_like(D_range, dtype=float)
        for info in peak_infos:
            f_segment = np.asarray(info.get("f_segment", []), dtype=float)
            if f_segment.size == D_range.size:
                global_y += np.nan_to_num(f_segment, nan=0.0, posinf=0.0, neginf=0.0)
        global_y = np.clip(global_y, 0.0, None)
        if not np.any(global_y > 0):
            return {}
        denominator = self._integrate_distribution_interval(
            D_range,
            global_y,
            float(np.nanmin(D_range)),
            float(np.nanmax(D_range)),
        )
        if denominator <= 0:
            denominator = float(np.sum(global_y))
        if denominator <= 0:
            return {}
        percentages: dict[str, str] = {}
        for label, left, right in self._particle_size_intervals():
            area = self._integrate_distribution_interval(D_range, global_y, left, right)
            pct = max(0.0, area / denominator * 100.0)
            percentages[label] = f"{pct:.2f}%"
        return percentages

    @staticmethod
    def _integrate_distribution_interval(x_values, y_values, left: float, right: float) -> float:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2 or right <= left:
            return 0.0
        order = np.argsort(x)
        x = x[order]
        y = np.clip(y[order], 0.0, None)
        lo = max(float(left), float(x[0]))
        hi = min(float(right), float(x[-1]))
        if hi <= lo:
            return 0.0
        inner = (x > lo) & (x < hi)
        xs = np.concatenate(([lo], x[inner], [hi]))
        ys = np.interp(xs, x, y)
        try:
            return float(np.trapezoid(ys, xs))
        except AttributeError:
            return float(np.trapz(ys, xs))

    def _result_rows(self) -> list[tuple[str, str, str, str, str, str]]:
        if not getattr(self, "results_ready", False):
            return []
        all_peak_info = list(getattr(self, "all_peak_info", []) or [])
        active_indices = list(getattr(self, "result_active_peak_indices", []) or [])
        rows: list[tuple[str, str, str, str, str, str]] = []
        for i, info in enumerate(all_peak_info):
            peak_idx = active_indices[i] if i < len(active_indices) else i
            peak_label = f"Peak{peak_idx + 1}"
            if peak_idx < len(getattr(self, "peak_mu_sliders", [])):
                peak_label += f" ({self.peak_mu_sliders[peak_idx].get():.2f}°)"
            details = list(info.get("peak_details", []) or [])
            total_area = self._xrd_fit_area(info)
            total_height = self._xrd_fit_height(info)
            rows.append((peak_label, self._format_area(total_area), self._format_height(total_height), "", "100%", "100%"))
            detail_metrics = []
            for j, det in enumerate(details, start=1):
                area = self._xrd_fit_area(info, det)
                height = self._xrd_fit_height(info, det)
                detail_metrics.append((j, det, area, height))
            area_denominator = sum(max(0.0, area) for _j, _det, area, _height in detail_metrics)
            height_denominator = sum(max(0.0, height) for _j, _det, _area, height in detail_metrics)
            if area_denominator <= 0:
                area_denominator = total_area
            if height_denominator <= 0:
                height_denominator = total_height
            for j, det, area, height in detail_metrics:
                center = det.get("center", None)
                area_percentage = (area / area_denominator * 100.0) if area_denominator > 0 else 0.0
                height_percentage = (height / height_denominator * 100.0) if height_denominator > 0 else 0.0
                rows.append(
                    (
                        f"Peak{peak_idx + 1}-{j}",
                        self._format_area(area),
                        self._format_height(height),
                        f"{float(center):.2f} nm" if center is not None else "",
                        f"{area_percentage:.1f}%",
                        f"{height_percentage:.1f}%",
                    )
                )
        return rows

    def _xrd_fit_area(self, info: dict, detail: dict | None = None) -> float:
        try:
            x = np.asarray(getattr(self, "x_segment"), dtype=float)
            y = np.asarray(getattr(self, "y_segment"), dtype=float)
            basis_k1 = np.asarray(info["basis_k1"], dtype=float)
            basis_k2 = np.asarray(info["basis_k2"], dtype=float)
            f_segment = np.asarray(info["f_segment"], dtype=float)
        except Exception:
            if detail is not None:
                return float(detail.get("area", 0.0) or 0.0)
            return float(np.sum(np.asarray(info.get("f_segment", []), dtype=float)))

        if detail is not None:
            idx = detail.get("indices", None)
            if idx is None or len(idx) == 0:
                return 0.0
            component = np.zeros_like(f_segment)
            component[np.asarray(idx, dtype=int)] = f_segment[np.asarray(idx, dtype=int)]
            f_segment = component

        if x.size == 0 or f_segment.size == 0:
            return 0.0
        y_scale = float(np.nanmax(y)) if y.size else 1.0
        signal = (basis_k1.dot(f_segment) + basis_k2.dot(f_segment)) * y_scale
        signal = np.clip(np.asarray(signal, dtype=float), 0.0, None)
        if signal.size != x.size:
            return float(np.sum(signal))
        try:
            return float(np.trapezoid(signal, x))
        except AttributeError:
            return float(np.trapz(signal, x))

    def _xrd_fit_height(self, info: dict, detail: dict | None = None) -> float:
        try:
            y = np.asarray(getattr(self, "y_segment"), dtype=float)
            basis_k1 = np.asarray(info["basis_k1"], dtype=float)
            basis_k2 = np.asarray(info["basis_k2"], dtype=float)
            f_segment = np.asarray(info["f_segment"], dtype=float)
        except Exception:
            if detail is not None:
                return float(detail.get("height", 0.0) or detail.get("area", 0.0) or 0.0)
            arr = np.asarray(info.get("f_segment", []), dtype=float)
            return float(np.nanmax(arr)) if arr.size else 0.0

        if detail is not None:
            idx = detail.get("indices", None)
            if idx is None or len(idx) == 0:
                return 0.0
            component = np.zeros_like(f_segment)
            safe_idx = np.asarray(idx, dtype=int)
            component[safe_idx] = f_segment[safe_idx]
            f_segment = component

        if f_segment.size == 0:
            return 0.0
        y_scale = float(np.nanmax(y)) if y.size else 1.0
        signal = (basis_k1.dot(f_segment) + basis_k2.dot(f_segment)) * y_scale
        signal = np.clip(np.asarray(signal, dtype=float), 0.0, None)
        return float(np.nanmax(signal)) if signal.size else 0.0

    @staticmethod
    def _format_area(value: float) -> str:
        value = float(value)
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.4g}"

    @staticmethod
    def _format_height(value: float) -> str:
        value = float(value)
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.4g}"

    def _metadata_rows(self, metadata: dict) -> list[tuple[str, str]]:
        if not metadata:
            return []
        r = (metadata.get("ranges") or [{}])[0]
        start = r.get("start")
        step = r.get("step")
        n_pts = r.get("n_steps")
        end = r.get("end") or (
            round(start + (n_pts - 1) * step, 4)
            if (start is not None and step and n_pts)
            else None
        )
        lam1 = metadata.get("wavelength_Ka1")
        lam2 = metadata.get("wavelength_Ka2")
        range_str = f"{start:.3f}° → {end:.3f}°" if start is not None and end is not None else ""
        return [
            ("文件名", str(metadata.get("file_name") or "")),
            ("样品名", str(metadata.get("sample_name") or "")),
            ("测量日期", str(metadata.get("date") or "")),
            ("文件格式", str(metadata.get("format") or "").replace("_", " ")),
            ("扫描模式", str(metadata.get("scan_mode") or "")),
            ("操作员", str(metadata.get("operator") or "")),
            ("靶材", str(metadata.get("anode_material") or "")),
            ("λ Kα1", f"{lam1 * 10:.5f} Å" if lam1 else ""),
            ("λ Kα2", f"{lam2 * 10:.5f} Å" if lam2 else ""),
            ("2θ 范围", range_str),
            ("步长", f"{step:.5f}°" if step else ""),
            ("数据点数", f"{n_pts}" if n_pts else ""),
            ("仪器型号", str(metadata.get("instrument") or "")),
            ("探测器", str(metadata.get("detector") or "")),
            ("光学配置", str(metadata.get("optical_config") or "")),
            ("发散狭缝", str(metadata.get("slit_div") or "")),
            ("接收狭缝", str(metadata.get("slit_receive") or "")),
            ("Kβ 滤片", str(metadata.get("kbeta_filter") or "")),
        ]
