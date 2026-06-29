"""
Small PyQt5 helpers used by the migrated UI.
"""
from __future__ import annotations

import math
import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QSlider,
    QWidget,
)


SLIDER_STYLE = """
QLabel {
    color: #111827;
    font: 9pt "Microsoft YaHei UI";
}
QDoubleSpinBox {
    min-height: 22px;
    padding: 1px 4px;
    border: 1px solid #c6cbd0;
    border-radius: 3px;
    background: #ffffff;
    color: #111827;
    font: 9pt "Microsoft YaHei UI";
}
QSlider {
    min-height: 24px;
    max-height: 24px;
}
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #d8dde2;
}
QSlider::sub-page:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #8f99a3;
}
QSlider::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
    border: 1px solid #66717b;
    background: #ffffff;
}
QSlider::handle:horizontal:hover {
    border-color: #3f6d94;
    background: #f7fbff;
}
"""


class QtDoubleSlider(QWidget):
    """A label + slider + spin box control with Tk-like get/set helpers."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        from_: float,
        to: float,
        value: float,
        resolution: float,
        parent: QWidget | None = None,
        color: str | None = None,
    ):
        super().__init__(parent)
        self.setStyleSheet(SLIDER_STYLE)
        self.setMinimumHeight(28)
        self._minimum = float(from_)
        self._maximum = float(to)
        self._resolution = float(resolution) if resolution else 0.01
        self._commands = []

        decimals = max(0, int(math.ceil(-math.log10(self._resolution))) if self._resolution < 1 else 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(4)

        self.label = QLabel(label)
        self.label.setMinimumWidth(78)
        if color:
            self.label.setStyleSheet(f'color: {color}; font: 9pt "Microsoft YaHei UI"; font-weight: 600;')
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFixedHeight(24)
        layout.addWidget(self.slider, 1)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(self._resolution)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(58)
        self.spin.setMaximumWidth(66)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider_value)
        self.spin.valueChanged.connect(self._on_spin_value)
        self._apply_range()
        self.set(value, emit=False)

    def _to_slider(self, value: float) -> int:
        return int(round((float(value) - self._minimum) / self._resolution))

    def _from_slider(self, value: int) -> float:
        return self._minimum + int(value) * self._resolution

    def _clamp(self, value: float) -> float:
        if self._maximum < self._minimum:
            self._maximum = self._minimum
        return min(self._maximum, max(self._minimum, float(value)))

    def _apply_range(self):
        if self._maximum < self._minimum:
            self._maximum = self._minimum
        steps = max(0, int(round((self._maximum - self._minimum) / self._resolution)))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setRange(0, steps)
        self.spin.setRange(self._minimum, self._maximum)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def _sync_widgets(self, value: float):
        value = self._clamp(value)
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.spin.setValue(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def _emit(self, value: float):
        self.valueChanged.emit(float(value))
        for command in list(self._commands):
            try:
                command(float(value))
            except TypeError:
                command()

    def _on_slider_value(self, value: int):
        new_value = self._clamp(self._from_slider(value))
        self._sync_widgets(new_value)
        self._emit(new_value)

    def _on_spin_value(self, value: float):
        new_value = self._clamp(value)
        self._sync_widgets(new_value)
        self._emit(new_value)

    def get(self) -> float:
        return float(self.spin.value())

    def set(self, value: float, emit: bool = True):
        old_value = self.get() if hasattr(self, "spin") else None
        new_value = self._clamp(value)
        self._sync_widgets(new_value)
        if emit and (old_value is None or abs(old_value - new_value) >= self._resolution / 2):
            self._emit(new_value)

    def config(self, **kwargs):
        if "from_" in kwargs:
            self._minimum = float(kwargs["from_"])
        if "to" in kwargs:
            self._maximum = float(kwargs["to"])
        if "label" in kwargs:
            self.label.setText(str(kwargs["label"]))
        if "fg" in kwargs:
            self.label.setStyleSheet(f'color: {kwargs["fg"]}; font: 9pt "Microsoft YaHei UI"; font-weight: 600;')
        if "state" in kwargs:
            enabled = str(kwargs["state"]).lower() not in {"disabled", "false", "0"}
            self.setEnabled(enabled)
        if "command" in kwargs and kwargs["command"] is not None:
            self._commands.append(kwargs["command"])
        self._apply_range()
        self.set(self.get(), emit=False)

    def destroy(self):
        self.setParent(None)
        self.deleteLater()


class QtLabeledSpin(QWidget):
    """A compact label + QDoubleSpinBox with Tk-like get/set helpers."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        from_: float,
        to: float,
        value: float,
        decimals: int = 2,
        step: float = 0.01,
        parent: QWidget | None = None,
        color: str | None = None,
    ):
        super().__init__(parent)
        self._commands = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(4)
        self.label = QLabel(label)
        self.label.setMinimumWidth(72)
        self.label.setStyleSheet(
            f'color: {color}; font: 9pt "Microsoft YaHei UI"; font-weight: 600;'
            if color
            else 'color: #111827; font: 9pt "Microsoft YaHei UI";'
        )
        layout.addWidget(self.label)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(float(from_), float(to))
        self.spin.setSingleStep(float(step))
        self.spin.setKeyboardTracking(False)
        self.spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin.setFixedWidth(86)
        self.spin.setStyleSheet(
            'QDoubleSpinBox { min-height: 22px; padding: 1px 4px;'
            ' border: 1px solid #c6cbd0; border-radius: 3px;'
            ' background: white; color: #111827; font: 9pt "Microsoft YaHei UI"; }'
        )
        self.spin.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.spin)
        layout.addStretch(1)
        self.set(value, emit=False)

    def _on_value_changed(self, value: float):
        value = float(value)
        self.valueChanged.emit(value)
        for command in list(self._commands):
            try:
                command(value)
            except TypeError:
                command()

    def get(self) -> float:
        return float(self.spin.value())

    def set(self, value: float, emit: bool = True):
        self.spin.blockSignals(True)
        self.spin.setValue(float(value))
        self.spin.blockSignals(False)
        if emit:
            self._on_value_changed(self.get())

    def config(self, **kwargs):
        if "from_" in kwargs or "to" in kwargs:
            low = float(kwargs.get("from_", self.spin.minimum()))
            high = float(kwargs.get("to", self.spin.maximum()))
            if high < low:
                high = low
            self.spin.setRange(low, high)
            self.set(self.get(), emit=False)
        if "label" in kwargs:
            self.label.setText(str(kwargs["label"]))
        if "fg" in kwargs:
            self.label.setStyleSheet(f'color: {kwargs["fg"]}; font: 9pt "Microsoft YaHei UI"; font-weight: 600;')
        if "state" in kwargs:
            enabled = str(kwargs["state"]).lower() not in {"disabled", "false", "0"}
            self.setEnabled(enabled)
        if "command" in kwargs and kwargs["command"] is not None:
            self._commands.append(kwargs["command"])

    def destroy(self):
        self.setParent(None)
        self.deleteLater()


class QtSpinSlider(QWidget):
    """Label + spin row with a horizontal slider underneath."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        from_: float,
        to: float,
        value: float,
        resolution: float,
        decimals: int = 2,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._minimum = float(from_)
        self._maximum = float(to)
        self._resolution = float(resolution)
        self._commands = []
        self.setStyleSheet(SLIDER_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(1)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        self.label = QLabel(label)
        self.label.setMinimumWidth(max(0, self.label.fontMetrics().horizontalAdvance(str(label)) + 8))
        row_layout.addWidget(self.label)
        row_layout.addStretch(1)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(self._minimum, self._maximum)
        self.spin.setSingleStep(self._resolution)
        self.spin.setKeyboardTracking(False)
        self.spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin.setFixedWidth(78)
        row_layout.addWidget(self.spin)
        layout.addWidget(row)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFixedHeight(24)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_slider_value)
        self.spin.valueChanged.connect(self._on_spin_value)
        self._apply_range()
        self.set(value, emit=False)

    def _to_slider(self, value: float) -> int:
        return int(round((float(value) - self._minimum) / self._resolution))

    def _from_slider(self, value: int) -> float:
        return self._minimum + int(value) * self._resolution

    def _apply_range(self):
        steps = max(0, int(round((self._maximum - self._minimum) / self._resolution)))
        self.slider.blockSignals(True)
        self.slider.setRange(0, steps)
        self.slider.blockSignals(False)

    def _sync(self, value: float):
        value = min(self._maximum, max(self._minimum, float(value)))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.spin.setValue(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        return value

    def _emit(self, value: float):
        self.valueChanged.emit(float(value))
        for command in list(self._commands):
            try:
                command(float(value))
            except TypeError:
                command()

    def _on_slider_value(self, value: int):
        self._emit(self._sync(self._from_slider(value)))

    def _on_spin_value(self, value: float):
        self._emit(self._sync(value))

    def get(self) -> float:
        return float(self.spin.value())

    def set(self, value: float, emit: bool = True):
        value = self._sync(value)
        if emit:
            self._emit(value)

    def config(self, **kwargs):
        if "command" in kwargs and kwargs["command"] is not None:
            self._commands.append(kwargs["command"])


class TextValue:
    """Minimal StringVar-like wrapper around a QLabel."""

    def __init__(self, label: QLabel):
        self.label = label
        self._value = ""

    def set(self, value):
        self._value = str(value)
        self.label.setText(self._value)

    def get(self) -> str:
        return self._value


class FileDialogAdapter:
    @staticmethod
    def askopenfilename(title="", filetypes=None):
        filt = (
            "所有支持的格式 (*.txt *.raw *.RAW);;"
            "TXT 文本文件 (*.txt);;"
            "Bruker/Rigaku RAW (*.raw *.RAW);;"
            "所有文件 (*)"
        )
        path, _ = QFileDialog.getOpenFileName(QApplication.activeWindow(), title, "", filt)
        return path

    @staticmethod
    def asksaveasfilename(defaultextension="", filetypes=None):
        filt = "CSV 文件 (*.csv);;所有文件 (*)"
        path, _ = QFileDialog.getSaveFileName(QApplication.activeWindow(), "保存结果", "", filt)
        if path and defaultextension and not os.path.splitext(path)[1]:
            path += defaultextension
        return path


class MessageBoxAdapter:
    @staticmethod
    def showwarning(title, message):
        QMessageBox.warning(QApplication.activeWindow(), title, message)

    @staticmethod
    def showinfo(title, message):
        QMessageBox.information(QApplication.activeWindow(), title, message)

    @staticmethod
    def showerror(title, message):
        QMessageBox.critical(QApplication.activeWindow(), title, message)

    @staticmethod
    def showerror(title, message):
        QMessageBox.critical(QApplication.activeWindow(), title, message)
