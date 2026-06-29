"""
XRD file import dialog, modeled after the BET app's two-list importer.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from PyQt5 import QtCore, QtWidgets


SUPPORTED_SUFFIXES = {".txt", ".raw"}


class XRDFileImportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        initial_dir: str | Path | None = None,
        existing_paths: Iterable[str] | None = None,
        available_sort: tuple[int, QtCore.Qt.SortOrder] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导入数据文件")
        self.resize(1060, 640)
        self.setMinimumSize(860, 520)

        self.current_directory = Path(initial_dir or Path.cwd())
        self._available_paths: list[Path] = []
        self._selected_paths: list[Path] = [Path(p) for p in (existing_paths or []) if p]
        self._available_sort_column = int(available_sort[0]) if available_sort is not None else 0
        self._available_sort_order = available_sort[1] if available_sort is not None else QtCore.Qt.AscendingOrder

        folder_label = QtWidgets.QLabel("文件夹")
        self.folder_edit = QtWidgets.QLineEdit(str(self.current_directory))
        self.folder_edit.returnPressed.connect(self._set_directory_from_edit)

        browse_button = QtWidgets.QToolButton()
        browse_button.setText("...")
        browse_button.setToolTip("选择文件夹")
        browse_button.clicked.connect(self._browse_directory)

        refresh_button = QtWidgets.QToolButton()
        refresh_button.setText("刷新")
        refresh_button.clicked.connect(self._scan_directory)

        folder_layout = QtWidgets.QHBoxLayout()
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_edit, 1)
        folder_layout.addWidget(browse_button)
        folder_layout.addWidget(refresh_button)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("按文件名筛选")
        self.search_edit.textChanged.connect(lambda _text: self._populate_tables())

        search_layout = QtWidgets.QHBoxLayout()
        search_layout.addWidget(QtWidgets.QLabel("筛选"))
        search_layout.addWidget(self.search_edit, 1)

        self.available_table = self._make_file_table()
        self.selected_table = self._make_file_table()
        self.available_table.setSortingEnabled(True)
        self.available_table.horizontalHeader().sortIndicatorChanged.connect(self._on_available_sort_changed)
        self.available_table.itemDoubleClicked.connect(lambda _item: self._move_selected_to_right())
        self.selected_table.itemDoubleClicked.connect(lambda _item: self._move_selected_to_left())

        available_box = self._make_group("可导入文件", self.available_table)
        selected_box = self._make_group("待导入文件", self.selected_table)

        self.to_right_button = self._arrow_button(">", "添加选中文件")
        self.to_left_button = self._arrow_button("<", "移回选中文件")
        self.all_right_button = self._arrow_button(">>", "添加全部文件")
        self.all_left_button = self._arrow_button("<<", "全部移回")
        self.to_right_button.clicked.connect(self._move_selected_to_right)
        self.to_left_button.clicked.connect(self._move_selected_to_left)
        self.all_right_button.clicked.connect(self._move_all_to_right)
        self.all_left_button.clicked.connect(self._move_all_to_left)

        move_layout = QtWidgets.QVBoxLayout()
        move_layout.addStretch(1)
        for button in (self.to_right_button, self.to_left_button, self.all_right_button, self.all_left_button):
            move_layout.addWidget(button)
        move_layout.addStretch(1)

        self.move_up_button = self._arrow_button("↑", "上移选中文件")
        self.move_down_button = self._arrow_button("↓", "下移选中文件")
        self.move_up_button.clicked.connect(lambda: self._move_selected_rows(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected_rows(1))
        order_layout = QtWidgets.QHBoxLayout()
        order_layout.addStretch(1)
        order_layout.addWidget(self.move_up_button)
        order_layout.addWidget(self.move_down_button)

        selected_panel = QtWidgets.QWidget()
        selected_layout = QtWidgets.QVBoxLayout(selected_panel)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.addWidget(selected_box, 1)
        selected_layout.addLayout(order_layout)

        picker_layout = QtWidgets.QHBoxLayout()
        picker_layout.addWidget(available_box, 1)
        picker_layout.addLayout(move_layout)
        picker_layout.addWidget(selected_panel, 1)

        self.count_label = QtWidgets.QLabel("")
        self.import_button = QtWidgets.QPushButton("导入")
        self.import_button.clicked.connect(self.accept)
        cancel_button = QtWidgets.QPushButton("取消")
        cancel_button.clicked.connect(self.reject)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(self.count_label, 1)
        bottom_layout.addWidget(self.import_button)
        bottom_layout.addWidget(cancel_button)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        main_layout.addLayout(folder_layout)
        main_layout.addLayout(search_layout)
        main_layout.addLayout(picker_layout, 1)
        main_layout.addLayout(bottom_layout)

        for table in (self.available_table, self.selected_table):
            table.itemSelectionChanged.connect(self._update_buttons)

        self._scan_directory()

    def selected_paths(self) -> list[str]:
        return [str(path) for path in self._selected_paths]

    def available_sort(self) -> tuple[int, QtCore.Qt.SortOrder]:
        return (self._available_sort_column, self._available_sort_order)

    @staticmethod
    def _make_group(title: str, table: QtWidgets.QTableWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(table)
        return group

    @staticmethod
    def _make_file_table() -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["文件名", "格式", "修改时间", "大小"])
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(26)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        return table

    @staticmethod
    def _arrow_button(text: str, tooltip: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setFixedSize(44, 32)
        button.setToolTip(tooltip)
        return button

    def _browse_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", str(self.current_directory))
        if path:
            self.current_directory = Path(path)
            self.folder_edit.setText(str(self.current_directory))
            self._scan_directory()

    def _set_directory_from_edit(self) -> None:
        path = Path(self.folder_edit.text().strip())
        if not path.is_dir():
            QtWidgets.QMessageBox.warning(self, "文件夹不存在", str(path))
            self.folder_edit.setText(str(self.current_directory))
            return
        self.current_directory = path
        self._scan_directory()

    def _scan_directory(self) -> None:
        self.folder_edit.setText(str(self.current_directory))
        selected = {self._path_key(path) for path in self._selected_paths}
        try:
            files = [
                path
                for path in self.current_directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and self._path_key(path) not in selected
            ]
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "无法读取文件夹", str(exc))
            files = []
        self._available_paths = sorted(files, key=lambda p: p.name.lower())
        self._populate_tables()

    def _populate_tables(self) -> None:
        query = self.search_edit.text().strip().lower()
        available = [p for p in self._available_paths if query in p.name.lower()]
        self._fill_table(self.available_table, available)
        self._fill_table(self.selected_table, self._selected_paths)
        self.count_label.setText(f"可导入 {len(available)} 个，待导入 {len(self._selected_paths)} 个")
        self._update_buttons()

    def _fill_table(self, table: QtWidgets.QTableWidget, paths: list[Path]) -> None:
        table.setSortingEnabled(False)
        table.setRowCount(0)
        for path in paths:
            row = table.rowCount()
            table.insertRow(row)
            values = [path.name, path.suffix.upper().lstrip("."), self._modified_text(path), self._size_text(path)]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.UserRole, str(path))
                if column in {1, 3}:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                table.setItem(row, column, item)
        table.setSortingEnabled(table is self.available_table)
        if table is self.available_table:
            table.sortItems(self._available_sort_column, self._available_sort_order)

    def _on_available_sort_changed(self, column: int, order: QtCore.Qt.SortOrder) -> None:
        self._available_sort_column = int(column)
        self._available_sort_order = order

    @staticmethod
    def _modified_text(path: Path) -> str:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return ""

    @staticmethod
    def _size_text(path: Path) -> str:
        try:
            size = path.stat().st_size
        except OSError:
            return ""
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        if size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"

    @staticmethod
    def _path_key(path: Path) -> str:
        try:
            return str(path.resolve()).lower()
        except OSError:
            return str(path).lower()

    def _selected_table_paths(self, table: QtWidgets.QTableWidget) -> list[Path]:
        rows = sorted({idx.row() for idx in table.selectionModel().selectedRows()})
        paths = []
        for row in rows:
            item = table.item(row, 0)
            if item is not None:
                paths.append(Path(str(item.data(QtCore.Qt.UserRole))))
        return paths

    def _move_selected_to_right(self) -> None:
        paths = self._selected_table_paths(self.available_table)
        self._add_to_selected(paths)

    def _add_to_selected(self, paths: list[Path]) -> None:
        if not paths:
            return
        existing = {self._path_key(path) for path in self._selected_paths}
        for path in paths:
            key = self._path_key(path)
            if key not in existing:
                self._selected_paths.append(path)
                existing.add(key)
        moved = {self._path_key(path) for path in paths}
        self._available_paths = [path for path in self._available_paths if self._path_key(path) not in moved]
        self._populate_tables()

    def _move_selected_to_left(self) -> None:
        paths = self._selected_table_paths(self.selected_table)
        self._remove_from_selected(paths)

    def _remove_from_selected(self, paths: list[Path]) -> None:
        if not paths:
            return
        remove = {self._path_key(path) for path in paths}
        self._selected_paths = [path for path in self._selected_paths if self._path_key(path) not in remove]
        existing = {self._path_key(path) for path in self._available_paths}
        for path in paths:
            if path.exists() and self._path_key(path) not in existing:
                self._available_paths.append(path)
                existing.add(self._path_key(path))
        self._available_paths.sort(key=lambda p: p.name.lower())
        self._populate_tables()

    def _move_all_to_right(self) -> None:
        self._add_to_selected(list(self._available_paths))

    def _move_all_to_left(self) -> None:
        self._remove_from_selected(list(self._selected_paths))

    def _move_selected_rows(self, direction: int) -> None:
        rows = sorted({index.row() for index in self.selected_table.selectionModel().selectedRows()})
        if not rows or (direction < 0 and rows[0] == 0) or (
            direction > 0 and rows[-1] >= len(self._selected_paths) - 1
        ):
            return
        if direction > 0:
            rows = list(reversed(rows))
        for row in rows:
            target = row + direction
            self._selected_paths[row], self._selected_paths[target] = (
                self._selected_paths[target],
                self._selected_paths[row],
            )
        selected_after = [row + direction for row in rows]
        self._populate_tables()
        self.selected_table.clearSelection()
        for row in selected_after:
            self.selected_table.selectRow(row)

    def _update_buttons(self) -> None:
        has_available_selection = bool(self.available_table.selectionModel().selectedRows())
        has_selected_selection = bool(self.selected_table.selectionModel().selectedRows())
        self.to_right_button.setEnabled(has_available_selection)
        self.to_left_button.setEnabled(has_selected_selection)
        self.all_right_button.setEnabled(bool(self._available_paths))
        self.all_left_button.setEnabled(bool(self._selected_paths))
        self.move_up_button.setEnabled(has_selected_selection and min(self._selected_rows(self.selected_table), default=0) > 0)
        self.move_down_button.setEnabled(
            has_selected_selection
            and max(self._selected_rows(self.selected_table), default=-1) < len(self._selected_paths) - 1
        )
        self.import_button.setEnabled(bool(self._selected_paths))
        self.count_label.setText(f"可导入 {len(self._available_paths)} 个，待导入 {len(self._selected_paths)} 个")

    @staticmethod
    def _selected_rows(table: QtWidgets.QTableWidget) -> list[int]:
        return sorted({index.row() for index in table.selectionModel().selectedRows()})
