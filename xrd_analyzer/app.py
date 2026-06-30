"""
app.py
------
程序入口。

运行方式（任选其一）
--------------------
  方式 1：在 Cursor / IDE 里直接打开本文件运行
          python xrd_analyzer/app.py   或直接点击 Run

  方式 2：在项目根目录运行（推荐）
          python run.py

  方式 3：作为包运行
          python -m xrd_analyzer
"""
import sys
import os

os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

# ── 路径修正 ─────────────────────────────────────────────────────────────
# 本文件在 xrd_analyzer/ 内部，直接运行时 Python 不会把父目录加入 sys.path。
# 手动插入，使 "from xrd_analyzer.xxx import ..." 在任何运行方式下都有效。
# ProcessPoolExecutor spawn 子进程重新执行本文件顶层代码，所以此修正
# 也会在子进程中生效，保证 _eval_candidate_for_index 可被正常导入。
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))  # .../xrd_analyzer/
_PARENT_DIR = os.path.dirname(_THIS_DIR)                   # 项目根目录
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import multiprocessing as mp

# ── Windows 多进程 spawn 设置（必须在任何 mp 调用之前）──────────────────
if sys.platform.startswith("win") or getattr(sys, "frozen", False):
    mp.set_start_method("spawn", force=True)


def main():
    """预编译 Numba 后启动 GUI。"""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QApplication
    from xrd_analyzer.core.peak_functions import precompile_numba_functions
    from xrd_analyzer.ui.app_window import XRDApp

    print("Pre-compiling Numba functions...")
    precompile_numba_functions()
    print("Done. Starting GUI.")

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    qt_app = QApplication.instance() or QApplication(sys.argv)
    qt_app.setStyle("windowsvista")
    qt_app.setFont(QFont("Microsoft YaHei UI", 9))
    win = XRDApp()
    win.show()
    try:
        sys.exit(qt_app.exec_())
    except KeyboardInterrupt:
        win.stop_flag.set()
        win.close()


if __name__ == "__main__":
    mp.freeze_support()   # PyInstaller 打包 Windows exe 必须
    main()
