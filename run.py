"""
run.py — 项目启动脚本
----------------------
把本文件（run.py）和 xrd_analyzer/ 文件夹放在同一目录下，
直接运行本文件即可：

    python run.py

无需任何路径配置。
"""
import sys
import os
import multiprocessing as mp

os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "INTEL")

# 确保 xrd_analyzer 包可以被找到
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _self_test() -> None:
    """Import numerical stack and compile Numba kernels for packaging checks."""
    import numpy as np
    from scipy.optimize import nnls
    from PyQt5.QtWidgets import QApplication
    import pyqtgraph as pg
    from xrd_analyzer.core.peak_functions import precompile_numba_functions

    precompile_numba_functions()
    a = np.eye(2)
    b = np.ones(2)
    nnls(a, b)
    app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    plot.plot([0, 1], [0, 1])
    plot.close()

if __name__ == "__main__":
    mp.freeze_support()
    if "--self-test" in sys.argv:
        try:
            _self_test()
        except Exception:
            import traceback

            log_path = os.path.join(
                os.path.dirname(os.path.abspath(sys.executable)),
                "selftest_error.log",
            )
            with open(log_path, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
            os._exit(1)
        os._exit(0)
    else:
        from xrd_analyzer.app import main

        main()
