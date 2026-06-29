"""
支持以 `python -m xrd_analyzer` 方式运行。
"""
import os
import multiprocessing as mp

os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "INTEL")

if __name__ == "__main__":
    mp.freeze_support()
    from xrd_analyzer.app import main

    main()
