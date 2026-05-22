"""
支持以 `python -m xrd_analyzer` 方式运行。
"""
import os
import multiprocessing as mp

os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
from xrd_analyzer.app import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
