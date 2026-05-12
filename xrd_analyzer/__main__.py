"""
支持以 `python -m xrd_analyzer` 方式运行。
"""
import multiprocessing as mp
from xrd_analyzer.app import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
