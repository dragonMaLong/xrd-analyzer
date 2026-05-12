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

# 确保 xrd_analyzer 包可以被找到
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xrd_analyzer.app import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
