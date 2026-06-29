"""
utils.py
--------
通用工具函数。
"""
import os
import sys


def resource_path(name: str) -> str:
    """
    返回资源文件（logo.ico、long.png 等）的绝对路径。
    兼容 PyInstaller 打包后的 _MEIPASS 临时目录，以及开发环境下的项目根目录。
    """
    # PyInstaller 打包后资源解压到 sys._MEIPASS
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        # 开发模式：资源与本包同级的项目根目录
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)
