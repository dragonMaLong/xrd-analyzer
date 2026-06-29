# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_DIR = Path(SPECPATH)
APP_NAME = "XRD-Analyzer"

datas = []
for asset_name in ("logo.ico", "xrd_analyzer/XRD-logo.png"):
    asset_path = PROJECT_DIR / asset_name
    if asset_path.exists():
        target = "." if asset_path.parent == PROJECT_DIR else str(asset_path.parent.relative_to(PROJECT_DIR))
        datas.append((str(asset_path), target))

hiddenimports = [
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "PyQt5.sip",
    "pyqtgraph",
    "scipy._lib.messagestream",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "numba.cloudpickle.cloudpickle_fast",
]

excludes = [
    "IPython",
    "jupyter",
    "notebook",
    "nbconvert",
    "nbformat",
    "pytest",
    "sphinx",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "gi",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_gtk3agg",
    "matplotlib.backends.backend_gtk4agg",
    "matplotlib.backends.backend_webagg",
    "matplotlib.backends.backend_wxagg",
    "aiohttp",
    "altair",
    "astropy",
    "astropy_iers_data",
    "av",
    "bokeh",
    "botocore",
    "boto3",
    "dask",
    "distributed",
    "docutils",
    "flask",
    "fsspec",
    "h5py",
    "holoviews",
    "imagecodecs",
    "intake",
    "jax",
    "jaxlib",
    "jedi",
    "jinja2",
    "lxml",
    "matplotlib",
    "mpl_toolkits",
    "nbclassic",
    "numexpr",
    "onnxruntime",
    "openpyxl",
    "pandas",
    "panel",
    "paramiko",
    "plotly",
    "psutil",
    "pyarrow",
    "pygments",
    "pywt",
    "skimage",
    "sklearn",
    "sqlite3",
    "sqlalchemy",
    "statsmodels",
    "tables",
    "tensorflow",
    "torch",
    "tornado",
    "win32com",
    "xarray",
    "yaml",
    "zmq",
    "tkinter",
    "_tkinter",
]


def _entry_name(item):
    return str(item[0]).replace("\\", "/").lower()


def _entry_source(item):
    if len(item) < 2:
        return ""
    return str(item[1]).replace("\\", "/").lower()


def _drop_unused_data(toc):
    drop_parts = (
        "matplotlib/backends/web_backend/",
        "matplotlib/testing/",
        "mpl_toolkits/tests/",
        "scipy/_lib/tests/",
        "scipy/linalg/tests/",
        "scipy/optimize/tests/",
        "scipy/signal/tests/",
        "scipy/sparse/tests/",
    )
    filtered = []
    for item in toc:
        name = _entry_name(item)
        src = _entry_source(item)
        if any(part in name or part in src for part in drop_parts):
            continue
        filtered.append(item)
    return filtered


def _drop_unused_mkl(toc):
    drop_prefixes = (
        "mkl_blacs",
        "mkl_scalapack",
        "mkl_cdft",
        "mkl_pgi_thread",
        "mkl_tbb_thread",
        "mkl_vml_",
    )
    drop_exact = {
        "libiomp5md_db.dll",
        "libiompstubs5md.dll",
        "mkl_avx.2.dll",
        "mkl_avx512.2.dll",
        "mkl_mc.2.dll",
        "mkl_mc3.2.dll",
        "mkl_msg.dll",
        "mkl_sequential.2.dll",
        "sqlite3.dll",
        "_sqlite3.pyd",
        "omptarget.dll",
        "omptarget.rtl.level0.dll",
        "omptarget.rtl.opencl.dll",
        "omptarget.sycl.wrap.dll",
    }
    filtered = []
    for item in toc:
        leaf = Path(_entry_name(item)).name
        if leaf in drop_exact or any(leaf.startswith(prefix) for prefix in drop_prefixes):
            continue
        filtered.append(item)
    return filtered


a = Analysis(
    ["run.py"],
    pathex=[str(PROJECT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Qt5Agg"]}},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

a.datas = _drop_unused_mkl(_drop_unused_data(a.datas))
a.binaries = _drop_unused_mkl(_drop_unused_data(a.binaries))

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    exclude_binaries=False,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_DIR / "logo.ico") if (PROJECT_DIR / "logo.ico").exists() else None,
)
