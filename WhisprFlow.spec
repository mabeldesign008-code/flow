# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for WhisprFlow.

Built on a real Windows runner by .github/workflows/release.yml -- these
are onefile, windowed (no console), and self-contained: the user needs no
Python install.

The original spec had datas=[] and hiddenimports=[], so the EXE shipped
without its assets and crashed on pystray/PIL backends that PyInstaller's
static analysis cannot see.
"""

import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# sounddevice ships the PortAudio DLL as package data; without this the
# EXE builds fine and then fails at runtime with "PortAudio library not
# found" the first time you press the hotkey.
binaries = collect_dynamic_libs("sounddevice")

hiddenimports = [
    # pystray and PIL pick a backend at runtime via importlib, which
    # PyInstaller's static analysis cannot follow.
    "pystray._win32",
    "PIL._tkinter_finder",
    # pynput likewise selects a platform backend dynamically.
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "sounddevice",
    "_sounddevice_data",
    "scipy.signal",
    "scipy.special._cdflib",
    # HTTP/2 and WebSocket transports are imported lazily.
    "h2",
    "hpack",
    "hyperframe",
    "websockets",
    "websockets.legacy",
    "websockets.legacy.client",
    # Optional context providers -- guarded at import, but bundle them so
    # the feature works out of the box.
    "psutil",
    "uiautomation",
    "comtypes",
    "comtypes.stream",
]
hiddenimports += collect_submodules("comtypes")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=[
        ("assets/icon.ico", "assets"),
        ("assets/icon.png", "assets"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Nothing here is used; excluding them cuts ~80 MB off the binary.
        "matplotlib", "pandas", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "pytest", "IPython", "notebook", "sphinx", "setuptools",
        "scipy.optimize", "scipy.sparse", "scipy.interpolate",
        "scipy.integrate", "scipy.spatial", "scipy.stats",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WhisprFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX corrupts some Python extension DLLs and trips antivirus
    # heuristics. The size saving is not worth a binary that will not run.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico",
    version="version_info.txt",
)
