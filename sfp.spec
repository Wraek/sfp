# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SFP Desktop GUI.

Build with:
    pyinstaller sfp.spec

Output:
    dist/sfp_gui/sfp_gui.exe  (one-dir mode)

Zip the dist/sfp_gui/ folder for distribution.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all sfp submodules so PyInstaller finds them
sfp_hiddenimports = collect_submodules("sfp")

# Torch hidden imports (common ones missed by PyInstaller)
torch_hiddenimports = [
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils",
    "torch.autograd",
]

a = Analysis(
    ["sfp_gui.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=sfp_hiddenimports + torch_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude dev tools
        "pytest",
        "mypy",
        "ruff",
        "IPython",
        "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sfp_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window — GUI only
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="sfp_gui",
)
