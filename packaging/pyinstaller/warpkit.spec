# PyInstaller multi-binary spec: one --onedir bundle, all wk-* CLIs share _internal/.
# Driven by packaging/pyinstaller/build_bundle.py.

from pathlib import Path

LAUNCHERS_DIR = Path(SPECPATH) / "launchers"
HOOKS_DIR = Path(SPECPATH) / "hooks"

SCRIPTS = [
    "wk-medic",
    "wk-unwrap-phase",
    "wk-compute-fieldmap",
    "wk-apply-warp",
    "wk-convert-warp",
    "wk-convert-fieldmap",
    "wk-compute-jacobian",
]

# nibabel + indexed_gzip rely on string-based imports that PyInstaller's static
# analysis misses; everything else (numpy/scipy/skimage/transforms3d) has a hook
# shipped with PyInstaller.
HIDDEN_IMPORTS = [
    "indexed_gzip",
    "nibabel.streamlines",
    "nibabel.nifti1",
    "nibabel.nifti2",
]

analyses = []
for name in SCRIPTS:
    a = Analysis(
        [str(LAUNCHERS_DIR / f"{name}.py")],
        pathex=[],
        binaries=[],
        datas=[],
        hiddenimports=HIDDEN_IMPORTS,
        hookspath=[str(HOOKS_DIR)],
        hooksconfig={},
        runtime_hooks=[],
        excludes=[],
        noarchive=False,
    )
    analyses.append(a)

# Deduplicate shared libraries/data across all analyses so _internal/ has one copy.
MERGE(*[(a, name, name) for a, name in zip(analyses, SCRIPTS)])

exe_list = []
for a, name in zip(analyses, SCRIPTS):
    pyz = PYZ(a.pure, a.zipped_data)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    exe_list.append(exe)

collect_args = list(exe_list)
for a in analyses:
    collect_args.extend([a.binaries, a.zipfiles, a.datas])

COLLECT(
    *collect_args,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="warpkit",
)
