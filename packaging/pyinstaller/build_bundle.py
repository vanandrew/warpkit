"""Drive PyInstaller to produce a versioned --onedir bundle + zip for one target.

Usage (from repo root, inside an env with warpkit + pyinstaller installed):

    python packaging/pyinstaller/build_bundle.py --target linux-x86_64

Produces:
    packaging/pyinstaller/dist/warpkit-${VERSION}/      (the bundle)
    packaging/pyinstaller/dist/warpkit-${VERSION}-${TARGET}.zip
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "wk-medic",
    "wk-unwrap-phase",
    "wk-compute-fieldmap",
    "wk-apply-warp",
    "wk-convert-warp",
    "wk-convert-fieldmap",
    "wk-compute-jacobian",
]


def detect_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x86_64"
        if machine in ("aarch64", "arm64"):
            return "linux-aarch64"
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        if machine in ("x86_64", "amd64"):
            return "macos-x86_64"
    raise RuntimeError(f"unsupported target: {system}/{machine}")


def get_version() -> str:
    from warpkit import __version__

    return __version__


def run_pyinstaller(spec: Path, dist: Path, work: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        str(spec),
    ]
    subprocess.run(cmd, check=True)


def adhoc_sign_macos(bundle: Path) -> None:
    if platform.system().lower() != "darwin":
        return
    # Ad-hoc sign every Mach-O in the bundle. `codesign -s -` is sufficient to
    # let users override Gatekeeper after the first launch (right-click → Open,
    # or `xattr -d com.apple.quarantine`).
    targets = [bundle / name for name in SCRIPTS]
    targets.extend(p for p in bundle.rglob("*.dylib") if p.is_file())
    targets.extend(p for p in (bundle / "_internal").rglob("*.so") if p.is_file())
    for t in targets:
        subprocess.run(
            ["codesign", "--force", "--sign", "-", "--timestamp=none", str(t)],
            check=False,
        )


def write_readme(bundle: Path, version: str, target: str) -> None:
    template = Path(__file__).parent / "bundle_README.md"
    body = (
        template.read_text().replace("@VERSION@", version).replace("@TARGET@", target)
    )
    (bundle / "README.md").write_text(body)


def make_zip(bundle: Path, out_zip: Path) -> None:
    # shutil.make_archive's base_dir keeps a tidy top-level folder inside the zip.
    base_name = str(out_zip.with_suffix(""))
    shutil.make_archive(
        base_name=base_name,
        format="zip",
        root_dir=str(bundle.parent),
        base_dir=bundle.name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=None,
        help="target triple (e.g. linux-x86_64); auto-detected by default",
    )
    args = parser.parse_args()

    here = Path(__file__).parent
    spec = here / "warpkit.spec"
    dist = here / "dist"
    work = here / "build"
    target = args.target or detect_target()
    version = get_version()

    if dist.exists():
        shutil.rmtree(dist)
    if work.exists():
        shutil.rmtree(work)

    run_pyinstaller(spec, dist, work)

    raw_bundle = dist / "warpkit"
    if not raw_bundle.is_dir():
        raise RuntimeError(f"PyInstaller did not produce {raw_bundle}")
    versioned = dist / f"warpkit-{version}"
    raw_bundle.rename(versioned)

    adhoc_sign_macos(versioned)
    write_readme(versioned, version, target)

    out_zip = dist / f"warpkit-{version}-{target}.zip"
    make_zip(versioned, out_zip)
    print(f"wrote {out_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
