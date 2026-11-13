#!/usr/bin/env python3
"""Create an isolated Python 3.12 environment and run this organ's own package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent


def source_stamp() -> str:
    digest = hashlib.sha256()
    paths = [ROOT / "requirements.txt", ROOT / "pyproject.toml"]
    paths += sorted((ROOT / "src").rglob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def clean_environment() -> dict:
    result = dict(os.environ)
    result.pop("PYTHONPATH", None)
    result.pop("PYTHONHOME", None)
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONUTF8"] = "1"
    result["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent organ learning starter; demo data are synthetic.")
    parser.add_argument("--no-install", action="store_true", help="Use the prepared environment without network or reinstall.")
    parser.add_argument("--h5ad", action="store_true", help="Also install the optional locked H5AD dependencies.")
    parser.add_argument("--offline-wheels", help="Install from a local directory of compatible wheels instead of PyPI.")
    parser.add_argument("--output", help="Result directory; default is output/<command> inside this package.")
    parser.add_argument("command", nargs="?", choices=("demo", "check", "ingest", "cli"), default="demo")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments forwarded to ingest or the full learning CLI.")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or struct.calcsize("P") != 8:
        parser.error("Use 64-bit CPython 3.12 for this locked distribution (Windows: py -3.12 run.py; macOS/Linux: python3.12 run.py).")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if config.get("organ") not in {"liver", "spleen"}:
        parser.error("config.json must specify exactly liver or spleen")
    env_dir = ROOT / ".venv"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stamp_path = env_dir / "organ_install.json"
    stamp = source_stamp()
    installed = json.loads(stamp_path.read_text()) if stamp_path.is_file() else {}
    environment = clean_environment()
    if not python.is_file():
        if args.no_install:
            parser.error("No isolated environment exists. Run once without --no-install.")
        print("Creating an isolated environment (no system packages)...", flush=True)
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env_dir)
    pip_options = ["--only-binary=:all:"]
    if args.offline_wheels:
        wheel_dir = Path(args.offline_wheels).expanduser().resolve()
        if not wheel_dir.is_dir():
            parser.error("--offline-wheels must point to a directory containing compatible Python 3.12 wheels")
        pip_options += ["--no-index", "--find-links", str(wheel_dir)]
    if installed.get("source_stamp") != stamp:
        if args.no_install:
            parser.error("Package source or dependency lock changed. Run once without --no-install.")
        print("Installing locked CPU dependencies. First installation needs internet unless --offline-wheels is supplied.", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", *pip_options, "-r", str(ROOT / "requirements.txt")],
                       check=True, cwd=ROOT, env=environment)
        # A normal installation verifies that the distribution contains all source
        # files. Runtime imports must not rely on this checkout or another project.
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(ROOT)],
                       check=True, cwd=ROOT, env=environment)
        installed = {"source_stamp": stamp, "python": sys.version.split()[0]}
        stamp_path.write_text(json.dumps(installed, indent=2), encoding="utf-8")
    h5ad_file = ROOT / "requirements-h5ad.txt"
    h5ad_stamp = hashlib.sha256(h5ad_file.read_bytes()).hexdigest()
    if args.h5ad and installed.get("h5ad_stamp") != h5ad_stamp:
        if args.no_install:
            parser.error("H5AD dependencies are not prepared. Run once with --h5ad without --no-install.")
        subprocess.run([str(python), "-m", "pip", "install", *pip_options, "-r", str(h5ad_file)],
                       check=True, cwd=ROOT, env=environment)
        installed["h5ad_stamp"] = h5ad_stamp
        stamp_path.write_text(json.dumps(installed, indent=2), encoding="utf-8")
    output = Path(args.output).expanduser().resolve() if args.output else ROOT / "output" / args.command
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [str(python), str(ROOT / "run_pipeline.py"), "--organ", config["organ"],
               "--output", str(output), args.command, *forwarded]
    result = subprocess.run(command, cwd=Path.cwd(), env=environment)
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Startup failed: {error}\nSee README.md troubleshooting. Completed data and previous model versions are retained.", file=sys.stderr)
        raise SystemExit(2)
