"""Build two independently installable organ starters, with complete local source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "distribution" / "organ_starter"


def build_package(organ: str, output_dir: str | Path) -> Path:
    from virtual_t_cell.learning.organ_demo import prepare_organ_inputs
    if organ not in {"liver", "spleen"}:
        raise ValueError("organ must be liver or spleen")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = f"virtual_{organ}_learning_starter"
    archive = output_dir / f"{package_name}.zip"
    chinese = "肝脏" if organ == "liver" else "脾脏"
    with tempfile.TemporaryDirectory(prefix="vtc-organ-build-") as temporary:
        root = Path(temporary) / package_name
        root.mkdir()
        shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        for name in ("pyproject.toml", "LICENSE", "THIRD_PARTY_DATA.md"):
            shutil.copyfile(ROOT / name, root / name)
        for name in ("run.py", "run_pipeline.py", "run_demo.sh", "run_demo.bat", "requirements.txt", "requirements-h5ad.txt"):
            shutil.copyfile(TEMPLATE / name, root / name)
        shutil.copyfile(TEMPLATE / "run_demo.sh", root / "run_demo.command")
        (root / "run_demo.sh").chmod(0o755)
        (root / "run_demo.command").chmod(0o755)
        readme = (TEMPLATE / "README.template.md").read_text(encoding="utf-8")
        readme = readme.replace("{{ORGAN}}", organ).replace("{{ORGAN_NAME}}", chinese)
        (root / "README.md").write_text(readme, encoding="utf-8")
        shutil.copytree(ROOT / "docs" / "learning", root / "docs" / "learning")
        tests = root / "tests"
        tests.mkdir()
        for path in sorted((ROOT / "tests").glob("test_learning*.py")):
            shutil.copyfile(path, tests / path.name)
        config = {"schema_version": 1, "organ": organ, "display_name": f"{chinese}虚拟T细胞持续学习研究版",
                  "default_evidence": "synthetic", "required_python": "3.12", "gpu_required": False}
        (root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prepare_organ_inputs(root / "examples", organ)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        files = sorted(path for path in root.rglob("*") if path.is_file())
        manifest = {"organ": organ, "evidence_type": "synthetic", "source_base_commit": commit,
                    "independent_of_other_organ_package": True, "real_organ_training_weights_included": False,
                    "files": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
        (root / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(root.parent).as_posix())
        with zipfile.ZipFile(archive) as handle:
            if handle.testzip() is not None:
                raise ValueError("archive failed its integrity check")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="run_output/organ_packages")
    parser.add_argument("--organ", choices=("liver", "spleen", "both"), default="both")
    args = parser.parse_args()
    organs = ("liver", "spleen") if args.organ == "both" else (args.organ,)
    output = []
    for organ in organs:
        path = build_package(organ, args.out_dir)
        output.append({"organ": organ, "file": str(path), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
