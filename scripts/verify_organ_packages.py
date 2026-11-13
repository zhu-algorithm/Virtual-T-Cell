"""Cold-install and execute each ZIP outside the checkout, including repeat runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command: list[str], *, cwd: Path, log_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"Command exited {result.returncode}: {command}\n{tail}")


def verify_package(archive: Path, organ: str, report_dir: Path, include_h5ad: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="VTC package test ") as temporary:
        outer = Path(temporary)
        unpack = outer / "独立 解压目录"
        unpack.mkdir()
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(unpack)
        root = unpack / f"virtual_{organ}_learning_starter"
        manifest = read_json(root / "PACKAGE_MANIFEST.json")
        assert manifest["organ"] == organ
        for name, expected in manifest["files"].items():
            assert sha256(root / name) == expected, f"packaged file changed: {name}"
        outside = outer / "another working directory"
        outside.mkdir()
        output = outside / "结果 output"
        launcher = [sys.executable, str(root / "run.py"), "--output", str(output)]
        first = launcher + (["--h5ad"] if include_h5ad else []) + ["demo"]
        execute(first, cwd=outside, log_path=report_dir / f"{organ}_cold_install.log")
        report = read_json(output / "organ_report.json")
        runtime = read_json(output / "runtime.json")
        assert report["organ"] == organ
        assert report["evidence_type"] == "synthetic"
        assert set(report["atlas"]["organs"]) == {organ}
        assert report["atlas"]["n_cells"] == 16
        assert report["first_learning_cycle"]["status"] == "promoted"
        assert report["repeat_learning_cycle"]["status"] == "no_new_evidence"
        assert report["plan"]["decision"] in {"experiment", "stop"}
        assert Path(runtime["package_file"]).resolve().is_relative_to((root / ".venv").resolve()), runtime
        initial_model_hash = sha256(output / "active_model.npz")
        execute(launcher + ["--no-install", "demo"], cwd=outside, log_path=report_dir / f"{organ}_repeat.log")
        repeated = read_json(output / "organ_report.json")
        assert repeated["learning_cycle"]["status"] == "no_new_evidence"
        assert repeated["atlas"]["n_cells"] == report["atlas"]["n_cells"]
        assert sha256(output / "active_model.npz") == initial_model_hash
        csv_output = outside / "CSV input check"
        counts = root / "examples" / "synthetic_counts_1.csv"
        metadata = root / "examples" / "synthetic_metadata_1.csv"
        # Names are discovered from the explicit example manifest, not guessed.
        example_manifest = read_json(root / "examples" / "inbox" / "batch_1.json")
        counts = (root / "examples" / "inbox" / example_manifest["atlas"]["counts_csv"]).resolve()
        metadata = (root / "examples" / "inbox" / example_manifest["atlas"]["metadata_csv"]).resolve()
        execute([sys.executable, str(root / "run.py"), "--no-install", "--output", str(csv_output),
                 "ingest", "--counts", str(counts), "--metadata", str(metadata),
                 "--batch-id", "portability_csv", "--source", "synthetic_portability_fixture", "--evidence-type", "synthetic"],
                cwd=outside, log_path=report_dir / f"{organ}_csv_ingest.log")
        csv_report = read_json(csv_output / "ingestion_report.json")
        assert set(csv_report["atlas"]["organs"]) == {organ}
        assert csv_report["atlas"]["evidence_counts"] == {"synthetic": 8}
        h5ad_status = "not_requested"
        if include_h5ad:
            fixture = outside / "make_h5ad.py"
            fixture.write_text("""import sys
import anndata as ad
import numpy as np
import pandas as pd
counts=pd.read_csv(sys.argv[1],index_col='cell_id')
obs=pd.read_csv(sys.argv[2],index_col='cell_id').loc[counts.index]
a=ad.AnnData(X=np.log1p(counts.to_numpy(dtype=float)),obs=obs,var=pd.DataFrame(index=counts.columns))
a.layers['counts']=counts.to_numpy(dtype='int64')
a.write_h5ad(sys.argv[3])
""", encoding="utf-8")
            vpython = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            h5ad = outside / "input.h5ad"
            execute([str(vpython), str(fixture), str(counts), str(metadata), str(h5ad)], cwd=outside,
                    log_path=report_dir / f"{organ}_h5ad_fixture.log")
            h5ad_output = outside / "H5AD input check"
            execute([sys.executable, str(root / "run.py"), "--no-install", "--output", str(h5ad_output),
                     "ingest", "--h5ad", str(h5ad), "--counts-layer", "counts", "--batch-id", "portability_h5ad",
                     "--source", "synthetic_portability_fixture", "--evidence-type", "synthetic"],
                    cwd=outside, log_path=report_dir / f"{organ}_h5ad_ingest.log")
            h5ad_report = read_json(h5ad_output / "ingestion_report.json")
            assert h5ad_report["atlas"]["organs"] == csv_report["atlas"]["organs"]
            assert h5ad_report["atlas"]["evidence_counts"] == {"synthetic": 8}
            h5ad_status = "passed"
        for name, expected in manifest["files"].items():
            assert sha256(root / name) == expected, f"installation changed a source file: {name}"
        return {"organ": organ, "archive_sha256": sha256(archive), "status": "passed",
                "python": runtime["python"], "platform": runtime["platform"],
                "isolated_noneditable_install": True, "outside_checkout": True,
                "unicode_and_space_paths": True, "repeat_without_install": "passed",
                "model_unchanged_on_repeat": True, "atlas_cells": report["atlas"]["n_cells"],
                "csv_ingest": "passed", "h5ad_ingest": h5ad_status,
                "first_update": report["first_learning_cycle"]["status"],
                "second_update": repeated["learning_cycle"]["status"],
                "biological_validation": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", default="run_output/organ_packages")
    parser.add_argument("--report-dir", default="run_output/organ_package_verification")
    parser.add_argument("--h5ad", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("package verification requires Python 3.12")
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for organ in ("liver", "spleen"):
        archive = Path(args.archives_dir).resolve() / f"virtual_{organ}_learning_starter.zip"
        print(f"Cold-installing and verifying {organ} package...", flush=True)
        reports.append(verify_package(archive, organ, report_dir, args.h5ad))
        (report_dir / "verification.json").write_text(json.dumps({"platform": platform.platform(), "packages": reports}, indent=2) + "\n")
        print(f"{organ}: passed", flush=True)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
