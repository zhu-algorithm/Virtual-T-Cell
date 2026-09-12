"""Inspect public multi-omics sources before constructing the compact evidence model."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

def head_gzip(path: Path, rows: int = 3) -> list[str]:
    with gzip.open(path, "rt", errors="replace") as handle:
        return [handle.readline().rstrip("\n")[:12000] for _ in range(rows)]


def main() -> None:
    root = Path("multiomics_raw")
    report = {
        "protein_projects": {},
        "protein_groups_head": (root / "proteinGroups.txt").open(
            "rt", errors="replace"
        ).readline().rstrip("\n").split("\t"),
        "genomics_head": head_gzip(root / "QTD000031.permuted.tsv.gz"),
        "methylation_head": head_gzip(root / "GSE174666_processed.txt.gz", 2),
    }
    for accession in ("PXD021250", "PXD025174"):
        records = json.loads((root / f"{accession}_files.json").read_text(encoding="utf-8"))
        report["protein_projects"][accession] = [
            {"name": item["fileName"], "bytes": item["fileSizeBytes"],
             "category": item["fileCategory"].get("value", "")}
            for item in records
        ]
    Path("multiomics_source_inspection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
