"""Inspect public multi-omics sources before constructing the compact evidence model."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd


def head_gzip(path: Path, rows: int = 3) -> list[str]:
    with gzip.open(path, "rt", errors="replace") as handle:
        return [handle.readline().rstrip("\n")[:12000] for _ in range(rows)]


def main() -> None:
    root = Path("multiomics_raw")
    protein = root / "jem_20211295_datas2.xlsx"
    workbook = pd.ExcelFile(protein)
    report = {
        "protein_bytes": protein.stat().st_size,
        "protein_sheets": workbook.sheet_names,
        "protein_sheet_columns": {},
        "genomics_head": head_gzip(root / "QTD000031.permuted.tsv.gz"),
        "methylation_head": head_gzip(root / "GSE174666_processed.txt.gz", 2),
    }
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(protein, sheet_name=sheet, nrows=5)
        report["protein_sheet_columns"][sheet] = {
            "columns": [str(value) for value in frame.columns],
            "preview": frame.fillna("").astype(str).to_dict(orient="records"),
        }
    Path("multiomics_source_inspection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
