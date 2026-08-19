"""Validate downloaded GEO archives without expanding the large sparse matrix."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path) -> int:
    with gzip.open(path, "rb") as handle:
        return sum(1 for _ in handle)


def matrix_shape(path: Path) -> list[int]:
    with gzip.open(path, "rt") as handle:
        if not handle.readline().startswith("%%MatrixMarket"):
            raise ValueError(f"Invalid MatrixMarket header: {path}")
        for line in handle:
            if not line.startswith("%"):
                return [int(value) for value in line.split()]
    raise ValueError(f"Missing MatrixMarket dimensions: {path}")


def main() -> None:
    root = Path("geo_raw")
    files = sorted(root.glob("*.gz"))
    required = {
        "GSE92872_CROP-seq_Jurkat_TCR.count_matrix.csv.gz",
        "GSE92872_CROP-seq_Jurkat_TCR.digital_expression.csv.gz",
        "GSE278572_barcodes.tsv.gz", "GSE278572_features.tsv.gz",
        "GSE278572_matrix.mtx.gz", "GSE278572_protospacer_calls_per_cell.csv.gz",
    }
    missing = required.difference(path.name for path in files)
    if missing:
        raise FileNotFoundError(f"Missing GEO files: {sorted(missing)}")
    manifest = {"datasets": ["GSE92872", "GSE278572"], "files": {}}
    for path in files:
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    matrix = root / "GSE278572_matrix.mtx.gz"
    manifest["gse278572_matrix_shape"] = matrix_shape(matrix)
    manifest["gse278572_barcodes"] = lines(root / "GSE278572_barcodes.tsv.gz")
    manifest["gse278572_features"] = lines(root / "GSE278572_features.tsv.gz")
    manifest["gse278572_protospacer_rows"] = lines(root / "GSE278572_protospacer_calls_per_cell.csv.gz") - 1
    Path("geo_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
