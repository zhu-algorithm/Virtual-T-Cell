"""Print schemas of the compact primary T-cell source archives in CI logs."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd


def inspect_archive(path: Path) -> None:
    print(f"ARCHIVE {path.name} {path.stat().st_size}")
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.lower().endswith((".xlsx", ".csv", ".txt", ".tsv")) and not "/._" in name:
                info = archive.getinfo(name)
                print(f"MEMBER {name} {info.file_size}")
                if name.lower().endswith(".xlsx") and info.file_size < 40_000_000:
                    with archive.open(name) as handle:
                        book = pd.ExcelFile(handle)
                        for sheet in book.sheet_names:
                            frame = pd.read_excel(book, sheet_name=sheet, nrows=3)
                            print(f"SHEET {name} :: {sheet} :: {frame.columns.tolist()}")
                            print(frame.head(2).to_json(orient="records"))
                elif info.file_size < 5_000_000:
                    with archive.open(name) as handle:
                        try:
                            frame = pd.read_csv(handle, sep=None, engine="python", nrows=3)
                            print(f"TABLE {name} :: {frame.columns.tolist()}")
                            print(frame.head(2).to_json(orient="records"))
                        except Exception as exc:
                            print(f"SKIP {name}: {exc}")


if __name__ == "__main__":
    for item in sys.argv[1:]:
        inspect_archive(Path(item))
