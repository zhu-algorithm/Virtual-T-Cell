"""TCR repertoire ingestion and evidence-based specificity matching."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def build_vdjdb(slim_tsv: Path, out: Path, species: str = "HomoSapiens") -> dict:
    """Build a compact, pickle-free TCR evidence database from VDJdb slim."""
    csv.field_size_limit(16 * 1024 * 1024)
    columns = ["gene", "cdr3", "antigen.epitope", "antigen.gene", "antigen.species",
               "complex.id", "v.segm", "j.segm", "mhc.a", "mhc.b", "mhc.class",
               "reference.id", "vdjdb.score"]
    limits = {"gene": 3, "cdr3": 64, "antigen.epitope": 128, "antigen.gene": 256,
              "antigen.species": 256, "complex.id": 32, "v.segm": 128, "j.segm": 128,
              "mhc.a": 128, "mhc.b": 128, "mhc.class": 16, "reference.id": 512,
              "vdjdb.score": 16}
    values = {column: [] for column in columns}
    with slim_tsv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("species") != species or row.get("gene") not in {"TRA", "TRB"}:
                continue
            cdr3 = row.get("cdr3", "").strip().upper()
            if not (cdr3.startswith("C") and cdr3[-1:] in {"F", "W"}):
                continue
            for column in columns:
                values[column].append(row.get(column, "")[:limits[column]])
    if not values["cdr3"]:
        raise ValueError("No valid human TRA/TRB records found")
    arrays = {"field_" + column.replace(".", "_"): np.asarray(value, dtype=str)
              for column, value in values.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, columns=np.asarray(columns), source=np.asarray("VDJdb"),
                        species=np.asarray(species), **arrays)
    chain = arrays["field_gene"]
    cdr3_values = arrays["field_cdr3"]
    epitopes = arrays["field_antigen_epitope"]
    summary = {
        "source": "VDJdb", "source_release": slim_tsv.parent.name,
        "species": species, "records": int(len(cdr3_values)),
        "unique_cdr3": int(len(set(cdr3_values))),
        "alpha_records": int(np.sum(chain == "TRA")),
        "beta_records": int(np.sum(chain == "TRB")),
        "unique_epitopes": int(len(set(epitopes) - {""})),
        "prediction_type": "curated evidence and sequence-neighbor inference",
    }
    out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _bounded_edit(a: str, b: str, limit: int) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            value = min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb))
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def predict_tcr(database: Path, out: Path, cdr3_beta: str | None = None,
                cdr3_alpha: str | None = None, max_distance: int = 1,
                top: int = 25) -> dict:
    db = np.load(database, allow_pickle=False)
    columns = db["columns"].tolist()
    fields = {name: db["field_" + name.replace(".", "_")] for name in columns}
    queries = [("TRA", cdr3_alpha), ("TRB", cdr3_beta)]
    hits = []
    for chain, query in queries:
        if not query:
            continue
        query = query.strip().upper()
        candidates = np.flatnonzero(fields["gene"] == chain)
        for idx in candidates:
            ref = fields["cdr3"][idx]
            if abs(len(ref) - len(query)) > max_distance:
                continue
            distance = _bounded_edit(query, ref, max_distance)
            if distance <= max_distance:
                item = {name: str(fields[name][idx]) for name in columns}
                item.update({"query_chain": chain, "query_cdr3": query, "edit_distance": distance})
                hits.append(item)
    if hits:
        frame = pd.DataFrame(hits)
        frame["vdjdb.score"] = pd.to_numeric(frame["vdjdb.score"], errors="coerce").fillna(0)
        chain_support = frame.groupby(["antigen.epitope", "antigen.species"])["query_chain"].transform("nunique")
        frame["paired_chain_support"] = chain_support
        frame["confidence"] = np.where(
            frame.edit_distance == 0,
            np.where(chain_support >= 2, "high_paired_exact", "curated_exact"),
            "sequence_neighbor_hypothesis",
        )
        frame = frame.sort_values(["edit_distance", "paired_chain_support", "vdjdb.score"], ascending=[True, False, False])
        frame = frame.drop_duplicates(["antigen.epitope", "antigen.species", "query_chain"]).head(top)
    else:
        frame = pd.DataFrame(columns=columns + ["query_chain", "query_cdr3", "edit_distance",
                                                "paired_chain_support", "confidence"])
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary = {
        "queries": {"TRA": cdr3_alpha, "TRB": cdr3_beta}, "matches": int(len(frame)),
        "max_edit_distance": max_distance,
        "warning": "Neighbor matches are hypotheses, not proof of antigen specificity.",
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def analyze_10x_repertoire(contigs: Path, out_dir: Path) -> dict:
    """Summarize productive human TCR clonotypes from Cell Ranger VDJ CSV."""
    df = pd.read_csv(contigs)
    required = {"barcode", "chain", "cdr3", "productive"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing 10x contig columns: {missing}")
    productive = df["productive"].astype(str).str.lower().isin({"true", "1"})
    df = df[productive & df.chain.isin(["TRA", "TRB"]) & df.cdr3.notna()].copy()
    df["cdr3"] = df.cdr3.astype(str).str.upper()
    clone_cols = [c for c in ["chain", "cdr3", "v_gene", "j_gene"] if c in df]
    clones = df.groupby(clone_cols, dropna=False).agg(cells=("barcode", "nunique")).reset_index()
    clones = clones.sort_values("cells", ascending=False)
    paired = df.groupby("barcode")["chain"].agg(lambda x: {"TRA", "TRB"}.issubset(set(x)))
    counts = clones.cells.to_numpy(dtype=float)
    frequencies = counts / counts.sum() if counts.sum() else counts
    summary = {
        "productive_contigs": int(len(df)), "cells": int(df.barcode.nunique()),
        "clonotypes": int(len(clones)), "paired_alpha_beta_cells": int(paired.sum()),
        "shannon_diversity": float(-(frequencies * np.log(frequencies + 1e-12)).sum()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    clones.to_csv(out_dir / "clonotypes.csv", index=False)
    (out_dir / "repertoire_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
