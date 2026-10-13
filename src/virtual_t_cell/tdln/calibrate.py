"""Build an auditable TDLN calibration from downloaded public GEO tables."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .model import CELL_TYPES


def _coarse(label: str) -> str:
    x = label.lower()
    if "epithelial" in x: return "tumor_cell"
    if "cd8" in x or "tissue resident t" in x or "mait" in x:
        if "dysfunction" in x: return "CD8_exhausted"
        if "naive" in x: return "CD8_naive"
        return "CD8_effector"
    if "treg" in x: return "Treg"
    if "tfh" in x: return "Tfh"
    if "cd4" in x: return "CD4_naive"
    if "gc b" in x: return "GC_B"
    if "plasma" in x or "plasmablast" in x: return "plasma"
    if " b" in x or x.endswith("b") or "b cell" in x: return "B_naive"
    if "nk" in x: return "NK"
    if "cdc1" in x: return "cDC1"
    if "cdc2" in x or "mregdc" in x: return "cDC2"
    if x == "pdc" or x == "il3": return "pDC"
    if "monocyte" in x: return "monocyte"
    if "macrophage" in x or "tam" in x or "myeloid" in x or "neutrophil" in x or "mast" in x:
        return "macrophage"
    return "macrophage"


def _distribution(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame["model_cell_type"].value_counts().reindex(CELL_TYPES, fill_value=0).astype(float)
    # Immune-focused scRNA under-samples structural cells; tiny explicit priors keep the organ executable.
    priors = {"lymphatic_endothelial": .002, "blood_endothelial": .002,
              "fibroblast": .003, "tumor_cell": .001}
    counts /= counts.sum()
    counts *= 1 - sum(priors.values())
    for key, value in priors.items(): counts[key] = value
    return {k: float(v) for k, v in counts.items()}


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return float((np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))) / 2)


def _protein_summary(path: Path, metadata: pd.DataFrame) -> dict:
    # GEO used '.1' in the protein table and '-1' in metadata for the same Cell Ranger barcode.
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        normalized = [x[:-2] + "-1" if x.endswith(".1") else x for x in header]
        lookup = metadata["model_cell_type"].to_dict()
        groups = np.asarray([lookup.get(x, "") for x in normalized], dtype=object)
        masks = {c: groups == c for c in CELL_TYPES}
        result: dict[str, dict[str, float]] = {c: {} for c in CELL_TYPES}
        features = []
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            feature = fields[0]; values = np.asarray(fields[1:], dtype=float)
            features.append(feature)
            for c, mask in masks.items():
                if mask.any(): result[c][feature] = float(values[mask].mean())
    return {"scale": "GEO-provided normalized values", "features": features,
            "mean_by_cell_type": result, "matched_cells": int(np.sum(groups != ""))}


def _activation(proteins: dict) -> dict[str, float]:
    positive = ("CD69", "CD38", "CD25", "HLA-DRA", "CD44")
    defaults = {"lymphatic_endothelial": .20, "blood_endothelial": .15,
                "fibroblast": .25, "tumor_cell": .45}
    raw = {}
    for cell, values in proteins["mean_by_cell_type"].items():
        observed = [values[x] for x in positive if x in values]
        if observed: raw[cell] = float(np.mean(observed))
    vals = np.asarray(list(raw.values()))
    lo, hi = np.nanpercentile(vals, [5, 95])
    calibrated = {c: float(np.clip(.15 + .7 * (v - lo) / max(hi - lo, 1e-9), .05, .95))
                  for c, v in raw.items()}
    return {c: calibrated[c] if c in calibrated else defaults.get(c, .30) for c in CELL_TYPES}


def _tcr_summary(paths: list[Path]) -> dict:
    clonotypes, chains, productive = [], Counter(), 0
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[frame["is_cell"].astype(str).str.lower().isin(["true", "1"])]
        productive_frame = frame[frame["productive"].astype(str).str.lower().isin(["true", "1"])]
        productive += len(productive_frame)
        chains.update(productive_frame["chain"].dropna().astype(str))
        clonotypes.extend(productive_frame["cdr3"].dropna().astype(str))
    counts = np.asarray(list(Counter(clonotypes).values()), dtype=float)
    freq = counts / counts.sum() if counts.size else counts
    entropy = float(-(freq * np.log(freq)).sum()) if counts.size else 0.0
    return {"productive_contigs": productive, "unique_cdr3": int(len(counts)),
            "shannon_entropy": entropy, "normalized_clonality": float(1 - entropy / np.log(len(counts)))
            if len(counts) > 1 else 0.0, "chain_counts": dict(chains)}


def _bulk_programs(path: Path) -> dict:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    programs = {
        "T_cell": ["CD3D", "CD3E", "TRAC"], "cytotoxicity": ["CD8A", "NKG7", "GNLY", "GZMB"],
        "B_cell": ["MS4A1", "CD79A", "CD37"], "antigen_presentation": ["HLA-DRA", "CD74", "B2M"],
        "checkpoint": ["PDCD1", "CD274", "CTLA4", "LAG3"], "lymphangiogenesis": ["VEGFC", "FLT4", "PDPN"],
    }
    out = {}
    for name, genes in programs.items():
        present = [g for g in genes if g in frame.index]
        score = np.log1p(frame.loc[present]).mean(axis=0)
        out[name] = {"genes": present, "mean_log1p_tpm": float(score.mean()),
                     "between_sample_sd": float(score.std(ddof=1))}
    return {"samples": len(frame.columns), "genes": len(frame), "programs": out}


def calibrate(raw_dir: Path, output: Path) -> dict:
    metadata = pd.read_csv(raw_dir / "GSE316782_Metadata.csv.gz", index_col=0)
    lymph = metadata[metadata["TissueType"].eq("LymphNode")].copy()
    lymph["model_cell_type"] = lymph["ClusterLabel"].map(_coarse)
    baseline = _distribution(lymph)
    per_patient = {p: _distribution(x) for p, x in lymph.groupby("Patient")}
    jsd = {}
    for patient, observed in per_patient.items():
        training = _distribution(lymph[lymph["Patient"].ne(patient)])
        jsd[patient] = _jsd(np.asarray([observed[c] for c in CELL_TYPES]),
                            np.asarray([training[c] for c in CELL_TYPES]))
    proteins = _protein_summary(raw_dir / "GSE316782_Protein_counts.tsv.gz", lymph)
    tcr_paths = sorted(raw_dir.glob("GSE212797_TCR_*.csv.gz"))
    result = {
        "schema_version": "tdln-public-calibration-v1", "clinical_use": False,
        "boundary": "Public processed cohort summaries; not a patient digital twin and not clinical prediction.",
        "sources": {
            "cell_composition_and_protein": "GSE316782", "tcr": "GSE212797", "bulk_rna": "GSE239514"
        },
        "cohort": {"lymph_node_cells": len(lymph), "patients": int(lymph.Patient.nunique()),
                   "fine_cell_labels": int(lymph.ClusterLabel.nunique())},
        "baseline_abundance": baseline, "activation": _activation(proteins),
        "protein": proteins, "tcr": _tcr_summary(tcr_paths),
        "bulk_rna": _bulk_programs(raw_dir / "GSE239514_TPM_expression_matrix.tsv.gz"),
        "validation": {"method": "leave-one-patient-out: held-out composition versus training-patient reference",
                       "per_patient_js_divergence": jsd, "mean_js_divergence": float(np.mean(list(jsd.values()))),
                       "max_js_divergence": float(np.max(list(jsd.values())))},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate TDLN model from downloaded GEO tables")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("models/tdln_public_calibration_v1.json"))
    args = parser.parse_args()
    result = calibrate(args.raw_dir, args.output)
    print(json.dumps({"output": str(args.output), "cohort": result["cohort"],
                      "validation": result["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
