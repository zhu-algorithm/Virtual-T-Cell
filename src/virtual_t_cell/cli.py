#!/usr/bin/env python3
"""Virtual T-cell perturbation simulator for public Perturb-seq atlases.

The default training path uses GSE92872 and has only NumPy/Pandas dependencies.
GSE137554 is supported as an optional 10x-HDF5 validation source when h5py is
installed. Predictions are transcriptomic hypotheses, not clinical efficacy.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PATHWAYS = {
    "TCR_SIGNALING": ["CD3D", "CD3E", "CD3G", "LCK", "FYN", "ZAP70", "LAT", "LCP2", "ITK", "PLCG1"],
    "NFAT": ["NFATC1", "NFATC2", "NFATC3", "RCAN1", "EGR1", "EGR2", "EGR3", "IL2"],
    "NFKB": ["NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFKBIA", "TNFAIP3", "BCL3"],
    "AP1_MAPK": ["FOS", "FOSB", "JUN", "JUNB", "JUND", "DUSP1", "DUSP2", "EGR1"],
    "JAK_STAT": ["JAK1", "JAK2", "JAK3", "STAT1", "STAT3", "STAT5A", "STAT5B", "SOCS1", "SOCS3"],
    "ACTIVATION": ["CD69", "IL2RA", "IL2", "IFNG", "TNF", "MIR155HG", "MYC", "IRF4"],
    "CYTOTOXICITY": ["NKG7", "GNLY", "PRF1", "GZMB", "GZMH", "CTSW", "IFNG"],
    "EXHAUSTION": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "NR4A1", "NR4A2"],
    "PROLIFERATION": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM4", "TYMS", "STMN1"],
    "APOPTOSIS": ["BAX", "BAK1", "BCL2L11", "PMAIP1", "BBC3", "CASP3", "FAS"],
}

NETWORK = {
    "LCK": ["FYN", "ZAP70", "LAT", "PTPN6", "PTPN11"],
    "ZAP70": ["LCK", "LAT", "LCP2", "ITK"],
    "LAT": ["ZAP70", "LCP2", "PLCG1", "ITK"],
    "DOK2": ["PTPN6", "PTPN11", "LAT"],
    "PTPN6": ["LCK", "ZAP70", "PTPN11"],
    "PTPN11": ["PTPN6", "LAT", "DOK2"],
    "NFKB1": ["NFKB2", "RELA", "REL"],
    "NFKB2": ["NFKB1", "RELA", "RELB"],
    "RELA": ["NFKB1", "NFKB2", "REL"],
    "NFATC1": ["NFAT5", "EGR1", "EGR2", "EGR3"],
    "NFAT5": ["NFATC1", "NFKB1"],
    "FOS": ["JUN", "JUND", "EGR1"],
    "JUN": ["FOS", "JUND", "EGR1"],
    "JUND": ["FOS", "JUN", "EGR1"],
    "EGR1": ["EGR2", "EGR3", "FOS", "JUN"],
    "EGR2": ["EGR1", "EGR3", "NFATC1"],
    "EGR3": ["EGR1", "EGR2", "NFATC1"],
    "NR4A1": ["NFATC1", "EGR2", "EGR3"],
    "RUNX2": ["JUN", "FOS", "EGR1"],
}


def _rows(path: Path) -> Iterable[list[str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        yield from csv.reader(fh)


def read_gse92872_metadata(path: Path):
    it = _rows(path)
    meta = {}
    for _ in range(5):
        row = next(it)
        meta[row[0]] = row[1:]
    cells = meta["cell"]
    if len(cells) != len(meta["condition"]):
        raise ValueError("GSE92872 metadata and cell columns do not match")
    return cells, meta


def prepare_gse92872(path: Path, out: Path, n_genes: int = 2000):
    cells, meta = read_gse92872_metadata(path)
    n_cells = len(cells)
    lib = np.zeros(n_cells, dtype=np.float64)
    gene_rows = 0
    for row in _rows(path):
        if row[0] in {"condition", "replicate", "cell", "grna", "gene", "GENE"}:
            continue
        vals = np.asarray(row[1:], dtype=np.float64)
        lib += vals
        gene_rows += 1
    lib[lib == 0] = 1.0

    forced_names = set(sum(PATHWAYS.values(), [])) | set(NETWORK) | set(sum(NETWORK.values(), []))
    forced = {}
    heap = []
    for row in _rows(path):
        if row[0] in {"condition", "replicate", "cell", "grna", "gene", "GENE"}:
            continue
        vals = np.asarray(row[1:], dtype=np.float32)
        x = np.log1p(vals / lib * 1e4).astype(np.float32)
        score = float(np.var(x))
        if row[0].upper() in forced_names:
            forced[row[0]] = x
        item = (score, row[0], x)
        if len(heap) < n_genes:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    selected = sorted(heap, reverse=True)
    selected_names = {x[1] for x in selected}
    selected.extend((float(np.var(x)), gene, x) for gene, x in forced.items() if gene not in selected_names)
    genes = np.asarray([x[1] for x in selected], dtype=str)
    expression = np.stack([x[2] for x in selected]).T
    obs = pd.DataFrame({
        "cell": cells,
        "condition": meta["condition"],
        "replicate": meta["replicate"],
        "grna": meta["grna"],
        "target": meta["gene"],
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=expression, genes=genes, **{c: obs[c].astype(str).to_numpy() for c in obs})
    summary = {"cells": n_cells, "input_genes": gene_rows, "selected_genes": len(genes),
               "conditions": obs.condition.value_counts().to_dict(), "targets": int(obs.target.nunique())}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _group_mean(X, labels, value):
    idx = np.asarray(labels) == value
    return X[idx].mean(axis=0), int(idx.sum())


def train_model(prepared: Path, out: Path, shrinkage: float = 20.0):
    # The prepared archive is locally generated; Pandas metadata columns may be
    # stored as object-string arrays by older/newer Pandas versions.
    d = np.load(prepared, allow_pickle=True)
    X, genes = d["X"], d["genes"]
    conditions, targets, grnas = d["condition"], d["target"], d["grna"]
    conditions = conditions.astype(str)
    targets = targets.astype(str)
    grnas = grnas.astype(str)
    conds = np.unique(conditions)
    target_names = sorted(t for t in np.unique(targets) if t != "CTRL")
    baseline = np.zeros((len(conds), X.shape[1]), dtype=np.float32)
    effects = np.zeros((len(conds), len(target_names), X.shape[1]), dtype=np.float32)
    residual = np.zeros_like(effects)
    counts = np.zeros((len(conds), len(target_names)), dtype=np.int32)

    for ci, cond in enumerate(conds):
        ctrl = (conditions == cond) & (targets == "CTRL")
        baseline[ci] = X[ctrl].mean(axis=0)
        for ti, target in enumerate(target_names):
            idx = (conditions == cond) & (targets == target)
            n = int(idx.sum())
            counts[ci, ti] = n
            if n:
                raw = X[idx].mean(axis=0) - baseline[ci]
                weight = n / (n + shrinkage)
                effects[ci, ti] = raw * weight
                residual[ci, ti] = X[idx].std(axis=0) / math.sqrt(max(n, 1))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, genes=genes.astype(str), conditions=conds.astype(str), targets=np.asarray(target_names, dtype=str),
                        baseline=baseline, effects=effects, uncertainty=residual, counts=counts)
    metrics = guide_replicate_evaluation(X, conditions, targets, grnas, conds)
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def guide_replicate_evaluation(X, conditions, targets, grnas, conds):
    corrs = []
    top_corrs = []
    per_target = []
    for cond in conds:
        ctrl = (conditions == cond) & (targets == "CTRL")
        base = X[ctrl].mean(axis=0)
        for target in sorted(set(targets[(conditions == cond)])):
            if target == "CTRL":
                continue
            gs = sorted(set(grnas[(conditions == cond) & (targets == target)]))
            if len(gs) < 2:
                continue
            a = X[(conditions == cond) & (grnas == gs[0])].mean(axis=0) - base
            b = X[(conditions == cond) & (grnas == gs[1])].mean(axis=0) - base
            if np.std(a) and np.std(b):
                r = float(np.corrcoef(a, b)[0, 1])
                top = np.argsort(np.maximum(np.abs(a), np.abs(b)))[-200:]
                rt = float(np.corrcoef(a[top], b[top])[0, 1])
                corrs.append(r)
                top_corrs.append(rt)
                per_target.append({"condition": str(cond), "target": str(target), "guide_correlation": r,
                                   "top200_de_correlation": rt})
    return {"evaluation": "independent-guide concordance", "n_pairs": len(corrs),
            "median_pearson": float(np.median(corrs)) if corrs else None,
            "mean_pearson": float(np.mean(corrs)) if corrs else None,
            "median_top200_de_pearson": float(np.median(top_corrs)) if top_corrs else None,
            "mean_top200_de_pearson": float(np.mean(top_corrs)) if top_corrs else None,
            "pairs": per_target}


def _effect_for(model, condition: str, target: str):
    conds, targets = model["conditions"].tolist(), model["targets"].tolist()
    ci = conds.index(condition)
    target = target.upper()
    if target in targets:
        ti = targets.index(target)
        return model["effects"][ci, ti], model["uncertainty"][ci, ti], "observed"
    neighbors = [n for n in NETWORK.get(target, []) if n in targets]
    if not neighbors:
        raise ValueError(f"Target {target} is not observed and has no trained network neighbor")
    idx = [targets.index(n) for n in neighbors]
    return model["effects"][ci, idx].mean(axis=0), model["uncertainty"][ci, idx].mean(axis=0), "network_neighbor"


def pathway_scores(genes, delta):
    pos = {str(g).upper(): i for i, g in enumerate(genes)}
    result = []
    for name, members in PATHWAYS.items():
        idx = [pos[g] for g in members if g in pos]
        result.append({"pathway": name, "delta_score": float(np.mean(delta[idx])) if idx else np.nan,
                       "genes_measured": len(idx)})
    return pd.DataFrame(result).sort_values("delta_score", ascending=False)


def predict(model_path: Path, condition: str, perturbations: list[tuple[str, float]], out_dir: Path):
    m = np.load(model_path, allow_pickle=False)
    conds = m["conditions"].tolist()
    if condition not in conds:
        raise ValueError(f"condition must be one of {conds}")
    ci = conds.index(condition)
    delta = np.zeros(m["genes"].shape[0], dtype=np.float32)
    variance = np.zeros_like(delta)
    modes = []
    for target, strength in perturbations:
        effect, unc, mode = _effect_for(m, condition, target)
        delta += float(strength) * effect
        variance += (float(strength) * unc) ** 2
        modes.append({"target": target.upper(), "strength": float(strength), "mode": mode})
    pred = np.maximum(m["baseline"][ci] + delta, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"gene": m["genes"], "baseline_log1p_cp10k": m["baseline"][ci],
                       "predicted_log1p_cp10k": pred, "delta": delta,
                       "uncertainty_se": np.sqrt(variance)})
    df.reindex(df.delta.abs().sort_values(ascending=False).index).to_csv(out_dir / "gene_predictions.csv", index=False)
    pathway_scores(m["genes"], delta).to_csv(out_dir / "pathway_predictions.csv", index=False)
    if "screen_targets" in m.files:
        available = m["screen_targets"].astype(str).tolist()
        phenotype_rows = []
        for target, strength in perturbations:
            target = target.upper()
            if target in available:
                ti = available.index(target)
                for pi, phenotype in enumerate(m["screen_phenotypes"].astype(str)):
                    phenotype_rows.append({"target": target, "phenotype": phenotype,
                        "screen_log_fold_change": float(m["screen_lfc"][pi, ti]) * float(strength),
                        "screen_fdr": float(m["screen_fdr"][pi, ti]), "source": "Zenodo 5784651"})
        pd.DataFrame(phenotype_rows).to_csv(out_dir / "phenotype_predictions.csv", index=False)
    (out_dir / "prediction_metadata.json").write_text(json.dumps({"condition": condition, "perturbations": modes,
        "interpretation": "Transcriptomic hypothesis; not a clinical efficacy estimate."}, indent=2), encoding="utf-8")


def inspect_gse137554(annotation: Path, h5: Path):
    ann = pd.read_csv(annotation, sep="\t")
    result = {"annotated_cells": len(ann), "samples": ann["Sample"].value_counts().to_dict(),
              "h5_bytes": h5.stat().st_size, "h5py_available": False}
    try:
        import h5py  # optional
        with h5py.File(h5, "r") as f:
            result["h5py_available"] = True
            result["h5_root_keys"] = list(f.keys())
    except ImportError:
        result["note"] = "Install h5py to convert the legacy 10x HDF5 matrix."
    return result


def parse_perturbations(items):
    result = []
    for item in items:
        if ":" in item:
            gene, strength = item.split(":", 1)
        else:
            gene, strength = item, "1"
        result.append((gene, float(strength)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--expression", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--genes", type=int, default=2000)
    p = sub.add_parser("train"); p.add_argument("--prepared", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--shrinkage", type=float, default=20)
    p = sub.add_parser("prepare-gse314342"); p.add_argument("--de-h5ad", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--targets", type=int, default=512); p.add_argument("--genes", type=int, default=2048)
    p = sub.add_parser("prepare-tcr"); p.add_argument("--vdjdb", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare-primary-context"); p.add_argument("--data-tables-zip", type=Path, required=True); p.add_argument("--screens-zip", type=Path, required=True); p.add_argument("--fallback-model", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("predict-tcr"); p.add_argument("--database", type=Path, required=True); p.add_argument("--cdr3-beta"); p.add_argument("--cdr3-alpha"); p.add_argument("--max-distance", type=int, default=1); p.add_argument("--top", type=int, default=25); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("analyze-tcr"); p.add_argument("--contigs", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("predict"); p.add_argument("--model", type=Path, required=True); p.add_argument("--condition", required=True); p.add_argument("--perturb", nargs="+", required=True); p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("inspect-gse137554"); p.add_argument("--annotation", type=Path, required=True); p.add_argument("--h5", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "prepare": print(json.dumps(prepare_gse92872(args.expression, args.out, args.genes), indent=2))
    elif args.cmd == "train": print(json.dumps(train_model(args.prepared, args.out, args.shrinkage), indent=2))
    elif args.cmd == "prepare-gse314342":
        from .gse314342 import build_gse314342_model
        print(json.dumps(build_gse314342_model(args.de_h5ad, args.out, args.targets, args.genes), indent=2))
    elif args.cmd == "prepare-tcr":
        from .tcr import build_vdjdb
        print(json.dumps(build_vdjdb(args.vdjdb, args.out), indent=2))
    elif args.cmd == "prepare-primary-context":
        from .primary_context import build_primary_context_model
        print(json.dumps(build_primary_context_model(args.data_tables_zip, args.screens_zip, args.fallback_model, args.out), indent=2))
    elif args.cmd == "predict-tcr":
        from .tcr import predict_tcr
        print(json.dumps(predict_tcr(args.database, args.out, args.cdr3_beta, args.cdr3_alpha, args.max_distance, args.top), indent=2))
    elif args.cmd == "analyze-tcr":
        from .tcr import analyze_10x_repertoire
        print(json.dumps(analyze_10x_repertoire(args.contigs, args.out_dir), indent=2))
    elif args.cmd == "predict": predict(args.model, args.condition, parse_perturbations(args.perturb), args.out_dir)
    else: print(json.dumps(inspect_gse137554(args.annotation, args.h5), indent=2))


if __name__ == "__main__":
    main()
