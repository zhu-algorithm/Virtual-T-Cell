"""Build a compact primary-CD4 T-cell model from GSE314342 DE results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_gse314342_model(
    de_h5ad: Path,
    out: Path,
    n_targets: int = 512,
    n_genes: int = 2048,
) -> dict:
    """Convert the published DE AnnData into the portable prediction format.

    Targets must be represented in Rest, Stim8hr and Stim48hr and pass the
    study's on-target, expression, guide-count and off-target QC annotations.
    The most reproducible/phenotypic targets and most variable response genes
    are retained so the distributed model remains small enough for GitHub.
    """
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - exercised in build workflow
        raise RuntimeError('Install the atlas extras: pip install -e ".[atlas]"') from exc

    a = ad.read_h5ad(de_h5ad, backed="r")
    required_obs = {
        "target_contrast_gene_name", "culture_condition", "ontarget_significant",
        "low_target_gex", "neighboring_gene_KD", "distal_offtarget_flag",
        "single_guide_estimate", "n_guides", "n_downstream", "n_cells_target",
        "guide_correlation_all",
    }
    missing = sorted(required_obs - set(a.obs.columns))
    required_layers = {"log_fc", "lfcSE", "baseMean"}
    missing_layers = sorted(required_layers - set(a.layers.keys()))
    if missing or missing_layers:
        raise ValueError(f"Unexpected GSE314342 schema; obs={missing}, layers={missing_layers}")

    obs = a.obs.copy()
    qc = (
        obs["ontarget_significant"].fillna(False).astype(bool)
        & ~obs["low_target_gex"].fillna(True).astype(bool)
        & ~obs["neighboring_gene_KD"].fillna(True).astype(bool)
        & ~obs["distal_offtarget_flag"].fillna(True).astype(bool)
        & ~obs["single_guide_estimate"].fillna(True).astype(bool)
        & (obs["n_guides"].fillna(0) >= 2)
    )
    obs = obs.loc[qc].copy()
    conditions = ["Rest", "Stim8hr", "Stim48hr"]
    complete = obs.groupby("target_contrast_gene_name")["culture_condition"].agg(
        lambda x: set(map(str, x)) >= set(conditions)
    )
    candidates = complete[complete].index
    obs = obs[obs["target_contrast_gene_name"].isin(candidates)].copy()
    obs["quality"] = (
        np.log1p(obs["n_cells_target"].fillna(0))
        + np.log1p(obs["n_downstream"].fillna(0))
        + obs["guide_correlation_all"].fillna(0).clip(lower=0)
    )
    ranking = obs.groupby("target_contrast_gene_name")["quality"].mean().sort_values(ascending=False)
    targets = ranking.head(n_targets).index.astype(str).tolist()
    if len(targets) < 200:
        raise ValueError(f"Only {len(targets)} high-confidence targets passed QC; need at least 200")

    row_lookup = {
        (str(row.target_contrast_gene_name), str(row.culture_condition)): a.obs_names.get_loc(idx)
        for idx, row in obs[obs["target_contrast_gene_name"].isin(targets)].iterrows()
    }
    rows = [row_lookup[(target, condition)] for condition in conditions for target in targets]
    log_fc_all = np.asarray(a.layers["log_fc"][rows, :], dtype=np.float32)
    lfc_se_all = np.asarray(a.layers["lfcSE"][rows, :], dtype=np.float32)
    base_mean_all = np.asarray(a.layers["baseMean"][rows, :], dtype=np.float32)
    log_fc_all = np.nan_to_num(log_fc_all, nan=0.0, posinf=0.0, neginf=0.0)
    lfc_se_all = np.nan_to_num(lfc_se_all, nan=0.0, posinf=0.0, neginf=0.0)
    base_mean_all = np.nan_to_num(base_mean_all, nan=0.0, posinf=0.0, neginf=0.0)

    gene_names = np.asarray(
        a.var["gene_name"].astype(str).to_numpy() if "gene_name" in a.var else a.var_names.astype(str).to_numpy(),
        dtype=str,
    )
    forced = {
        "CD3D", "CD3E", "CD3G", "LCK", "FYN", "ZAP70", "LAT", "LCP2", "ITK", "PLCG1",
        "NFATC1", "NFATC2", "NFKB1", "RELA", "FOS", "JUN", "IL2", "IFNG", "TNF",
        "PDCD1", "CTLA4", "LAG3", "TIGIT", "TOX", "MKI67", "BCL2", "FAS",
    }
    variance_rank = np.argsort(np.var(log_fc_all, axis=0))[::-1]
    selected = list(variance_rank[:n_genes])
    gene_pos = {g.upper(): i for i, g in enumerate(gene_names)}
    selected.extend(gene_pos[g] for g in forced if g in gene_pos)
    selected = np.asarray(sorted(set(selected)), dtype=int)

    effects = log_fc_all[:, selected].reshape(len(conditions), len(targets), -1)
    uncertainty = lfc_se_all[:, selected].reshape(len(conditions), len(targets), -1)
    baseline = np.log2(1.0 + base_mean_all[:, selected].reshape(len(conditions), len(targets), -1).mean(axis=1))
    counts = np.zeros((len(conditions), len(targets)), dtype=np.int32)
    for ci, condition in enumerate(conditions):
        for ti, target in enumerate(targets):
            source = obs[(obs.target_contrast_gene_name == target) & (obs.culture_condition == condition)].iloc[0]
            counts[ci, ti] = int(source.n_cells_target)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        genes=gene_names[selected], conditions=np.asarray(conditions), targets=np.asarray(targets),
        baseline=baseline, effects=effects, uncertainty=uncertainty, counts=counts,
        effect_unit=np.asarray("log2_fold_change"), source=np.asarray("GSE314342"),
    )
    summary = {
        "source": "GSE314342 / Primary Human CD4+ T Cell Perturb-seq v1.0",
        "model_targets": len(targets), "minimum_required_targets": 200,
        "measured_response_genes": len(selected), "conditions": conditions,
        "input_perturbation_condition_rows": int(a.n_obs), "input_measured_genes": int(a.n_vars),
        "qc": "on-target significant; >=2 guides; excludes low-expression, cis/distal off-target and single-guide estimates",
        "effect_unit": "log2 fold change", "clinical_use": False,
    }
    out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    a.file.close()
    return summary
