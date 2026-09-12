"""Add reference expression profiles for sorted human T-cell subtypes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DICE_SUBTYPES = {
    "CD4_naive": ("T cell, CD4, naive",),
    "CD4_naive_activated": ("T cell, CD4, naive [activated]",),
    "Treg": ("T cell, CD4, naive TREG", "T cell, CD4, memory TREG"),
    "Treg_naive": ("T cell, CD4, naive TREG",),
    "Treg_memory": ("T cell, CD4, memory TREG",),
    "Tfh": ("T cell, CD4, TFH",),
    "Th1": ("T cell, CD4, TH1",),
    "Th1_17": ("T cell, CD4, TH1/17",),
    "Th17": ("T cell, CD4, TH17",),
    "Th2": ("T cell, CD4, TH2",),
    "CD8": ("T cell, CD8, naive", "T cell, CD8, naive [activated]"),
    "CD8_naive": ("T cell, CD8, naive",),
    "CD8_naive_activated": ("T cell, CD8, naive [activated]",),
}


def build_subtype_model(base_model: Path, dice_tpm: Path, hgnc: Path, out: Path) -> dict:
    """Copy a perturbation model and append DICE subtype reference arrays.

    DICE values are sorted-population mean TPM.  The stored offset is centered
    gene-by-gene across T-cell types and clipped so it can safely condition the
    perturbation baseline without pretending that DICE measured perturbations.
    """
    base = np.load(base_model, allow_pickle=False)
    genes = base["genes"].astype(str)
    hgnc_frame = pd.read_csv(hgnc, sep="\t", dtype=str, low_memory=False)
    gene_map = dict(zip(
        hgnc_frame["ensembl_gene_id"].fillna("").str.split(".").str[0],
        hgnc_frame["symbol"].fillna("").str.upper(),
    ))
    source_columns = list(dict.fromkeys(column for columns in DICE_SUBTYPES.values() for column in columns))
    dice = pd.read_csv(dice_tpm, usecols=["gene", *source_columns])
    dice["symbol"] = dice["gene"].astype(str).str.split(".").str[0].map(gene_map).fillna("")
    dice = dice[dice.symbol.ne("")].groupby("symbol", as_index=False)[source_columns].mean()
    aligned = pd.DataFrame({"symbol": genes}).merge(dice, on="symbol", how="left")
    subtype_names = list(DICE_SUBTYPES)
    tpm = np.vstack([
        aligned[list(DICE_SUBTYPES[name])].fillna(0).mean(axis=1).to_numpy(np.float32)
        for name in subtype_names
    ])
    log_tpm = np.log1p(tpm)
    offset = np.clip(log_tpm - log_tpm.mean(axis=0, keepdims=True), -2.0, 2.0).astype(np.float32)
    arrays = {key: base[key] for key in base.files}
    arrays.update({
        "subtype_version": np.asarray("dice_sorted_tcell_v1"),
        "subtype_source": np.asarray("DICE-DB_build_2022-02-25"),
        "subtypes": np.asarray(subtype_names),
        "subtype_reference_tpm": tpm,
        "subtype_baseline_offset": offset,
        "subtype_baseline_weight": np.asarray(0.5, dtype=np.float32),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    shared = (tpm.max(axis=0) > 0)
    summary = {
        "model": "T-cell perturbation model with sorted-cell subtype context",
        "subtype_source": "DICE Database mean TPM",
        "subtypes": subtype_names,
        "subtype_count": len(subtype_names),
        "model_genes": len(genes),
        "genes_with_subtype_expression": int(shared.sum()),
        "perturbation_effects_are_subtype_specific": False,
        "clinical_use": False,
    }
    out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
