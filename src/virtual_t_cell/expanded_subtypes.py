"""Expand the DICE subtype layer with HPA/Monaco sorted T-cell references."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


MONACO_TYPES = {
    "CD8_central_memory": "Central memory CD8 T-cell",
    "CD8_effector_memory": "Effector memory CD8 T-cell",
    "MAIT": "MAIT T-cell",
    "Th1": "Memory CD4 T-cell Th1",
    "Th1_17": "Memory CD4 T-cell Th1/Th17",
    "Th17": "Memory CD4 T-cell Th17",
    "Th2": "Memory CD4 T-cell Th2",
    "Tfh": "Memory CD4 T-cell TFH",
    "CD4_naive": "naive CD4 T-cell",
    "CD8_naive": "naive CD8 T-cell",
    "gdT_non_Vd2": "Non-Vd2 gdTCR",
    "Treg": "T-reg",
    "CD4_TEMRA": "Terminal effector memory CD4 T-cell",
    "CD8_TEMRA": "Terminal effector memory CD8 T-cell",
    "gdT_Vd2": "Vd2 gdTCR",
}

HPA_SAMPLE_TYPES = {
    "gdT": "gdT-cell",
    "MAIT": "MAIT T-cell",
    "CD4_memory": "memory CD4 T-cell",
    "CD8_memory": "memory CD8 T-cell",
    "CD4_naive": "naive CD4 T-cell",
    "CD8_naive": "naive CD8 T-cell",
    "Treg": "T-reg",
}

GSE80306_TYPES = {
    "CD8_TEMRA": "TEMRA",
    "CD8_effector_memory": "EM",
    "CD8_naive": "N",
    "CD8_virtual_naive_memory": "TMNP",
    "CD8_central_memory": "CM",
}

GSE135390_TYPES = {
    "Th17": "Th17", "Th22": "Th22", "Th2": "Th2", "Th1_17": "Th1/17",
    "Th1": "Th1", "CD4_naive": "Naive", "Treg_Th17": "Treg17",
    "Treg_Th22": "Treg22", "Treg_Th2": "Treg2",
    "Treg_Th1_17": "Treg1/17", "Treg_Th1": "Treg1",
}


def _read_zipped_tsv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".tsv"))
        with archive.open(member) as handle:
            return pd.read_csv(handle, sep="\t", low_memory=False)


def build_expanded_subtype_model(base_model: Path, monaco_zip: Path, hpa_samples_zip: Path,
                                 gse80306_counts: Path, gse135390_expression: Path,
                                 out: Path) -> dict:
    """Fuse Monaco profiles into DICE and add memory/innate-like T-cell types."""
    base = np.load(base_model, allow_pickle=False)
    genes = base["genes"].astype(str)
    names = base["subtypes"].astype(str).tolist()
    tpm = base["subtype_reference_tpm"].astype(np.float32)
    offset = base["subtype_baseline_offset"].astype(np.float32)
    source_count = np.ones(len(names), dtype=np.int16)
    donor_count = np.zeros(len(names), dtype=np.int16)
    donor_log1p_sd = np.full((len(names), len(genes)), np.nan, dtype=np.float32)

    frame = _read_zipped_tsv(monaco_zip)
    frame["Gene name"] = frame["Gene name"].astype(str).str.upper()
    frame = frame[frame["Immune cell"].isin(MONACO_TYPES.values())]
    pivot = frame.pivot_table(index="Gene name", columns="Immune cell", values="TPM", aggfunc="mean")
    aligned = pivot.reindex(genes).fillna(0)
    monaco_profiles = np.vstack([
        aligned[source].to_numpy(np.float32) for source in MONACO_TYPES.values()
    ])
    monaco_log = np.log1p(monaco_profiles)
    monaco_offset = np.clip(monaco_log - monaco_log.mean(axis=0, keepdims=True), -2.0, 2.0)

    for row, name in enumerate(MONACO_TYPES):
        if name in names:
            idx = names.index(name)
            # Equal-source consensus avoids treating either cohort as ground truth.
            tpm[idx] = (tpm[idx] + monaco_profiles[row]) / 2
            offset[idx] = (offset[idx] + monaco_offset[row]) / 2
            source_count[idx] += 1
        else:
            names.append(name)
            tpm = np.vstack([tpm, monaco_profiles[row]])
            offset = np.vstack([offset, monaco_offset[row]])
            source_count = np.append(source_count, 1)
            donor_count = np.append(donor_count, 0)
            donor_log1p_sd = np.vstack([donor_log1p_sd, np.full((1, len(genes)), np.nan, np.float32)])

    samples = _read_zipped_tsv(hpa_samples_zip)
    samples["Gene name"] = samples["Gene name"].astype(str).str.upper()
    samples = samples[samples["Immune cell"].isin(HPA_SAMPLE_TYPES.values())]
    sample_means = samples.groupby(["Gene name", "Immune cell"], as_index=False).TPM.mean().pivot(
        index="Gene name", columns="Immune cell", values="TPM").reindex(genes).fillna(0)
    donor_means = samples.groupby(["Gene name", "Immune cell", "Sample ID"], as_index=False).TPM.mean()
    sample_sd = donor_means.assign(log1p=lambda x: np.log1p(x.TPM)).groupby(
        ["Gene name", "Immune cell"]).log1p.std().unstack().reindex(genes)
    hpa_profiles = np.vstack([sample_means[source].to_numpy(np.float32) for source in HPA_SAMPLE_TYPES.values()])
    hpa_log = np.log1p(hpa_profiles)
    hpa_offset = np.clip(hpa_log - hpa_log.mean(axis=0, keepdims=True), -2.0, 2.0)
    for row, name in enumerate(HPA_SAMPLE_TYPES):
        source = HPA_SAMPLE_TYPES[name]
        sd = sample_sd[source].to_numpy(np.float32)
        count = int(samples.loc[samples["Immune cell"].eq(source), "Sample ID"].nunique())
        if name in names:
            idx = names.index(name)
            n = float(source_count[idx])
            tpm[idx] = (tpm[idx] * n + hpa_profiles[row]) / (n + 1)
            offset[idx] = (offset[idx] * n + hpa_offset[row]) / (n + 1)
            source_count[idx] += 1
            donor_count[idx] = count
            donor_log1p_sd[idx] = sd
        else:
            names.append(name)
            tpm = np.vstack([tpm, hpa_profiles[row]])
            offset = np.vstack([offset, hpa_offset[row]])
            source_count = np.append(source_count, 1)
            donor_count = np.append(donor_count, count)
            donor_log1p_sd = np.vstack([donor_log1p_sd, sd])

    counts = pd.read_csv(gse80306_counts, sep="\t")
    counts["gene"] = counts.gene.astype(str).str.replace(r"\.\d+$", "", regex=True).str.upper()
    counts = counts.groupby("gene", as_index=False).sum(numeric_only=True).set_index("gene")
    cpm = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1) * 1_000_000
    gse_count = np.zeros(len(names), dtype=np.int16)
    gse_sd = np.full((len(names), len(genes)), np.nan, dtype=np.float32)
    group_profiles = []
    group_sds = []
    group_counts = []
    for suffix in GSE80306_TYPES.values():
        columns = [column for column in cpm if column.upper().endswith("_" + suffix)]
        aligned_group = cpm[columns].reindex(genes).fillna(0)
        group_profiles.append(aligned_group.mean(axis=1).to_numpy(np.float32))
        group_sds.append(np.log1p(aligned_group).std(axis=1).to_numpy(np.float32))
        group_counts.append(len(columns))
    gse_profiles = np.vstack(group_profiles)
    gse_log = np.log1p(gse_profiles)
    gse_offset = np.clip(gse_log - gse_log.mean(axis=0, keepdims=True), -2.0, 2.0)
    for row, name in enumerate(GSE80306_TYPES):
        if name in names:
            idx = names.index(name)
            n = float(source_count[idx])
            tpm[idx] = (tpm[idx] * n + gse_profiles[row]) / (n + 1)
            offset[idx] = (offset[idx] * n + gse_offset[row]) / (n + 1)
            source_count[idx] += 1
            gse_count[idx] = group_counts[row]
            gse_sd[idx] = group_sds[row]
        else:
            names.append(name)
            tpm = np.vstack([tpm, gse_profiles[row]])
            offset = np.vstack([offset, gse_offset[row]])
            source_count = np.append(source_count, 1)
            donor_count = np.append(donor_count, 0)
            donor_log1p_sd = np.vstack([donor_log1p_sd, np.full((1, len(genes)), np.nan, np.float32)])
            gse_count = np.append(gse_count, group_counts[row])
            gse_sd = np.vstack([gse_sd, group_sds[row]])

    cd4 = pd.read_csv(gse135390_expression)
    labels = cd4.iloc[0, 1:].astype(str)
    cd4 = cd4.iloc[1:].copy()
    cd4.index = cd4.iloc[:, 0].astype(str).str.upper()
    cd4 = cd4.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    cd4.columns = labels
    cd4 = cd4.groupby(level=0).mean().reindex(genes)
    cd4_profiles, cd4_sds, cd4_counts = [], [], []
    for suffix in GSE135390_TYPES.values():
        columns = [column for column in cd4 if column.endswith("_" + suffix)]
        cd4_profiles.append(cd4[columns].mean(axis=1).to_numpy(np.float32))
        cd4_sds.append(cd4[columns].std(axis=1).to_numpy(np.float32))
        cd4_counts.append(len(columns))
    cd4_profiles = np.vstack(cd4_profiles)
    valid = np.isfinite(cd4_profiles)
    denominator = valid.sum(axis=0, keepdims=True)
    cd4_mean = np.divide(np.nansum(cd4_profiles, axis=0, keepdims=True), denominator,
                         out=np.zeros((1, cd4_profiles.shape[1]), dtype=np.float32), where=denominator > 0)
    cd4_centered = cd4_profiles - cd4_mean
    cd4_offset = np.clip(np.nan_to_num(cd4_centered), -2.0, 2.0)
    gse135_count = np.zeros(len(names), dtype=np.int16)
    gse135_expression = np.full((len(names), len(genes)), np.nan, dtype=np.float32)
    gse135_sd = np.full_like(gse135_expression, np.nan)
    for row, name in enumerate(GSE135390_TYPES):
        if name in names:
            idx = names.index(name)
            n = float(source_count[idx])
            offset[idx] = (offset[idx] * n + cd4_offset[row]) / (n + 1)
            source_count[idx] += 1
            gse135_count[idx] = cd4_counts[row]
            gse135_expression[idx] = cd4_profiles[row]
            gse135_sd[idx] = cd4_sds[row]
        else:
            names.append(name)
            tpm = np.vstack([tpm, np.zeros((1, len(genes)), np.float32)])
            offset = np.vstack([offset, cd4_offset[row]])
            source_count = np.append(source_count, 1)
            donor_count = np.append(donor_count, 0)
            donor_log1p_sd = np.vstack([donor_log1p_sd, np.full((1, len(genes)), np.nan, np.float32)])
            gse_count = np.append(gse_count, 0)
            gse_sd = np.vstack([gse_sd, np.full((1, len(genes)), np.nan, np.float32)])
            gse135_count = np.append(gse135_count, cd4_counts[row])
            gse135_expression = np.vstack([gse135_expression, cd4_profiles[row]])
            gse135_sd = np.vstack([gse135_sd, cd4_sds[row]])

    arrays = {key: base[key] for key in base.files}
    arrays.update({
        "subtype_version": np.asarray("dice_monaco_sorted_tcell_v2"),
        "subtype_sources": np.asarray(["DICE-DB_build_2022-02-25", "HPA_Monaco_v24", "HPA_immune_cell_samples_v25", "GSE80306", "GSE135390"]),
        "subtypes": np.asarray(names),
        "subtype_reference_tpm": tpm.astype(np.float32),
        "subtype_baseline_offset": offset.astype(np.float32),
        "subtype_reference_source_count": source_count,
        "subtype_donor_count": donor_count,
        "subtype_donor_log1p_tpm_sd": donor_log1p_sd,
        "gse80306_replicate_count": gse_count,
        "gse80306_log1p_cpm_sd": gse_sd,
        "gse135390_replicate_count": gse135_count,
        "gse135390_normalized_expression": gse135_expression,
        "gse135390_normalized_expression_sd": gse135_sd,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    summary = {
        "model": "expanded sorted-human-T-cell subtype context",
        "sources": ["DICE Database", "Human Protein Atlas / Monaco", "Human Protein Atlas donor samples", "GSE80306", "GSE135390"],
        "subtypes": names,
        "subtype_count": len(names),
        "model_genes": len(genes),
        "genes_with_subtype_expression": int((tpm.max(axis=0) > 0).sum()),
        "multi_source_subtypes": [name for name, count in zip(names, source_count) if count > 1],
        "donor_supported_subtypes": {name: int(count) for name, count in zip(names, donor_count) if count > 0},
        "gse80306_supported_subtypes": {name: int(count) for name, count in zip(names, gse_count) if count > 0},
        "gse135390_supported_subtypes": {name: int(count) for name, count in zip(names, gse135_count) if count > 0},
        "perturbation_effects_are_subtype_specific": False,
        "clinical_use": False,
    }
    out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
