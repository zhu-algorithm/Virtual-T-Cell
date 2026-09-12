"""Construct a compact cross-cohort T-cell multi-omics evidence model."""
from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _gene_from_fasta(value: object) -> str:
    match = re.search(r"(?:^|\s)GN=([^\s;]+)", str(value))
    return match.group(1).upper() if match else ""


def _manifest_csv(path: Path) -> tuple[Path, int]:
    if path.suffix.lower() != ".zip":
        return path, 0
    target = path.parent / "epic_manifest.csv"
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as source, target.open("wb") as destination:
            destination.write(source.read())
    return target, 0


def _read_manifest(path: Path) -> pd.DataFrame:
    path, _ = _manifest_csv(path)
    skip = 0
    with path.open("rt", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.lstrip('"').startswith("IlmnID"):
                break
            skip += 1
    return pd.read_csv(path, skiprows=skip, low_memory=False,
                       usecols=["IlmnID", "UCSC_RefGene_Name", "UCSC_RefGene_Group"])


def _protein_evidence(path: Path, hgnc_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame["gene"] = frame["Fasta headers"].map(_gene_from_fasta)
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    uniprot_to_gene = {}
    for row in hgnc[["symbol", "uniprot_ids"]].dropna().itertuples(index=False):
        for accession in str(row.uniprot_ids).split("|"):
            uniprot_to_gene[accession] = str(row.symbol).upper()
    def protein_id_gene(value: object) -> str:
        for item in str(value).split(";"):
            accession = item.split("|")[1] if item.count("|") >= 2 else item
            accession = accession.split("-")[0]
            if accession in uniprot_to_gene:
                return uniprot_to_gene[accession]
        return ""
    missing = frame.gene.eq("")
    frame.loc[missing, "gene"] = frame.loc[missing, "Majority protein IDs"].map(protein_id_gene)
    frame = frame[frame.gene.ne("")]
    activated = [c for c in frame if c.startswith("LFQ intensity Activated_")]
    naive = [c for c in frame if c.startswith("LFQ intensity Naive_")]
    a = frame[activated].replace(0, np.nan)
    n = frame[naive].replace(0, np.nan)
    frame["protein_activation_log2fc"] = np.log2(a.mean(axis=1) + 1) - np.log2(n.mean(axis=1) + 1)
    frame["protein_detection_fraction"] = pd.concat([a, n], axis=1).notna().mean(axis=1)
    return frame.groupby("gene", as_index=False).agg(
        protein_activation_log2fc=("protein_activation_log2fc", "mean"),
        protein_detection_fraction=("protein_detection_fraction", "max"))


def _genomic_evidence(path: Path, hgnc_path: Path) -> pd.DataFrame:
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    mapping = dict(zip(hgnc["ensembl_gene_id"].fillna(""), hgnc["symbol"].fillna("")))
    frame = pd.read_csv(path, sep="\t")
    frame["gene"] = frame["molecular_trait_id"].str.split(".").str[0].map(mapping).fillna("").str.upper()
    frame = frame[frame.gene.ne("")].sort_values("pvalue").drop_duplicates("gene")
    frame["eqtl_confidence"] = np.clip(-np.log10(frame.pvalue.clip(lower=1e-300)) / 20, 0, 1)
    return frame[["gene", "variant", "beta", "pvalue", "eqtl_confidence"]].rename(
        columns={"variant": "eqtl_variant", "beta": "eqtl_beta", "pvalue": "eqtl_pvalue"})


def _methylation_evidence(matrix_path: Path, manifest_path: Path) -> pd.DataFrame:
    manifest = _read_manifest(manifest_path)
    promoter = manifest.UCSC_RefGene_Group.fillna("").str.contains("TSS200|TSS1500|5'UTR|1stExon")
    manifest = manifest[promoter & manifest.UCSC_RefGene_Name.notna()].set_index("IlmnID")
    rows = []
    for chunk in pd.read_csv(matrix_path, sep="\t", chunksize=4000, low_memory=False):
        chunk.columns = [str(c).strip('"') for c in chunk.columns]
        chunk["ID_REF"] = chunk["ID_REF"].astype(str).str.strip('"')
        value_cols = [c for c in chunk if c != "ID_REF" and "Detection Pval" not in c]
        pval_cols = [f"{c} Detection Pval" for c in value_cols]
        values = chunk[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
        pvals = chunk[pval_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        values[pvals > 0.01] = np.nan
        mean_beta = np.nanmean(values, axis=1)
        for probe, beta in zip(chunk.ID_REF, mean_beta):
            if probe in manifest.index and np.isfinite(beta):
                genes = str(manifest.at[probe, "UCSC_RefGene_Name"]).split(";")
                rows.extend((gene.upper(), float(beta)) for gene in set(genes) if gene)
    result = pd.DataFrame(rows, columns=["gene", "promoter_methylation"])
    return result.groupby("gene", as_index=False).agg(
        promoter_methylation=("promoter_methylation", "mean"),
        methylation_probe_count=("promoter_methylation", "size"))


def build_multiomics_model(base_model: Path, protein_groups: Path, eqtl: Path,
                            methylation: Path, epic_manifest: Path, hgnc: Path,
                            out: Path) -> dict:
    base = np.load(base_model, allow_pickle=False)
    genes = base["genes"].astype(str)
    table = pd.DataFrame({"gene": genes})
    table = table.merge(_protein_evidence(protein_groups, hgnc), on="gene", how="left")
    table = table.merge(_genomic_evidence(eqtl, hgnc), on="gene", how="left")
    table = table.merge(_methylation_evidence(methylation, epic_manifest), on="gene", how="left")
    arrays = {key: base[key] for key in base.files}
    arrays.update({
        "multiomics_version": np.asarray("cross_cohort_v1"),
        "protein_source": np.asarray("PXD021250"),
        "genomics_source": np.asarray("BLUEPRINT_QTD000031"),
        "methylation_source": np.asarray("GSE174666"),
        "protein_activation_log2fc": table.protein_activation_log2fc.fillna(np.nan).to_numpy(np.float32),
        "protein_detection_fraction": table.protein_detection_fraction.fillna(0).to_numpy(np.float32),
        "eqtl_variant": table.eqtl_variant.fillna("").to_numpy(str),
        "eqtl_beta": table.eqtl_beta.fillna(np.nan).to_numpy(np.float32),
        "eqtl_pvalue": table.eqtl_pvalue.fillna(np.nan).to_numpy(np.float64),
        "eqtl_confidence": table.eqtl_confidence.fillna(0).to_numpy(np.float32),
        "promoter_methylation": table.promoter_methylation.fillna(np.nan).to_numpy(np.float32),
        "methylation_probe_count": table.methylation_probe_count.fillna(0).to_numpy(np.int32),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    summary = {
        "model": "cross-cohort T-cell multi-omics evidence fusion",
        "genes": len(genes),
        "protein_genes": int(table.protein_detection_fraction.notna().sum()),
        "genomic_eGenes": int(table.eqtl_variant.notna().sum()),
        "methylation_genes": int(table.promoter_methylation.notna().sum()),
        "sources": ["PXD021250", "BLUEPRINT QTD000031", "GSE174666"],
        "paired_samples": False,
        "clinical_use": False,
    }
    out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
