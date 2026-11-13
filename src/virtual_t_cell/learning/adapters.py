"""Import repository experimental summaries without treating predictions as data."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cli import NETWORK, PATHWAYS
from .atlas import file_sha256
from .schema import content_hash, validate_dataset


DEFAULT_OUTCOME_GENES = ("IL2", "IFNG", "TNF", "PDCD1", "TOX", "MKI67", "GZMB", "CD3D", "FOXP3")


def _scalar_text(archive, key: str) -> str:
    if key not in archive:
        return ""
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"archive {key} must be scalar text")
    text = str(value.reshape(-1)[0]).strip()
    if not text:
        raise ValueError(f"archive {key} must be nonempty text")
    return text


def import_reference_npz(path: str | Path, *, organ: str, cell_type: str,
                         outcome_genes: list[str] | tuple[str, ...] | None = None,
                         effect_unit_override: str | None = None) -> tuple[list[dict], dict]:
    """Import one effect per target/condition from a measured summary archive.

    Inputs are fixed target annotations from the existing repository. In
    particular, the archive's baseline is NOT used: GSE314342 baseline derives
    from DE baseMean values across target contrasts and is not established as
    pre-intervention expression. Most targets lack these sparse annotations.
    No raw cells, donor replicates, dose effects or organ-specific experiments
    are created by this adapter.
    """
    if not isinstance(organ, str) or not organ.strip() or not isinstance(cell_type, str) or not cell_type.strip():
        raise ValueError("organ and cell_type must be explicitly supplied from source provenance")
    organ, cell_type = organ.strip().lower(), cell_type.strip()
    if effect_unit_override is not None and (not isinstance(effect_unit_override, str) or not effect_unit_override.strip()):
        raise ValueError("effect_unit_override must be nonempty text")
    requested = list(DEFAULT_OUTCOME_GENES if outcome_genes is None else outcome_genes)
    if not requested or any(not isinstance(gene, str) or not gene.strip() for gene in requested) or len(set(requested)) != len(requested):
        raise ValueError("outcome_genes must be a nonempty unique list of gene identifiers")
    checksum = file_sha256(path)
    feature_definition_hash = content_hash({"pathways": PATHWAYS, "network": NETWORK, "readouts": requested})
    with np.load(path, allow_pickle=False) as archive:
        required = {"genes", "conditions", "targets", "effects", "uncertainty"}
        if not required.issubset(archive.files):
            raise ValueError(f"missing archive fields: {sorted(required - set(archive.files))}")
        source = _scalar_text(archive, "source")
        if not source:
            raise ValueError("archive source provenance is required; do not infer it from a filename")
        unit = _scalar_text(archive, "effect_unit")
        if unit and effect_unit_override is not None and unit != effect_unit_override.strip():
            raise ValueError("effect_unit_override cannot replace a conflicting declared unit")
        if not unit:
            if effect_unit_override is None:
                raise ValueError("archive has no explicit effect_unit; supply a verified effect_unit_override")
            unit = effect_unit_override.strip()
        # This archive contains cultured, blood-derived CD4 cells. Restrict tissue
        # labels instead of allowing the caller to turn them into liver/spleen.
        if source == "GSE314342":
            if organ not in {"blood", "peripheral_blood"}:
                raise ValueError("GSE314342 is blood-derived CD4 evidence; it cannot be relabelled as liver or spleen")
            organ = "blood"
            if "cd4" not in cell_type.lower():
                raise ValueError("GSE314342 requires an explicit CD4 cell_type label")
        # Mixed-source fallback archives need per-entry provenance first.
        if "effect_source" in archive:
            raise ValueError("mixed-source effect archives require a per-entry provenance adapter")
        genes = [str(value) for value in archive["genes"]]
        targets = [str(value) for value in archive["targets"]]
        conditions = [str(value) for value in archive["conditions"]]
        for name, values in (("genes", genes), ("targets", targets), ("conditions", conditions)):
            if not values or len(values) != len(set(values)) or any(not value.strip() for value in values):
                raise ValueError(f"archive {name} must be nonempty and unique")
        effects = np.asarray(archive["effects"], dtype=float)
        uncertainty = np.asarray(archive["uncertainty"], dtype=float)
        shape = (len(conditions), len(targets), len(genes))
        if effects.shape != shape or uncertainty.shape != shape:
            raise ValueError("effect/uncertainty arrays must have condition x target x gene shape")
        if np.any(np.isfinite(uncertainty) & (uncertainty < 0)):
            raise ValueError("negative measurement standard errors in archive")
        gene_positions = {gene: index for index, gene in enumerate(genes)}
        present = [gene for gene in requested if gene in gene_positions]
        missing = [gene for gene in requested if gene not in gene_positions]
        if not present:
            raise ValueError("none of the requested outcome genes are present")
        records = []
        masked = 0
        unannotated = 0
        for target_index, target in enumerate(targets):
            features = {"target_in_" + name.lower(): float(target.upper() in members) for name, members in sorted(PATHWAYS.items())}
            features.update({"target_is_readout_" + gene: float(target == gene) for gene in requested})
            features["curated_network_degree"] = float(len(NETWORK.get(target.upper(), [])))
            if not any(features.values()):
                unannotated += 1
            for condition_index, condition in enumerate(conditions):
                outcomes, noise = {}, {}
                for gene in present:
                    gene_index = gene_positions[gene]
                    effect = float(effects[condition_index, target_index, gene_index])
                    se = float(uncertainty[condition_index, target_index, gene_index])
                    if not np.isfinite(effect) or not np.isfinite(se) or se <= 0:
                        masked += 1
                        continue
                    outcomes[gene], noise[gene] = effect, se
                if not outcomes:
                    continue
                source_record_id = f"{source}:{target}:{condition}"
                record = {"observation_id": source_record_id, "source_record_id": source_record_id,
                          "experiment_id": source_record_id, "group_id": f"{source}:{target}",
                          "organ": organ, "cell_type": cell_type, "condition": condition,
                          "intervention": "CRISPRi:" + target if source == "GSE314342" else "source_defined_perturbation:" + target,
                          "target": target, "source": source, "evidence_type": "experimental_summary",
                          "features": dict(features), "outcomes": outcomes,
                          "outcome_units": {gene: unit for gene in outcomes}, "measurement_sd": noise,
                          "provenance": {"archive_sha256": checksum, "source_record_id": source_record_id,
                                         "source_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={source}" if source.startswith("GSE") else None,
                                         "effect_unit_declared": "effect_unit" in archive,
                                         "feature_definition_hash": feature_definition_hash,
                                         "feature_source": "fixed repository PATHWAYS/NETWORK and target/readout identity; no response-derived features",
                                         "observed_mask": "finite effect and finite strictly positive measurement SE"}}
                if source == "GSE314342":
                    record["species"] = "human"
                    record["provenance"]["tissue_source_url"] = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9393885"
                records.append(record)
    validate_dataset(records)
    report = {"source": source, "archive_sha256": checksum, "records": len(records),
              "targets": len(targets), "conditions": conditions, "organ": organ, "cell_type": cell_type,
              "evidence_type": "experimental_summary", "effect_unit": unit, "outcome_genes": present,
              "missing_requested_genes": missing, "masked_outcome_entries": masked,
              "targets_without_feature_annotations": unannotated,
              "feature_schema_hash": content_hash(sorted(records[0]["features"])),
              "feature_definition_hash": feature_definition_hash,
              "data_role": "retrospective replay of previously selected published effect summaries",
              "limitations": ["Not raw single-cell training, new sequencing, or a prospective biological validation.",
                              "The distributed archive was target/gene-selected using original study results; this is a conditional archive-level benchmark.",
                              "Measurement SE does not measure extrapolation uncertainty or prove interval calibration.",
                              "Sparse fixed annotations are only a baseline; unannotated targets can have identical features.",
                              "Archive baseline is excluded: DE baseMean is not established pre-intervention expression.",
                              "No individual donor, dose, or biological time is invented from pooled summaries."]}
    return records, report
