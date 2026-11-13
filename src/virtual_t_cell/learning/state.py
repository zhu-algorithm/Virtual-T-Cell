"""Donor-aware empirical belief about observed tissue states, separate from response models."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import canonical_json, content_hash


def summarize_atlas_states(records: list[dict]) -> dict:
    """Recompute an empirical state belief from the CURRENT atlas snapshot.

    Samples are averaged within each donor before calculating between-donor
    variability. No confidence interval is invented from the number of cells.
    Biological time, species, measured gene panel, and source remain separate.
    This estimates measured states; it does not identify intervention effects.
    """
    contexts = {}
    seen_groups = set()
    for record in records:
        if not record.get("state_snapshot") or record.get("response_training_eligible") is not False:
            raise ValueError("state belief requires current atlas snapshots, not response labels")
        if record.get("evidence_type") not in {"observational", "synthetic"}:
            raise ValueError("state snapshots must be explicitly observational or synthetic")
        identity = record["source_record_id"]
        if identity in seen_groups:
            raise ValueError("overlapping snapshots cannot count as independent evidence")
        seen_groups.add(identity)
        context = {name: record.get(name) for name in (
            "source", "evidence_type", "species", "organ", "cell_type", "condition",
            "biological_time_hours", "gene_panel_id", "intervention", "dose", "dose_unit")}
        key = canonical_json(context)
        group = contexts.setdefault(key, {"context": context, "donors": defaultdict(list),
                                          "n_cells": 0, "samples": set(), "units": record["outcome_units"]})
        if group["units"] != record["outcome_units"]:
            raise ValueError("state outcome panels and units must match within context")
        values = np.asarray([record["outcomes"][gene] for gene in sorted(group["units"])], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("state outcomes must be finite")
        group["donors"][record["donor_id"]].append(values)
        group["n_cells"] += int(record["n_cells"])
        group["samples"].add(record["sample_id"])
    profiles = []
    for key in sorted(contexts):
        group = contexts[key]
        genes = sorted(group["units"])
        donor_values = np.stack([np.mean(samples, axis=0) for samples in group["donors"].values()])
        donors = len(donor_values)
        mean = donor_values.mean(axis=0)
        sd = donor_values.std(axis=0, ddof=1) if donors > 1 else None
        profiles.append({"context": group["context"], "n_donors": donors, "n_samples": len(group["samples"]),
                         "n_cells": group["n_cells"], "gene_means": dict(zip(genes, mean.tolist())),
                         "between_donor_sd": dict(zip(genes, sd.tolist())) if sd is not None else None,
                         "mean_standard_error": dict(zip(genes, (sd / np.sqrt(donors)).tolist())) if sd is not None else None,
                         "outcome_units": group["units"],
                         "uncertainty_status": "empirical_between_donor_variation" if donors > 1 else "insufficient_independent_donors"})
    return {"state_version": content_hash(sorted(record["snapshot_hash"] for record in records)),
            "snapshot_count": len(records), "profiles": profiles,
            "response_model_updated": False,
            "limitations": ["Descriptive donor-weighted means; not a trained latent-state foundation model.",
                            "Empirical standard errors assume independent donors; systematic batch effects remain.",
                            "Contexts with one donor have unknown between-donor uncertainty.",
                            "Recompute from current snapshots; do not append overlapping snapshot histories as new evidence."]}
