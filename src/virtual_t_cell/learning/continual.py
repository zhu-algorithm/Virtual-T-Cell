"""Independent evaluation and replay-based, explicitly gated model updates.

An accepted proposal is still only a proposal: this module never overwrites the
current model, promotes a model, or turns observational atlas rows into responses.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .schema import content_hash, evidence_identity, validate_dataset


_LIMITATIONS = [
    "Group independence follows supplied group_id values; it is not proof of donor or study independence.",
    "Retrospective prediction metrics do not establish prospective experiment value or causality.",
    "Gaussian interval coverage is a diagnostic, not a claim of calibrated biological uncertainty.",
    "Repeated reuse of one validation set can overfit the gate; reserve an untouched final test set.",
    "A gate decision never promotes or overwrites the current model automatically.",
]


def _overlap_guard(metadata: dict, records: list[dict]) -> None:
    checks = (
        ("observation_id", "training_ids"),
        ("group_id", "training_groups"),
        ("experiment_id", "training_experiments"),
    )
    for field, stored in checks:
        overlap = set(metadata.get(stored, [])) & {record[field] for record in records}
        if overlap:
            raise ValueError(f"training/validation {field} overlap: {sorted(overlap)[:5]}")
    if "training_source_identities" in metadata:
        overlap = set(metadata["training_source_identities"]) & {
            evidence_identity(record) for record in records
        }
        if overlap:
            raise ValueError("training/validation original source identity overlap")


def _evidence_compatible(metadata: dict, records: list[dict]) -> None:
    model_kinds = set(metadata.get("evidence_types", []))
    data_kinds = {record["evidence_type"] for record in records}
    if model_kinds and (("synthetic" in model_kinds) != ("synthetic" in data_kinds)):
        raise ValueError("synthetic and experimental model/evaluation evidence cannot be mixed")


def _group_mean(values: np.ndarray, records: list[dict]) -> float:
    grouped: dict[str, list[float]] = {}
    for value, record in zip(values, records):
        grouped.setdefault(record["group_id"], []).append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def _output_metrics(model, records: list[dict], prediction: dict, outcome: str,
                    indices: list[int]) -> dict | None:
    indices = [index for index in indices if outcome in records[index]["outcomes"]]
    if not indices:
        return None
    column = prediction["outcome_names"].index(outcome)
    rows = [records[index] for index in indices]
    target = np.asarray([record["outcomes"][outcome] for record in rows], dtype=float)
    estimates = np.asarray(prediction["mean"], dtype=float)[indices, column]
    variance = (np.asarray(prediction["epistemic_variance"], dtype=float)[indices, column]
                + np.asarray(prediction["noise_variance"], dtype=float)[indices, column])
    if not np.isfinite(estimates).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("model returned nonfinite predictions or invalid variance")
    baseline = model.metadata.get("outcome_means", {}).get(outcome)
    if baseline is None or not math.isfinite(float(baseline)):
        raise ValueError(f"model must retain its training outcome mean for {outcome}")
    errors = estimates - target
    squared = errors ** 2
    baseline_squared = (float(baseline) - target) ** 2
    group_rmse = math.sqrt(_group_mean(squared, rows))
    group_baseline_rmse = math.sqrt(_group_mean(baseline_squared, rows))
    safe_variance = np.maximum(variance, 1e-12)
    nll = 0.5 * (np.log(2.0 * math.pi * safe_variance) + squared / safe_variance)
    coverage = np.abs(errors) <= 1.6448536269514722 * np.sqrt(variance)
    domain = np.asarray(prediction["in_domain"], dtype=bool)[indices]
    return {
        "n": len(rows), "n_groups": len({record["group_id"] for record in rows}),
        "unit": model.metadata["outcome_units"][outcome],
        "mse": float(np.mean(squared)), "rmse": float(math.sqrt(np.mean(squared))),
        "mae": float(np.mean(np.abs(errors))), "group_balanced_rmse": group_rmse,
        "training_mean_baseline": float(baseline),
        "mean_baseline_mse": float(np.mean(baseline_squared)),
        "mean_baseline_rmse": float(math.sqrt(np.mean(baseline_squared))),
        "group_balanced_baseline_rmse": group_baseline_rmse,
        "relative_rmse_to_mean_baseline": (group_rmse / group_baseline_rmse
                                            if group_baseline_rmse > 1e-12 else None),
        "gaussian_nll": float(np.mean(nll)), "nll_variance_floor": 1e-12,
        "approximate_gaussian_90_coverage": float(np.mean(coverage)),
        "in_domain_n": int(domain.sum()), "extrapolated_n": int((~domain).sum()),
    }


def _summarize(outputs: dict[str, dict], records: list[dict]) -> dict:
    observations = sum(values["n"] for values in outputs.values())
    same_unit = len({values["unit"] for values in outputs.values()}) == 1
    mse = (sum(values["mse"] * values["n"] for values in outputs.values()) / observations
           if observations and same_unit else None)
    baseline_mse = (sum(values["mean_baseline_mse"] * values["n"] for values in outputs.values()) / observations
                    if observations and same_unit else None)
    ratios = [values["relative_rmse_to_mean_baseline"] for values in outputs.values()
              if values["relative_rmse_to_mean_baseline"] is not None]
    return {
        "n_records": len(records), "n_groups": len({record["group_id"] for record in records}),
        "n_observed_outcomes": observations, "n_scored_outputs": len(outputs),
        "rmse": math.sqrt(mse) if mse is not None else None, "mse": mse,
        "mean_baseline_rmse": math.sqrt(baseline_mse) if baseline_mse is not None else None,
        "macro_relative_rmse_to_mean_baseline": float(np.mean(ratios)) if ratios else None,
        "relative_rmse_output_count": len(ratios),
        "pooled_unit": next(iter(outputs.values()))["unit"] if same_unit else None,
        "approximate_gaussian_90_coverage": (
            sum(values["approximate_gaussian_90_coverage"] * values["n"] for values in outputs.values()) / observations
            if observations else None),
        "aggregation": "Pooled RMSE only for matching units; each output also reports equal-group-weighted RMSE.",
    }


def _stratified_metrics(model, records: list[dict], prediction: dict, field: str) -> dict:
    strata = {}
    labels = [record.get(field, "unspecified") for record in records]
    domain = np.asarray(prediction["in_domain"], dtype=bool)
    for label in sorted(set(labels)):
        indices = [index for index, value in enumerate(labels) if value == label]
        outputs = {}
        for outcome in prediction["outcome_names"]:
            metrics = _output_metrics(model, records, prediction, outcome, indices)
            if metrics is not None:
                outputs[outcome] = metrics
        subset_domain = domain[indices]
        strata[label] = {
            **_summarize(outputs, [records[index] for index in indices]),
            "per_output": outputs, "in_domain_records": int(subset_domain.sum()),
            "extrapolated_records": int((~subset_domain).sum()),
            "in_domain_fraction": float(np.mean(subset_domain)),
        }
    return strata


def evaluate_model(model, records: list[dict]) -> dict:
    """Evaluate on evidence that is disjoint from model training groups and IDs.

    Mean baselines are fitted on training labels, never on validation labels.
    Labels for output heads not yet fitted are counted and disclosed, not scored.
    Scores report both pooled rows and equally weighted independent groups.
    """
    records = list(records)
    validate_dataset(records, allow_synthetic=True)
    _evidence_compatible(model.metadata, records)
    _overlap_guard(model.metadata, records)
    units = model.metadata.get("outcome_units", {})
    for record in records:
        for outcome, unit in record["outcome_units"].items():
            if outcome in units and units[outcome] != unit:
                raise ValueError(f"outcome unit mismatch for {outcome}")
    prediction = model.predict(records)
    expected_shape = (len(records), len(prediction["outcome_names"]))
    for key in ("mean", "epistemic_variance", "noise_variance"):
        values = np.asarray(prediction[key], dtype=float)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"invalid prediction {key} dimensions or numbers")
        if key != "mean" and (values < 0).any():
            raise ValueError("model returned a negative prediction variance")
    indices = list(range(len(records)))
    outputs = {}
    for outcome in prediction["outcome_names"]:
        metrics = _output_metrics(model, records, prediction, outcome, indices)
        if metrics is not None:
            outputs[outcome] = metrics
    if not outputs:
        raise ValueError("evaluation contains no observed labels for trained model outputs")
    per_organ = _stratified_metrics(model, records, prediction, "organ")
    per_species = _stratified_metrics(model, records, prediction, "species")
    unseen_labels = Counter(outcome for record in records for outcome in record["outcomes"]
                            if outcome not in prediction["outcome_names"])
    reasons = Counter(reason for row in prediction["domain_reasons"] for reason in row)
    domain = np.asarray(prediction["in_domain"], dtype=bool)
    return {
        "model_id": model.metadata.get("model_id"), "overall": _summarize(outputs, records),
        "per_output": outputs, "per_organ": per_organ, "per_species": per_species,
        "independent_validation": True,
        "disjointness_fields": ["observation_id", "group_id"] + (
            ["experiment_id"] if "training_experiments" in model.metadata else []) + (
            ["source_identity"] if "training_source_identities" in model.metadata else []),
        "evaluation_groups": sorted({record["group_id"] for record in records}),
        "evidence_types": sorted({record["evidence_type"] for record in records}),
        "sources": sorted({record["source"] for record in records}),
        "baseline": "Training outcome means; no fitting on validation labels.",
        "coverage": {"in_domain_records": int(domain.sum()), "extrapolated_records": int((~domain).sum()),
                     "in_domain_fraction": float(np.mean(domain)), "domain_reasons": dict(reasons),
                     "unscored_outcomes": dict(unseen_labels)},
        "limitations": list(_LIMITATIONS),
    }


def _evidence_fingerprint(record: dict) -> str:
    """Catch identical evidence relabeled with a different observation/group ID."""
    return content_hash({key: value for key, value in record.items()
                         if key not in {"observation_id", "group_id", "ingested_at", "recorded_at"}})


def _regression_check(label: str, before: float, after: float, tolerance: float,
                      *, metric: str) -> dict:
    # The numerical floor handles a perfect incumbent without division by zero.
    limit = before * (1.0 + tolerance) + 1e-12
    return {"slice": label, "metric": metric, "current": before, "candidate": after,
            "allowed_maximum": limit,
            "relative_change": (after / before - 1.0) if before > 1e-12 else None,
            "passed": bool(after <= limit)}


def _aggregate_for_gate(current_outputs: dict, candidate_outputs: dict) -> tuple[float, float] | None:
    names = sorted(set(current_outputs) & set(candidate_outputs))
    if not names:
        return None
    before, after = [], []
    for name in names:
        old = current_outputs[name]
        new = candidate_outputs[name]
        scale = max(old["group_balanced_baseline_rmse"], old["group_balanced_rmse"], 1e-12)
        before.append((old["group_balanced_rmse"] / scale) ** 2)
        after.append((new["group_balanced_rmse"] / scale) ** 2)
    return math.sqrt(float(np.mean(before))), math.sqrt(float(np.mean(after)))


def propose_update(current, replay_records: list[dict], new_records: list[dict],
                   validation_records: list[dict], *, max_relative_regression: float = 0.05,
                   allow_synthetic: bool = False):
    """Fit a candidate with full replay, evaluate independently, and return a gate.

    Invalid/duplicated/leaked evidence raises ValueError.  A valid candidate that
    fails quality or tissue coverage checks is returned with ``accepted=False``.
    Every old/new tissue and output needs independent validation.  New output
    heads must also improve upon their own training-mean baseline.
    """
    from .world_model import BayesianWorldModel

    if (isinstance(max_relative_regression, bool)
            or not isinstance(max_relative_regression, (int, float))
            or not math.isfinite(max_relative_regression) or max_relative_regression < 0):
        raise ValueError("max_relative_regression must be a finite nonnegative number")
    replay_records, new_records, validation_records = map(list, (replay_records, new_records, validation_records))
    if not replay_records or not new_records or not validation_records:
        raise ValueError("replay, new evidence and independent validation must each be nonempty")
    combined = replay_records + new_records
    validate_dataset(combined + validation_records, allow_synthetic=allow_synthetic)
    _evidence_compatible(current.metadata, combined + validation_records)
    old_ids = set(current.metadata.get("training_ids", []))
    replay_ids = {record["observation_id"] for record in replay_records}
    if not old_ids or not old_ids <= replay_ids:
        raise ValueError("replay must cover every current training observation_id to protect old evidence")
    fingerprints = current.metadata.get("training_record_hashes", {})
    if not old_ids <= set(fingerprints):
        raise ValueError("current model lacks training record fingerprints; refit with the current schema")
    for record in replay_records:
        identity = record["observation_id"]
        if identity in old_ids and content_hash(record) != fingerprints[identity]:
            raise ValueError(f"replay changed previously incorporated evidence: {identity}")
    if old_ids & {record["observation_id"] for record in new_records}:
        raise ValueError("new evidence repeats a current training observation_id")
    units = current.metadata.get("outcome_units", {})
    for record in combined + validation_records:
        for outcome, unit in record["outcome_units"].items():
            if outcome in units and units[outcome] != unit:
                raise ValueError(f"outcome unit mismatch for {outcome}")
    seen_fingerprints: set[str] = set()
    for record in combined + validation_records:
        fingerprint = _evidence_fingerprint(record)
        if fingerprint in seen_fingerprints:
            raise ValueError("repeated evidence under different observation or group IDs")
        seen_fingerprints.add(fingerprint)
    combined_metadata = {
        "training_ids": [record["observation_id"] for record in combined],
        "training_groups": [record["group_id"] for record in combined],
        "training_experiments": [record["experiment_id"] for record in combined],
    }
    _overlap_guard(combined_metadata, validation_records)

    config = current.metadata.get("fit_config", current.metadata)
    candidate = BayesianWorldModel.fit(
        combined, seed=int(config.get("seed", 0)),
        random_features=int(config.get("random_features", 16)),
        prior_precision=float(config.get("prior_precision", 1.0)),
        allow_synthetic=allow_synthetic,
    )
    before = evaluate_model(current, validation_records)
    after = evaluate_model(candidate, validation_records)
    checks: dict = {"overall": None, "per_output": {}, "per_organ": {}, "per_species": {}, "new_outputs": {}}
    reasons = []
    common = _aggregate_for_gate(before["per_output"], after["per_output"])
    if common is None:
        reasons.append("no_common_outputs_for_regression_check")
    else:
        checks["overall"] = _regression_check("overall", *common, max_relative_regression,
                                              metric="shared_scale_group_balanced_macro_rmse")
        if not checks["overall"]["passed"]:
            reasons.append("overall_prediction_regression")
    required_outputs = set(candidate.metadata["outcome_units"])
    for outcome in sorted(required_outputs):
        if outcome not in after["per_output"]:
            reasons.append(f"missing_validation_output:{outcome}")
            continue
        candidate_values = after["per_output"][outcome]
        if outcome in units:
            if outcome not in before["per_output"]:
                reasons.append(f"missing_incumbent_validation_output:{outcome}")
                continue
            check = _regression_check(
                outcome, before["per_output"][outcome]["group_balanced_rmse"],
                candidate_values["group_balanced_rmse"], max_relative_regression,
                metric="group_balanced_rmse",
            )
            checks["per_output"][outcome] = check
            if not check["passed"]:
                reasons.append(f"output_prediction_regression:{outcome}")
        else:
            new_check = {
                "metric": "group_balanced_rmse", "candidate": candidate_values["group_balanced_rmse"],
                "training_mean_baseline": candidate_values["group_balanced_baseline_rmse"],
                "passed": bool(candidate_values["group_balanced_rmse"]
                               < candidate_values["group_balanced_baseline_rmse"] - 1e-12),
            }
            checks["new_outputs"][outcome] = new_check
            if not new_check["passed"]:
                reasons.append(f"new_output_does_not_improve_mean_baseline:{outcome}")

    required_organs = sorted({record["organ"] for record in combined})
    old_organs = {record["organ"] for record in replay_records if record["observation_id"] in old_ids}
    for organ in required_organs:
        old_organ = before["per_organ"].get(organ)
        new_organ = after["per_organ"].get(organ)
        if not new_organ or not new_organ["per_output"]:
            reasons.append(f"missing_independent_validation_organ:{organ}")
            checks["per_organ"][organ] = {"passed": False, "reason": "missing_independent_validation"}
            continue
        supported_outputs = {outcome for record in combined if record["organ"] == organ
                             for outcome in record["outcomes"]}
        missing = supported_outputs - set(new_organ["per_output"])
        if missing:
            reasons.append(f"missing_validation_organ_outputs:{organ}:{','.join(sorted(missing))}")
        aggregate = _aggregate_for_gate(old_organ["per_output"], new_organ["per_output"]) if old_organ else None
        if aggregate is None:
            # A tissue with only new heads is validated against its training-mean baseline.
            passed = all(value["group_balanced_rmse"] < value["group_balanced_baseline_rmse"] - 1e-12
                         for value in new_organ["per_output"].values()) and not missing
            checks["per_organ"][organ] = {
                "passed": passed, "metric": "new_tissue_outputs_vs_training_mean_baseline",
                "validation_group_count": new_organ["n_groups"],
            }
            if not passed:
                reasons.append(f"new_organ_does_not_improve_mean_baseline:{organ}")
        else:
            check = _regression_check(f"organ:{organ}", *aggregate, max_relative_regression,
                                      metric="shared_scale_group_balanced_macro_rmse")
            check["validation_group_count"] = new_organ["n_groups"]
            check["missing_validation_outputs"] = sorted(missing)
            check["passed"] = check["passed"] and not missing
            checks["per_organ"][organ] = check
            if not check["passed"]:
                reasons.append(f"organ_prediction_regression_or_missing_outputs:{organ}")

        # An organ-level average must not hide regression in a specific outcome.
        organ_checks = checks["per_organ"][organ]
        organ_checks["per_output"] = {}
        organ_checks["incumbent_organ_supported"] = organ in old_organs
        for outcome in sorted(supported_outputs & set(new_organ["per_output"])):
            values = new_organ["per_output"][outcome]
            if organ in old_organs and outcome in units and old_organ and outcome in old_organ["per_output"]:
                output_check = _regression_check(
                    f"organ:{organ}:output:{outcome}",
                    old_organ["per_output"][outcome]["group_balanced_rmse"],
                    values["group_balanced_rmse"], max_relative_regression, metric="group_balanced_rmse",
                )
            else:
                output_check = {
                    "metric": "group_balanced_rmse_vs_training_mean_baseline",
                    "candidate": values["group_balanced_rmse"],
                    "training_mean_baseline": values["group_balanced_baseline_rmse"],
                    "passed": bool(values["group_balanced_rmse"] < values["group_balanced_baseline_rmse"] - 1e-12),
                }
            organ_checks["per_output"][outcome] = output_check
            if not output_check["passed"]:
                organ_checks["passed"] = False
                reasons.append(f"organ_output_gate_failed:{organ}:{outcome}")

    required_species = sorted({record.get("species", "unspecified") for record in combined})
    for species in required_species:
        metrics = after["per_species"].get(species)
        if not metrics or not metrics["per_output"]:
            reasons.append(f"missing_independent_validation_species:{species}")
            checks["per_species"][species] = {"passed": False, "reason": "missing_independent_validation"}
            continue
        supported = metrics["in_domain_records"] > 0
        checks["per_species"][species] = {
            "passed": supported, "validation_group_count": metrics["n_groups"],
            "in_domain_records": metrics["in_domain_records"],
            "extrapolated_records": metrics["extrapolated_records"],
            "in_domain_fraction": metrics["in_domain_fraction"],
            "criterion": "At least one independent validation record must lie in the candidate support domain.",
        }
        if not supported:
            reasons.append(f"no_in_domain_validation_species:{species}")

    accepted = not reasons
    report = {
        "decision": "accept" if accepted else "reject", "accepted": accepted, "reasons": reasons,
        "current_model_id": current.metadata.get("model_id"), "candidate_model_id": candidate.metadata.get("model_id"),
        "current_metrics": before, "candidate_metrics": after, "regression_checks": checks,
        "max_relative_regression": float(max_relative_regression),
        "regression_metric": "Equal-group-weighted RMSE; aggregate outputs use common incumbent-based scales.",
        "aggregate_scale_basis": "Per output: max(current mean-baseline RMSE, current model RMSE, 1e-12).",
        "new_evidence_count": len(new_records), "replay_count": len(replay_records),
        "replay_covers_all_training_ids": True,
        "validation_group_count": len({record["group_id"] for record in validation_records}),
        "required_organs": required_organs, "required_species": required_species,
        "independent_validation": True,
        "promoted": False, "limitations": list(_LIMITATIONS),
    }
    return candidate, report
