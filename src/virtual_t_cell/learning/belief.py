"""An evidence ledger attached to a real fitted response posterior.

Recording evidence does not train a model.  Pending records are only marked as
incorporated after the caller binds a newly fitted model; the continual-learning
validation gate owns the decision to promote that model.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np

from .schema import content_hash, validate_record
from .world_model import BayesianWorldModel


class BeliefState:
    """Posterior identity, evidence accounting and pre-update residual diagnostics.

    Residuals recorded here are descriptive checks on newly received observations.
    Their interval hit rate is not an independent calibration estimate, especially
    when observations were selected adaptively.  Use held-out grouped evaluation
    for assessment; this ledger neither discovers causes nor invents evidence.
    """

    def __init__(self, model: BayesianWorldModel):
        self._known: dict[str, dict] = {}
        self._residuals: list[dict] = []
        self._history: list[dict] = []
        self.bind_model(model)

    @classmethod
    def from_model(cls, model: BayesianWorldModel) -> "BeliefState":
        return cls(model)

    def bind_model(self, model: BayesianWorldModel) -> None:
        """Attach a fitted posterior without implicitly approving its promotion.

        Existing evidence fingerprints must agree.  A caller should run the
        continual-learning gate before binding a candidate model.
        """
        if not isinstance(model, BayesianWorldModel):
            raise TypeError("belief state requires a fitted BayesianWorldModel")
        metadata = model.metadata
        hashes = metadata["training_record_hashes"]
        for observation_id, fingerprint in hashes.items():
            existing = self._known.get(observation_id)
            if existing is not None and existing["hash"] != fingerprint:
                raise ValueError(f"changed evidence content for observation_id: {observation_id}")
        previous = getattr(self, "_metadata", {}).get("model_id")
        for observation_id, fingerprint in hashes.items():
            self._known.setdefault(observation_id, {
                "hash": fingerprint,
                "evidence_type": metadata["training_record_evidence_types"][observation_id],
                "organ": metadata["training_record_organs"][observation_id],
            })
        self._model = model
        self._metadata = metadata
        self._incorporated = set(metadata["training_ids"])
        if previous != metadata["model_id"]:
            self._history.append({
                "previous_model_id": previous,
                "model_id": metadata["model_id"],
                "incorporated_observations": len(self._incorporated),
                "promotion_checked_by": "caller; binding alone is not validation approval",
            })

    def predict(self, records) -> dict:
        """Use the bound posterior. Predictions never increment evidence counts."""
        return self._model.predict(records)

    def record_observation(self, record: dict, prediction: dict | None = None) -> dict:
        """Record one new observation and its pre-update prediction residuals.

        ``prediction`` may be the one-row output of the bound model's ``predict``;
        if omitted it is computed before evidence is recorded.  The record remains
        pending until it is present in a subsequently bound model's training IDs.
        Repeated identical observations are idempotent.  Predictions supplied as
        evidence and conflicting edits to an existing observation ID are rejected.
        """
        validate_record(record, require_outcomes=record.get("evidence_type") != "observational")
        if record["evidence_type"] == "model_prediction":
            raise ValueError("model predictions cannot be recorded as biological evidence")
        observation_id = record["observation_id"]
        fingerprint = content_hash(record)
        previous = self._known.get(observation_id)
        if previous is not None:
            if previous["hash"] != fingerprint:
                raise ValueError(f"changed evidence content for observation_id: {observation_id}")
            return {"status": "duplicate", "observation_id": observation_id,
                    "incorporated": observation_id in self._incorporated,
                    "posterior_updated": False}

        computed = self._model.predict([record])
        if prediction is None:
            prediction = computed
        elif prediction.get("model_id") != self._metadata["model_id"]:
            raise ValueError("prediction must identify the currently bound model")
        names = prediction.get("outcome_names")
        if names != self._model.outcome_names:
            raise ValueError("prediction outcome names do not match the bound model")
        arrays = {}
        for field in ("mean", "epistemic_variance", "noise_variance"):
            try:
                values = np.asarray(prediction[field], dtype=float)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid prediction field: {field}") from error
            if values.shape != (1, len(names)) or not np.isfinite(values).all():
                raise ValueError(f"prediction {field} must be a finite one-row array")
            if field != "mean" and np.any(values < 0):
                raise ValueError("prediction variances must be nonnegative")
            # A caller cannot attach arbitrary numbers to a posterior identity.
            if not np.allclose(values, computed[field], rtol=1e-10, atol=1e-12):
                raise ValueError(f"provided prediction does not match record/model: {field}")
            arrays[field] = values

        diagnostics = []
        for column, outcome in enumerate(names):
            if outcome not in record["outcomes"]:
                continue
            predicted_mean = float(arrays["mean"][0, column])
            observed = float(record["outcomes"][outcome])
            epistemic = float(arrays["epistemic_variance"][0, column])
            noise = float(arrays["noise_variance"][0, column])
            residual = observed - predicted_mean
            sd = math.sqrt(epistemic + noise)
            standardized = residual / sd if sd > 0 else None
            if not math.isfinite(residual) or (standardized is not None and not math.isfinite(standardized)):
                raise ValueError("observation residual overflows; rescale the outcome")
            diagnostics.append({
                "observation_id": observation_id,
                "outcome": outcome,
                "unit": self._metadata["outcome_units"][outcome],
                "organ": record["organ"],
                "evidence_type": record["evidence_type"],
                "source": record["source"],
                "model_id": self._metadata["model_id"],
                "observed": observed,
                "predicted_mean": predicted_mean,
                "residual": residual,
                "epistemic_variance": epistemic,
                "noise_variance": noise,
                "standardized_residual": standardized,
                "within_nominal_95_percent_interval": bool(abs(residual) <= 1.959963984540054 * sd),
                "in_domain": bool(computed["in_domain"][0]),
                "domain_reasons": list(computed["domain_reasons"][0]),
                "prediction_precedes_incorporation": True,
            })
        self._known[observation_id] = {
            "hash": fingerprint,
            "evidence_type": record["evidence_type"],
            "organ": record["organ"],
        }
        self._residuals.extend(diagnostics)
        return {"status": "recorded", "observation_id": observation_id,
                "incorporated": False, "posterior_updated": False,
                "residuals": deepcopy(diagnostics),
                "unmodeled_outcomes": sorted(set(record["outcomes"]) - set(names))}

    def to_dict(self) -> dict:
        """A JSON-safe summary; neither arrays nor evidence labels are fabricated."""
        incorporated = [value for key, value in self._known.items() if key in self._incorporated]
        pending = [value for key, value in self._known.items() if key not in self._incorporated]

        def counts(records):
            return dict(sorted(Counter(row["evidence_type"] for row in records).items()))

        summaries = {}
        # Keep source types separate: a synthetic residual is not experimental
        # validation.  Model revisions also get separate residual summaries.
        groups = {}
        for row in self._residuals:
            key = (row["model_id"], row["evidence_type"], row["organ"], row["outcome"])
            groups.setdefault(key, []).append(row)
        for (model_id, evidence_type, organ, outcome), rows in sorted(groups.items()):
            residuals = np.asarray([row["residual"] for row in rows])
            scale = float(np.max(np.abs(residuals)))
            scaled = residuals / scale if scale else residuals
            key = f"{model_id}/{evidence_type}/{organ}/{outcome}"
            summaries[key] = {
                "model_id": model_id,
                "evidence_type": evidence_type,
                "organ": organ,
                "outcome": outcome,
                "count": len(rows),
                "mean_residual": float(scaled.mean() * scale),
                "mean_absolute_residual": float(np.abs(scaled).mean() * scale),
                "root_mean_squared_residual": float(np.sqrt(np.mean(scaled ** 2)) * scale),
                "nominal_95_percent_interval_hit_rate": float(np.mean([
                    row["within_nominal_95_percent_interval"] for row in rows
                ])),
                "out_of_domain_count": sum(not row["in_domain"] for row in rows),
            }
        return {
            "belief_type": "fitted_bayesian_response_posterior_with_evidence_ledger",
            "model_id": self._metadata["model_id"],
            "outcome_units": deepcopy(self._metadata["outcome_units"]),
            "training_coverage": deepcopy(self._metadata["training_coverage"]),
            "evidence_counts": {
                "total_unique": len(self._known),
                "by_type": counts(self._known.values()),
                "incorporated": counts(incorporated),
                "pending": counts(pending),
            },
            "pending_observation_ids": sorted(set(self._known) - self._incorporated),
            "residual_summaries": summaries,
            "residuals": deepcopy(self._residuals),
            "model_history": deepcopy(self._history),
            "uncertainty_calibration": self._metadata["uncertainty_calibration"],
            "diagnostic_limitations": [
                "Residuals and interval hits are descriptive; they are not independent calibration evidence.",
                "Recording an observation does not update model parameters or approve a model revision.",
                "Observational and synthetic evidence remain labeled separately from experimental evidence.",
                "No causal mechanism is inferred from an unexpected prediction residual.",
            ],
        }
