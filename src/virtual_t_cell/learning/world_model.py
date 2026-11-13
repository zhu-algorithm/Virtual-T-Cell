"""A small, inspectable probabilistic response surrogate.

The nonlinear basis is fixed at initialization; output coefficients are learned
with Gaussian Bayesian regression.  Each outcome has its own posterior and can
be missing from individual records.  This is not a mechanistic simulator, a
causal discovery method, or a calibrated uncertainty guarantee.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .schema import canonical_json, content_hash, evidence_identity, validate_dataset, validate_record


_CONTEXT_FIELDS = ("organ", "cell_type", "condition", "species")
_FORMAT = "virtual_t_cell.random_feature_bayes.v1"
_ARRAY_NAMES = (
    "feature_mean", "feature_scale", "feature_min", "feature_max",
    "random_weights", "random_phase", "posterior_mean", "posterior_covariance",
    "outcome_center", "outcome_scale", "residual_variance",
    "default_measurement_variance",
)


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _context(row: dict, field: str) -> str:
    value = row.get(field, "unspecified") if field == "species" else row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string when provided")
    return value


def _positive_integer(value, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return int(value)


def _posterior(design: np.ndarray, target: np.ndarray, known_variance: np.ndarray,
               residual_variance: float, prior_precision: float):
    """Posterior and marginal negative log likelihood in standardized units.

    The last quadratic is evaluated as a regularized residual sum of squares,
    avoiding cancellation between two large quadratic forms at near-zero noise.
    """
    variance = known_variance + residual_variance
    weights = 1.0 / variance
    precision = prior_precision * np.eye(design.shape[1])
    precision += design.T @ (design * weights[:, None])
    precision = (precision + precision.T) * 0.5
    chol = np.linalg.cholesky(precision)
    rhs = design.T @ (weights * target)
    mean = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    errors = target - design @ mean
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    objective = 0.5 * (
        np.log(variance).sum() + logdet
        - design.shape[1] * math.log(prior_precision)
        + np.dot(errors * weights, errors)
        + prior_precision * np.dot(mean, mean)
        + len(target) * math.log(2.0 * math.pi)
    )
    return float(objective), mean, chol


def _fit_head(design: np.ndarray, target: np.ndarray, known_variance: np.ndarray,
              prior_precision: float):
    """Empirical-Bayes residual noise, conditional on reported measurement SD.

    One-dimensional marginal-likelihood search avoids interpreting a noisy
    measurement outlier as equal-precision evidence.  The small positive floor
    is a numerical assumption, not a measured biological variance.
    """
    floor = 1e-8
    upper = max(float(np.mean(target * target)), 1.0) * 10.0
    grid = np.linspace(math.log(floor), math.log(upper), 17)

    def score(log_variance: float) -> float:
        return _posterior(design, target, known_variance,
                          math.exp(log_variance), prior_precision)[0]

    scores = np.asarray([score(float(value)) for value in grid])
    best = int(np.argmin(scores))
    lo, hi = float(grid[max(0, best - 1)]), float(grid[min(len(grid) - 1, best + 1)])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)
    left_score, right_score = score(left), score(right)
    for _ in range(18):
        if left_score <= right_score:
            hi, right, right_score = right, left, left_score
            left = hi - ratio * (hi - lo)
            left_score = score(left)
        else:
            lo, left, left_score = left, right, right_score
            right = lo + ratio * (hi - lo)
            right_score = score(right)
    choices = [(float(grid[best]), float(scores[best])), (left, left_score), (right, right_score)]
    log_variance = min(choices, key=lambda item: item[1])[0]
    residual_variance = math.exp(log_variance)
    objective, mean, chol = _posterior(
        design, target, known_variance, residual_variance, prior_precision
    )
    inverse_chol = np.linalg.solve(chol, np.eye(design.shape[1]))
    covariance = inverse_chol.T @ inverse_chol
    covariance = (covariance + covariance.T) * 0.5
    return mean, covariance, residual_variance, objective


class BayesianWorldModel:
    """Independent Gaussian response heads over a shared nonlinear basis.

    ``features`` must contain all numeric descriptors the model should use,
    including action, dose or biological time descriptors when relevant.  Record
    IDs and intervention names are provenance, not learned numeric features.
    Organ, cell type, condition and species are always encoded explicitly.
    An absent species is the separate category ``unspecified``, never an inferred
    human or mouse label.

    Uncertainty is conditional on this basis, Gaussian likelihood, estimated
    noise and fixed hyperparameters.  It does not include uncertainty about the
    feature representation or unknown mechanisms; independent heads do not
    model correlation between outcomes.
    """

    @classmethod
    def fit(cls, records, *, seed=0, random_features=16, prior_precision=1.0,
            allow_synthetic=False) -> "BayesianWorldModel":
        records = list(records)
        validate_dataset(records, allow_synthetic=allow_synthetic)
        seed = _positive_integer(seed, "seed", allow_zero=True)
        random_features = _positive_integer(random_features, "random_features", allow_zero=True)
        if (isinstance(prior_precision, bool)
                or not isinstance(prior_precision, (int, float))
                or not math.isfinite(prior_precision) or prior_precision <= 0):
            raise ValueError("prior_precision must be finite and positive")
        prior_precision = float(prior_precision)
        model = cls()
        model._feature_names = sorted(records[0]["features"])
        model._outcome_names = sorted({key for row in records for key in row["outcomes"]})
        model._context_levels = {
            field: sorted({_context(row, field) for row in records}) for field in _CONTEXT_FIELDS
        }
        numeric = np.asarray([
            [row["features"][key] for key in model._feature_names] for row in records
        ], dtype=np.float64)
        model._feature_mean = numeric.mean(axis=0)
        model._feature_scale = numeric.std(axis=0)
        model._feature_scale[model._feature_scale < 1e-12] = 1.0
        model._feature_min = numeric.min(axis=0)
        model._feature_max = numeric.max(axis=0)
        if not all(np.isfinite(value).all() for value in (
            model._feature_mean, model._feature_scale, model._feature_min, model._feature_max
        )):
            raise ValueError("numeric feature magnitudes overflow training normalization")
        base = model._encode(records)
        rng = np.random.default_rng(seed)
        # Each one-hot context contributes one active coordinate.
        bandwidth = math.sqrt(len(model._feature_names) + len(_CONTEXT_FIELDS))
        model._random_weights = rng.normal(
            scale=1.0 / bandwidth, size=(base.shape[1], random_features)
        )
        model._random_phase = rng.uniform(0.0, 2.0 * math.pi, size=random_features)
        design = model._design(base)
        heads, dimensions = len(model._outcome_names), design.shape[1]
        model._posterior_mean = np.empty((heads, dimensions), dtype=np.float64)
        model._posterior_covariance = np.empty((heads, dimensions, dimensions), dtype=np.float64)
        model._outcome_center = np.empty(heads, dtype=np.float64)
        model._outcome_scale = np.empty(heads, dtype=np.float64)
        model._residual_variance = np.empty(heads, dtype=np.float64)
        model._default_measurement_variance = np.empty(heads, dtype=np.float64)
        output_counts, outcome_means, units, output_coverage, objectives = {}, {}, {}, {}, {}

        for column, outcome in enumerate(model._outcome_names):
            indices = [index for index, row in enumerate(records) if outcome in row["outcomes"]]
            observed = [records[index] for index in indices]
            target = np.asarray([row["outcomes"][outcome] for row in observed], dtype=np.float64)
            center, scale = float(target.mean()), float(target.std())
            if not math.isfinite(center) or not math.isfinite(scale):
                raise ValueError(f"numeric magnitudes overflow outcome normalization: {outcome}")
            scale = scale if scale >= 1e-12 else 1.0
            measurement = np.asarray([
                row.get("measurement_sd", {}).get(outcome, 0.0) for row in observed
            ], dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                known_variance = (measurement / scale) ** 2
                reported_sd = np.asarray([
                    float(row["measurement_sd"][outcome])
                    for row in observed if outcome in row.get("measurement_sd", {})
                ], dtype=np.float64)
                reported_variances = np.square(reported_sd).tolist()
            if not np.isfinite(known_variance).all() or any(
                not math.isfinite(value) for value in reported_variances
            ):
                raise ValueError(f"measurement variance overflow: {outcome}")
            try:
                mean, covariance, residual, objective = _fit_head(
                    design[indices], (target - center) / scale, known_variance, prior_precision
                )
            except np.linalg.LinAlgError as error:
                raise ValueError("posterior could not be factorized; rescale features/outcomes") from error
            model._posterior_mean[column] = mean
            model._posterior_covariance[column] = covariance
            model._outcome_center[column], model._outcome_scale[column] = center, scale
            model._residual_variance[column] = residual * scale * scale
            model._default_measurement_variance[column] = (
                float(np.median(reported_variances)) if reported_variances else 0.0
            )
            output_counts[outcome] = len(observed)
            outcome_means[outcome] = center
            units[outcome] = observed[0]["outcome_units"][outcome]
            output_coverage[outcome] = {
                "count": len(observed),
                "organs": _counts(row["organ"] for row in observed),
                "cell_types": _counts(row["cell_type"] for row in observed),
                "conditions": _counts(row["condition"] for row in observed),
                "species": _counts(_context(row, "species") for row in observed),
                "known_measurement_sd_count": len(reported_variances),
            }
            objectives[outcome] = objective

        context_counts = Counter(tuple(_context(row, field) for field in _CONTEXT_FIELDS) for row in records)
        metadata = {
            "format": _FORMAT,
            "version": 1,
            "model_type": "random_feature_bayesian_regression",
            "seed": seed,
            "random_features": random_features,
            "prior_precision": prior_precision,
            "fit_config": {"seed": seed, "random_features": random_features,
                           "prior_precision": prior_precision},
            "feature_names": model._feature_names,
            "outcome_names": model._outcome_names,
            "context_levels": model._context_levels,
            "training_ids": sorted(row["observation_id"] for row in records),
            "training_groups": sorted({row["group_id"] for row in records}),
            "training_experiments": sorted({row["experiment_id"] for row in records}),
            "training_record_hashes": {
                row["observation_id"]: content_hash(row) for row in records
            },
            "training_source_identities": sorted(evidence_identity(row) for row in records),
            "training_record_evidence_types": {
                row["observation_id"]: row["evidence_type"] for row in records
            },
            "training_record_organs": {row["observation_id"]: row["organ"] for row in records},
            "training_hash": content_hash(sorted(records, key=lambda row: row["observation_id"])),
            "training_count": len(records),
            "outcome_units": units,
            "outcome_means": outcome_means,
            "output_counts": output_counts,
            "evidence_types": sorted({row["evidence_type"] for row in records}),
            "evidence_counts": _counts(row["evidence_type"] for row in records),
            "sources": sorted({row["source"] for row in records}),
            "training_interventions": sorted({row["intervention"] for row in records}),
            "training_coverage": {
                "organs": _counts(row["organ"] for row in records),
                "cell_types": _counts(row["cell_type"] for row in records),
                "conditions": _counts(row["condition"] for row in records),
                "species": _counts(_context(row, "species") for row in records),
                "groups": len({row["group_id"] for row in records}),
                "contexts": [dict(zip(_CONTEXT_FIELDS, context), count=count)
                             for context, count in sorted(context_counts.items())],
                "outcomes": output_coverage,
                "features": {
                    name: {"mean": float(model._feature_mean[index]),
                           "scale": float(model._feature_scale[index]),
                           "min": float(model._feature_min[index]),
                           "max": float(model._feature_max[index])}
                    for index, name in enumerate(model._feature_names)
                },
            },
            "noise_model": {
                "residual_estimation": "training marginal likelihood with known measurement variances",
                "normalized_residual_variance_floor": 1e-8,
                "missing_training_measurement_sd": "no separately known measurement variance; residual absorbs it",
                "missing_query_measurement_sd": "median reported training measurement variance, or zero if none",
                "residual_variance": dict(zip(model._outcome_names, model._residual_variance.tolist())),
                "default_measurement_variance": dict(zip(
                    model._outcome_names, model._default_measurement_variance.tolist()
                )),
            },
            "training_negative_log_marginal_likelihood_standardized": objectives,
            "uncertainty_calibration": "not established; conditional empirical-Bayes approximation",
            "limitations": [
                "Independent output heads omit cross-outcome covariance.",
                "Likelihood treats records as conditionally independent; correlated summaries need appropriate experimental units.",
                "Posterior uncertainty conditions on fitted normalization, basis and noise hyperparameters.",
                "In-domain status is a coverage check, not a guarantee for unseen mechanisms.",
                "Numeric features must explicitly encode relevant action, dose and biological time descriptors.",
                "Response associations do not establish causal identification or biological dynamics.",
            ],
        }
        model._metadata = metadata
        model._check_arrays()
        model._metadata["model_id"] = model._identity()
        return model

    @property
    def metadata(self) -> dict:
        """Return a copy so provenance cannot be silently edited in place."""
        return deepcopy(self._metadata)

    @property
    def outcome_names(self) -> list[str]:
        return list(self._outcome_names)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _encode(self, records) -> np.ndarray:
        numeric = np.asarray([
            [row["features"][key] for key in self._feature_names] for row in records
        ], dtype=np.float64).reshape(len(records), len(self._feature_names))
        with np.errstate(over="ignore", invalid="ignore"):
            numeric = (numeric - self._feature_mean) / self._feature_scale
        categorical = np.zeros((len(records), sum(map(len, self._context_levels.values()))))
        offset = 0
        for field in _CONTEXT_FIELDS:
            indices = {value: index for index, value in enumerate(self._context_levels[field])}
            for row_index, row in enumerate(records):
                index = indices.get(_context(row, field))
                if index is not None:
                    categorical[row_index, offset + index] = 1.0
            offset += len(indices)
        encoded = np.concatenate((numeric, categorical), axis=1)
        if not np.isfinite(encoded).all():
            raise ValueError("numeric feature magnitudes overflow model normalization")
        return encoded

    def _design(self, encoded: np.ndarray) -> np.ndarray:
        count = self._random_weights.shape[1]
        blocks = [np.ones((len(encoded), 1)), encoded]
        if count:
            with np.errstate(over="ignore", invalid="ignore"):
                random = math.sqrt(2.0 / count) * np.cos(
                    encoded @ self._random_weights + self._random_phase
                )
            blocks.append(random)
        design = np.concatenate(blocks, axis=1)
        if not np.isfinite(design).all():
            raise ValueError("numeric feature magnitudes overflow random features")
        return design

    def _query(self, records):
        records = list(records)
        expected = set(self._feature_names)
        for row in records:
            validate_record(row, require_outcomes=False)
            actual = set(row["features"])
            if actual != expected:
                raise ValueError(f"feature contract mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
            for outcome, unit in row.get("outcome_units", {}).items():
                known_unit = self._metadata["outcome_units"].get(outcome)
                if known_unit is not None and unit != known_unit:
                    raise ValueError(f"inconsistent units for {outcome}: {known_unit} versus {unit}")
        return records, self._design(self._encode(records))

    def _domain(self, records):
        contexts = {
            tuple(item[field] for field in _CONTEXT_FIELDS)
            for item in self._metadata["training_coverage"]["contexts"]
        }
        all_reasons = []
        for row in records:
            reasons = []
            for field in _CONTEXT_FIELDS:
                value = _context(row, field)
                if value not in self._context_levels[field]:
                    reasons.append(f"unknown_{field}:{value}")
            if not reasons and tuple(_context(row, field) for field in _CONTEXT_FIELDS) not in contexts:
                reasons.append("unseen_organ_cell_type_condition_species_combination")
            for column, name in enumerate(self._feature_names):
                value = row["features"][name]
                lower, upper = self._feature_min[column], self._feature_max[column]
                # Small tolerance is only for floating point boundary roundoff.
                tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
                if value < lower - tolerance or value > upper + tolerance:
                    reasons.append(f"feature_outside_training_range:{name}")
            if row["organ"] in self._context_levels["organ"]:
                for outcome in self._outcome_names:
                    support = self._metadata["training_coverage"]["outcomes"][outcome]
                    if row["organ"] not in support["organs"]:
                        reasons.append(f"outcome_without_organ_evidence:{outcome}:{row['organ']}")
            species = _context(row, "species")
            if species in self._context_levels["species"]:
                for outcome in self._outcome_names:
                    support = self._metadata["training_coverage"]["outcomes"][outcome]
                    if species not in support["species"]:
                        reasons.append(f"outcome_without_species_evidence:{outcome}:{species}")
            all_reasons.append(reasons)
        return np.asarray([not reasons for reasons in all_reasons], dtype=bool), all_reasons

    def predict(self, records) -> dict:
        records, design = self._query(records)
        mean = (design @ self._posterior_mean.T) * self._outcome_scale + self._outcome_center
        epistemic = np.empty((len(records), len(self._outcome_names)), dtype=np.float64)
        for column in range(len(self._outcome_names)):
            epistemic[:, column] = np.maximum(
                np.sum((design @ self._posterior_covariance[column]) * design, axis=1), 0.0
            ) * self._outcome_scale[column] ** 2
        measurement = np.broadcast_to(
            self._default_measurement_variance, mean.shape
        ).copy()
        for row_index, row in enumerate(records):
            for column, outcome in enumerate(self._outcome_names):
                if outcome in row.get("measurement_sd", {}):
                    sd = row["measurement_sd"][outcome]
                    try:
                        measurement[row_index, column] = sd ** 2
                    except OverflowError as error:
                        raise ValueError(f"measurement variance overflow: {outcome}") from error
        residual = np.broadcast_to(self._residual_variance, mean.shape).copy()
        noise = residual + measurement
        if not all(np.isfinite(value).all() for value in (mean, epistemic, noise)):
            raise ValueError("prediction overflows; numeric inputs are outside a usable scale")
        in_domain, reasons = self._domain(records)
        return {
            "outcome_names": self.outcome_names,
            "mean": mean,
            "epistemic_variance": epistemic,
            "noise_variance": noise,
            "residual_variance": residual,
            "measurement_variance": measurement,
            "in_domain": in_domain,
            "domain_reasons": reasons,
            "model_id": self._metadata["model_id"],
            "uncertainty_calibration": self._metadata["uncertainty_calibration"],
        }

    def covariance(self, records, outcome: str) -> np.ndarray:
        """Latent covariance; observation/measurement noise is not added."""
        if outcome not in self._outcome_names:
            raise ValueError(f"unknown outcome: {outcome}")
        _, design = self._query(records)
        column = self._outcome_names.index(outcome)
        covariance = design @ self._posterior_covariance[column] @ design.T
        covariance *= self._outcome_scale[column] ** 2
        covariance = (covariance + covariance.T) * 0.5
        if not np.isfinite(covariance).all():
            raise ValueError("covariance overflows; numeric inputs are outside a usable scale")
        return covariance

    def _arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, "_" + name) for name in _ARRAY_NAMES}

    def _identity(self) -> str:
        metadata = {key: value for key, value in self._metadata.items() if key != "model_id"}
        digest = hashlib.sha256(canonical_json(metadata).encode("utf-8"))
        for name, values in sorted(self._arrays().items()):
            canonical = np.ascontiguousarray(values, dtype="<f8")
            digest.update(name.encode("ascii"))
            digest.update(canonical_json(list(canonical.shape)).encode("ascii"))
            digest.update(canonical.tobytes())
        return digest.hexdigest()

    def _check_arrays(self):
        inputs = len(self._feature_names)
        base = inputs + sum(len(self._context_levels[field]) for field in _CONTEXT_FIELDS)
        random = self._metadata["random_features"]
        dimensions = 1 + base + random
        heads = len(self._outcome_names)
        shapes = {
            "feature_mean": (inputs,), "feature_scale": (inputs,),
            "feature_min": (inputs,), "feature_max": (inputs,),
            "random_weights": (base, random), "random_phase": (random,),
            "posterior_mean": (heads, dimensions),
            "posterior_covariance": (heads, dimensions, dimensions),
            "outcome_center": (heads,), "outcome_scale": (heads,),
            "residual_variance": (heads,), "default_measurement_variance": (heads,),
        }
        for name, values in self._arrays().items():
            if values.shape != shapes[name] or not np.isfinite(values).all():
                raise ValueError(f"invalid finite array or shape in model: {name}")
        if (np.any(self._feature_scale <= 0) or np.any(self._outcome_scale <= 0)
                or np.any(self._residual_variance <= 0)
                or np.any(self._default_measurement_variance < 0)
                or np.any(self._feature_min > self._feature_max)):
            raise ValueError("model contains invalid scales, bounds or noise variances")

    def save(self, path: str | Path) -> None:
        """Save numeric arrays and JSON metadata without Python object pickles."""
        self._check_arrays()
        if self._metadata["model_id"] != self._identity():
            raise ValueError("model content has changed since it was fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, metadata_json=np.asarray(canonical_json(self._metadata)),
                                **self._arrays())

    @classmethod
    def load(cls, path: str | Path) -> "BayesianWorldModel":
        with np.load(Path(path), allow_pickle=False) as archive:
            expected = set(_ARRAY_NAMES) | {"metadata_json"}
            if set(archive.files) != expected:
                raise ValueError("model archive has missing or unexpected entries")
            encoded_metadata = archive["metadata_json"]
            if encoded_metadata.ndim != 0 or encoded_metadata.dtype.kind != "U":
                raise ValueError("model metadata must be a Unicode JSON scalar")
            metadata = json.loads(str(encoded_metadata.item()))
            if metadata.get("format") != _FORMAT or metadata.get("version") != 1:
                raise ValueError("unsupported model format/version")
            model = cls()
            model._metadata = metadata
            model._feature_names = list(metadata["feature_names"])
            model._outcome_names = list(metadata["outcome_names"])
            model._context_levels = deepcopy(metadata["context_levels"])
            for name in _ARRAY_NAMES:
                array = archive[name]
                if array.dtype.kind != "f":
                    raise ValueError(f"model array must be floating point: {name}")
                setattr(model, "_" + name, np.asarray(array, dtype=np.float64).copy())
        model._check_arrays()
        if metadata.get("model_id") != model._identity():
            raise ValueError("model content checksum mismatch")
        return model
