"""Finite-candidate, correlated Gaussian knowledge-gradient experiment planning.

This module ranks proposed measurements.  It does not operate instruments.  The
one-step objective is the expected improvement in the *best posterior mean* on
one named outcome, not the largest prediction or the largest marginal variance.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .schema import canonical_json, validate_record


_LIMITATIONS = [
    "One-step finite-candidate Gaussian approximation; it is not a multistep policy.",
    "Only the named outcome is optimized; other biological objectives require explicit constraints.",
    "Uncertainty is model-based and has no general calibration or causal validity guarantee.",
    "The terminal decision set contains only affordable candidates admitted by the domain policy.",
    "A recommendation does not execute an experiment or create experimental evidence.",
]


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _normal_pdf(value: float) -> float:
    return 0.0 if math.isinf(value) else math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_interval(lower: float, upper: float) -> float:
    """Normal probability without subtracting two near-one CDF values."""
    if lower >= 0:
        return 0.5 * (math.erfc(lower / math.sqrt(2.0)) - math.erfc(upper / math.sqrt(2.0)))
    if upper <= 0:
        return 0.5 * (math.erfc(-upper / math.sqrt(2.0)) - math.erfc(-lower / math.sqrt(2.0)))
    return 1.0 - 0.5 * math.erfc(upper / math.sqrt(2.0)) - 0.5 * math.erfc(-lower / math.sqrt(2.0))


def _expected_max_normal_lines(intercepts: np.ndarray, slopes: np.ndarray) -> float:
    """Integrate max_i(a_i + b_i Z), Z ~ N(0,1), using its upper envelope.

    Unlike a finite quadrature grid, this also captures small gains occurring
    beyond a grid's final node.  All correlations enter through the slopes.
    """
    lines = sorted(zip(slopes.tolist(), intercepts.tolist()))
    distinct: list[tuple[float, float]] = []
    for slope, intercept in lines:
        if distinct and slope == distinct[-1][0]:
            if intercept > distinct[-1][1]:
                distinct[-1] = (slope, intercept)
        else:
            distinct.append((slope, intercept))
    hull: list[tuple[float, float]] = []
    starts: list[float] = []
    for slope, intercept in distinct:
        start = -math.inf
        while hull:
            previous_slope, previous_intercept = hull[-1]
            start = (previous_intercept - intercept) / (slope - previous_slope)
            if start > starts[-1]:
                break
            hull.pop()
            starts.pop()
        hull.append((slope, intercept))
        starts.append(start if len(hull) > 1 else -math.inf)
    expectation = 0.0
    for index, (slope, intercept) in enumerate(hull):
        lower = starts[index]
        upper = starts[index + 1] if index + 1 < len(starts) else math.inf
        expectation += intercept * _normal_interval(lower, upper)
        expectation += slope * (_normal_pdf(lower) - _normal_pdf(upper))
    return float(expectation)


def plan_experiment(model, candidates: list[dict], *, outcome: str, budget: float,
                    direction: str = "maximize", min_information_value: float = 1e-6,
                    allow_extrapolation: bool = False) -> dict:
    """Choose an affordable measurement by correlated knowledge gradient / cost.

    ``min_information_value`` is an absolute expected decision gain in the
    outcome's units.  Cost has budget units and is never subtracted from the
    response.  A single eligible alternative has zero decision information value.
    Unsupported candidates remain visible in rankings but are excluded by default.
    """
    budget = _finite_number(budget, "budget")
    threshold = _finite_number(min_information_value, "min_information_value")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    if not isinstance(outcome, str) or not outcome.strip():
        raise ValueError("outcome must be a nonempty name")
    if not isinstance(allow_extrapolation, bool):
        raise ValueError("allow_extrapolation must be a boolean")
    identities: set[str] = set()
    for candidate in candidates:
        validate_record(candidate, require_outcomes=False)
        canonical_json(candidate)
        identity = candidate.get("candidate_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("each candidate requires a nonempty candidate_id")
        if identity in identities:
            raise ValueError(f"duplicate candidate_id: {identity}")
        identities.add(identity)
        _finite_number(candidate.get("cost"), "candidate cost", positive=True)
        if candidate["evidence_type"] != "model_prediction":
            raise ValueError("a candidate must be marked model_prediction, not experimental evidence")
        if candidate.get("outcomes") or candidate.get("outcome_units"):
            raise ValueError("unmeasured candidates must have empty outcomes and outcome_units")

    metadata = getattr(model, "metadata", {})
    result = {
        "decision": "stop", "selected_candidate": None, "selected": None,
        "rankings": [], "stop_reason": None, "outcome": outcome,
        "outcome_unit": metadata.get("outcome_units", {}).get(outcome),
        "direction": direction, "budget": budget,
        "min_information_value": threshold, "allow_extrapolation": allow_extrapolation,
        "model_id": metadata.get("model_id"), "method": "correlated_gaussian_one_step_knowledge_gradient",
        "information_value_integration": "analytic_piecewise_linear_normal_integral",
        "limitations": list(_LIMITATIONS),
    }
    if not candidates:
        result["stop_reason"] = "no_candidates"
        return result

    prediction = model.predict(candidates)
    if outcome not in prediction["outcome_names"]:
        raise ValueError(f"model has no trained output named {outcome}")
    column = prediction["outcome_names"].index(outcome)
    means = np.asarray(prediction["mean"], dtype=float)[:, column]
    epistemic = np.asarray(prediction["epistemic_variance"], dtype=float)[:, column]
    noise = np.asarray(prediction["noise_variance"], dtype=float)[:, column]
    in_domain = np.asarray(prediction["in_domain"], dtype=bool)
    count = len(candidates)
    if any(array.shape != (count,) for array in (means, epistemic, noise, in_domain)):
        raise ValueError("model returned inconsistent prediction dimensions")
    if not all(np.isfinite(array).all() for array in (means, epistemic, noise)):
        raise ValueError("model returned nonfinite predictions")
    if (epistemic < 0).any() or (noise < 0).any():
        raise ValueError("model returned negative variance")

    eligible_indices = []
    rankings = []
    for index, candidate in enumerate(candidates):
        exclusions = []
        if candidate["cost"] > budget:
            exclusions.append("over_budget")
        if not in_domain[index] and not allow_extrapolation:
            exclusions.append("out_of_domain")
        if not exclusions:
            eligible_indices.append(index)
        rankings.append({
            "candidate_id": candidate["candidate_id"], "cost": float(candidate["cost"]),
            "eligible": not exclusions, "exclusion_reasons": exclusions,
            "in_domain": bool(in_domain[index]),
            "domain_reasons": list(prediction["domain_reasons"][index]),
            "predicted_mean": float(means[index]),
            "epistemic_sd": float(math.sqrt(epistemic[index])),
            "noise_sd": float(math.sqrt(noise[index])),
            "knowledge_gradient": None, "knowledge_gradient_per_cost": None,
        })
    result["rankings"] = rankings
    if not eligible_indices:
        result["stop_reason"] = "no_affordable_supported_candidates"
        return result

    eligible = [candidates[index] for index in eligible_indices]
    covariance = np.asarray(model.covariance(eligible, outcome), dtype=float)
    size = len(eligible)
    if covariance.shape != (size, size) or not np.isfinite(covariance).all():
        raise ValueError("model returned invalid latent covariance")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, atol=1e-9 * scale, rtol=1e-7):
        raise ValueError("model covariance is not symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-8 * scale:
        raise ValueError("model covariance is not positive semidefinite")
    if not np.allclose(np.diag(covariance), epistemic[eligible_indices], rtol=1e-5, atol=1e-9 * scale):
        raise ValueError("latent covariance diagonal disagrees with epistemic variance")
    sign = 1.0 if direction == "maximize" else -1.0
    intercepts = sign * means[eligible_indices]
    current_best = float(intercepts.max())
    for local, original in enumerate(eligible_indices):
        predictive_variance = max(0.0, float(covariance[local, local])) + float(noise[original])
        if predictive_variance <= 0.0:
            value = 0.0
        else:
            slopes = sign * covariance[:, local] / math.sqrt(predictive_variance)
            value = max(0.0, _expected_max_normal_lines(intercepts, slopes) - current_best)
        rankings[original]["knowledge_gradient"] = float(value)
        rankings[original]["knowledge_gradient_per_cost"] = float(value / candidates[original]["cost"])

    rankings.sort(key=lambda item: (
        not item["eligible"],
        -(item["knowledge_gradient_per_cost"] or 0.0), item["candidate_id"],
    ))
    useful = [row for row in rankings if row["eligible"] and row["knowledge_gradient"] > threshold]
    if not useful:
        result["stop_reason"] = "insufficient_expected_decision_gain"
        return result
    selected_id = useful[0]["candidate_id"]
    selected = next(candidate for candidate in candidates if candidate["candidate_id"] == selected_id)
    result["decision"] = "experiment"
    result["selected_candidate"] = copy.deepcopy(selected)
    result["selected"] = copy.deepcopy(selected)
    result["expected_decision_gain"] = useful[0]["knowledge_gradient"]
    result["knowledge_gradient_per_cost"] = useful[0]["knowledge_gradient_per_cost"]
    result["remaining_budget_if_executed"] = float(budget - selected["cost"])
    return result
