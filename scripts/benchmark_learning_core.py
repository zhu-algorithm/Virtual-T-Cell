"""Reproduce a strictly retrospective learning-core check on the repository NPZ.

The marker panel, split seed, and model settings are fixed in advance. Target
groups, not individual condition rows, are split. Donor-independent evaluation
is impossible with this aggregate archive and is not claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from virtual_t_cell.learning.adapters import import_reference_npz
from virtual_t_cell.learning.cli import write_report
from virtual_t_cell.learning.continual import evaluate_model, propose_update
from virtual_t_cell.learning.schema import content_hash, write_jsonl
from virtual_t_cell.learning.world_model import BayesianWorldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="models/gse314342_primary_cd4_virtual_t_cell.npz")
    parser.add_argument("--out-dir", default="run_output/reference_benchmark")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, provenance = import_reference_npz(args.reference, organ="blood", cell_type="CD4 T cell")
    groups = sorted({record["group_id"] for record in records}, key=lambda name: content_hash([17, name]))
    n = len(groups)
    if n < 40:
        raise ValueError("this reference protocol requires at least 40 independent target groups")
    boundaries = [0, int(n * 0.60), int(n * 0.75), int(n * 0.875), n]
    splits = {}
    split_groups = {}
    for index, name in enumerate(("initial", "new", "validation", "test")):
        selected = set(groups[boundaries[index]:boundaries[index + 1]])
        split_groups[name] = sorted(selected)
        splits[name] = [record for record in records if record["group_id"] in selected]
        write_jsonl(output / f"{name}.jsonl", splits[name])
    initial = BayesianWorldModel.fit(splits["initial"], seed=0, random_features=16)
    candidate, gate = propose_update(initial, splits["initial"], splits["new"], splits["validation"])
    active = candidate if gate["accepted"] else initial
    initial.save(output / "initial_model.npz")
    candidate.save(output / "candidate_model.npz")
    active.save(output / "active_model.npz")
    active_training = splits["initial"] + (splits["new"] if gate["accepted"] else [])
    # No nonlinear features: transparent linear/ridge alternative with the same
    # Bayesian noise treatment and the SAME training evidence as the active model.
    linear = BayesianWorldModel.fit(active_training, seed=0, random_features=0)
    test = evaluate_model(active, splits["test"])
    linear_test = evaluate_model(linear, splits["test"])
    zero_error = [value * value for row in splits["test"] for value in row["outcomes"].values()]
    target_features = {row["group_id"]: row["features"] for row in records}
    feature_patterns = Counter(content_hash(values) for values in target_features.values())
    report = {
        "benchmark": "fixed_panel_target_group_holdout_reference_summary_replay_v1",
        "evidence_type": "experimental_summary", "source": provenance,
        "protocol": {"split_hash_seed": 17, "model_seed": 0, "random_features": 16,
                     "prior_precision": 1.0, "gate_max_relative_regression": 0.05,
                     "held_out_unit": "perturbation target (all three condition summaries stay together)",
                     "split_groups": split_groups,
                     "split_record_counts": {name: len(rows) for name, rows in splits.items()},
                     "test_used_to_select_model": False},
        "feature_support": {"target_count": len(target_features), "distinct_target_feature_vectors": len(feature_patterns),
                            "largest_indistinguishable_target_group": max(feature_patterns.values()),
                            "description": "Fixed pathway/network priors only. No contrast-derived baseMean or response values used as inputs."},
        "active_model_id": active.metadata["model_id"], "update_gate": gate,
        "held_out_test": test, "linear_baseline_test": linear_test,
        "zero_effect_baseline_test_rmse": float(np.sqrt(np.mean(zero_error))),
        "limitations": [
            "Retrospective replay of a selected, processed public summary archive; not raw single-cell foundation-model training.",
            "Targets are held out; donors and the original study are NOT independent across splits.",
            "The parent archive already selected targets/genes using global data, so this is not an unbiased discovery benchmark.",
            "Sparse prior annotations make most targets indistinguishable. A fitted surrogate is not evidence of general mechanism learning.",
            "Blood CD4 T cell evidence does not establish liver or spleen response accuracy.",
            "A rejection is retained as a real result; the held-out test does not change the gate or hyperparameters.",
        ],
    }
    write_report(output / "benchmark.json", report)
    summary = {"records": len(records), "targets": n, "unique_feature_vectors": len(feature_patterns),
               "gate": gate["decision"], "gate_reasons": gate["reasons"],
               "test_rmse": test["overall"]["rmse"], "mean_baseline_rmse": test["overall"]["mean_baseline_rmse"],
               "linear_baseline_rmse": linear_test["overall"]["rmse"],
               "zero_baseline_rmse": report["zero_effect_baseline_test_rmse"],
               "test_90_coverage": test["overall"]["approximate_gaussian_90_coverage"],
               "test_domain": test["coverage"], "output": str(output / "benchmark.json")}
    write_report(output / "summary.json", summary)
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
