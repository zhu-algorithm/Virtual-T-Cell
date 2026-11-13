"""Independent, CPU-only organ packages exercising the SYNTHETIC learning flow.

``prepare_organ_inputs(directory, organ)`` writes portable demonstration inputs
directly below ``directory``. ``run_organ_demo(output_dir, organ)`` uses its
``inputs/`` child and writes the following stable output paths::

    atlas.sqlite, evidence.sqlite, initial_model.npz, active_model.npz
    active_model.npz.belief.json, atlas_states.jsonl, atlas_belief.json
    learning/first_cycle.json, learning/attempts/, learning/versions/
    test_evaluation.json, experiment_plan.json, test_predictions.csv
    candidate_rankings.csv, organ_report.json

All example data and reported performance concern a numerical synthetic fixture.
No organ-specific biological predictive accuracy, wet-lab execution, or AGI is
established. Repeated runs retain the active model and consume no duplicate data.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .atlas import AtlasStore
from .belief import BeliefState
from .continual import evaluate_model
from .demo import make_response_fixture
from .evidence import EvidenceStore
from .pipeline import sync_inbox
from .planner import plan_experiment
from .runner import run_learning_cycle
from .schema import canonical_json, read_jsonl
from .state import summarize_atlas_states
from .world_model import BayesianWorldModel


_INPUT_FORMAT = "virtual_t_cell.synthetic_organ_inputs.v1"
_MODEL_CONFIG = {"seed": 0, "random_features": 0, "prior_precision": 1.0}


def _organ(value: str) -> str:
    if value not in {"liver", "spleen"}:
        raise ValueError("organ must be exactly 'liver' or 'spleen'")
    return value


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(records: list[dict]) -> bytes:
    return ("".join(canonical_json(row) + "\n" for row in records)).encode("utf-8")


def _csv_bytes(header: list[str], rows: list[list]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _immutable_input(path: Path, data: bytes) -> None:
    """Do not silently replace user-edited inputs or another organ's fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"demo input differs from this organ/seed fixture; use a fresh output directory: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def _output(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_organ_inputs(output_dir: str | Path, organ: str, seed: int = 7) -> dict:
    """Prepare one organ's immutable synthetic counts, responses and candidates.

    Response groups are split once into 32 initial training, 20 arriving evidence,
    14 validation and 14 final test groups. The two sequencing batches use
    different donors at the same biological time. Batch order is not a cell
    trajectory. Existing matching inputs are reusable after extracting a ZIP.
    """
    organ = _organ(organ)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    directory = Path(output_dir)
    identity_path = directory / "organ_inputs.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if (not isinstance(identity, dict) or identity.get("format") != _INPUT_FORMAT
                or identity.get("organ") != organ or identity.get("seed") != seed):
            raise ValueError("input directory belongs to a different organ, seed or format")
    source = f"synthetic_{organ}_response_fixture_v1"
    rows = [row for row in make_response_fixture(seed=seed, groups=80) if row["organ"] == organ]
    for row in rows:
        row["source"] = source
        row["source_record_id"] = row["observation_id"]
    splits = {"train": rows[:32], "new": rows[32:52], "validation": rows[52:66], "test": rows[66:]}
    files = {f"{name}.jsonl": _jsonl_bytes(split) for name, split in splits.items()}

    candidates = []
    for index in range(6):
        x = -0.4 + index * 0.16
        # Close numerical alternatives exercise information-value ranking. This
        # relation belongs to the fixture and is not a proposed biological law.
        candidates.append({"observation_id": f"candidate:{organ}:{index}", "candidate_id": f"candidate:{organ}:{index}",
                           "experiment_id": f"future:{organ}:{index}", "group_id": f"future:{organ}:{index}",
                           "organ": organ, "cell_type": "CD8_T", "condition": "fixture_context", "species": "synthetic",
                           "intervention": f"synthetic_candidate_{index}", "evidence_type": "model_prediction",
                           "source": f"synthetic_{organ}_candidate_specification", "features": {
                               "descriptor_x": x, "descriptor_z": 2 * x},
                           "outcomes": {}, "outcome_units": {}, "cost": float(1 + index % 3)})
    files["candidates.jsonl"] = _jsonl_bytes(candidates)

    genes = ["CD3D", "CD8A", "MS4A1", "ALB", "LYZ", "MT-CO1"]
    for batch in (1, 2):
        counts, metadata = [], []
        for index in range(8):
            if index < 6:
                cell_type, values = "CD8_T", [5 + index + batch, 4 + batch, 0, 0, 0, 1]
            elif index == 6:
                cell_type, values = "macrophage", [0, 0, 0, 0, 15 + batch, 1]
            elif organ == "liver":
                cell_type, values = "hepatocyte", [0, 0, 0, 20 + batch, 0, 1]
            else:
                cell_type, values = "B_cell", [0, 0, 20 + batch, 0, 0, 1]
            cell = f"barcode-{index}"
            counts.append([cell, *values])
            metadata.append([cell, f"synthetic_{organ}_sample_{batch}", f"synthetic_{organ}_donor_{batch}",
                             organ, cell_type, "synthetic_unperturbed", "synthetic", 0])
        counts_name, metadata_name = f"synthetic_counts_{batch}.csv", f"synthetic_metadata_{batch}.csv"
        files[counts_name] = _csv_bytes(["cell_id", *genes], counts)
        files[metadata_name] = _csv_bytes(
            ["cell_id", "sample_id", "donor_id", "organ", "cell_type", "condition", "species", "biological_time_hours"], metadata)
        manifest = {"schema_version": 1, "batch_id": f"synthetic_{organ}_batch_{batch}",
                    "source": f"synthetic_{organ}_atlas_fixture_v1", "atlas_evidence_type": "synthetic",
                    "atlas": {"counts_csv": f"../{counts_name}", "metadata_csv": f"../{metadata_name}",
                              "min_counts": 1, "min_genes": 1, "max_mito_fraction": 0.5}}
        files[f"inbox/batch_{batch}.json"] = _json_bytes(manifest)
    manifest = {"format": _INPUT_FORMAT, "organ": organ, "seed": seed, "evidence_type": "synthetic",
                "biological_validation": False, "source": source,
                "description": "SYNTHETIC software integration fixture; not real sequencing or biological validation.",
                "split_sizes": {name: len(split) for name, split in splits.items()},
                "sequencing_batches": 2, "synthetic_cells_per_batch": 8,
                "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}}
    # Check all existing files before creating any missing files on a resumed run.
    complete = {**files, "organ_inputs.json": _json_bytes(manifest)}
    for name, data in complete.items():
        target = directory / name
        if target.exists() and (not target.is_file() or target.read_bytes() != data):
            raise ValueError(f"demo input was modified or belongs to another fixture: {target}")
    for name, data in complete.items():
        _immutable_input(directory / name, data)
    return {**manifest, "directory": str(directory), "inbox": str(directory / "inbox")}


def _scope_model(model: BayesianWorldModel, organ: str) -> None:
    metadata = model.metadata
    if (set(metadata["training_coverage"]["organs"]) != {organ}
            or set(metadata["evidence_types"]) != {"synthetic"}
            or set(metadata["sources"]) != {f"synthetic_{organ}_response_fixture_v1"}
            or metadata.get("fit_config") != _MODEL_CONFIG):
        raise ValueError("existing model does not belong to this organ's synthetic demo; use a separate output directory")


def run_organ_demo(output_dir: str | Path, organ: str) -> dict:
    """Execute or resume a single-organ example, preserving a promoted model.

    The model uses a linear Bayesian configuration because the supplied law is
    linear. This demonstrates software wiring and probabilistic parameter
    fitting, not a nonlinear model advantage or real organ-response accuracy.
    Validation gates are still enforced; a rejection remains visible and is
    never overridden to make a demonstration succeed.
    """
    organ = _organ(organ)
    output = Path(output_dir)
    inputs = prepare_organ_inputs(output / "inputs", organ)
    train = read_jsonl(output / "inputs/train.jsonl")
    new = read_jsonl(output / "inputs/new.jsonl")
    test = read_jsonl(output / "inputs/test.jsonl")
    candidates = read_jsonl(output / "inputs/candidates.jsonl")
    active_path, initial_path = output / "active_model.npz", output / "initial_model.npz"
    initialized = False
    if active_path.is_file():
        current = BayesianWorldModel.load(active_path)
        _scope_model(current, organ)
    else:
        if initial_path.is_file():
            current = BayesianWorldModel.load(initial_path)
            _scope_model(current, organ)
            if set(current.metadata["training_ids"]) != {row["observation_id"] for row in train}:
                raise ValueError("saved initial model does not match this fixture's training split")
        else:
            current = BayesianWorldModel.fit(train, **_MODEL_CONFIG, allow_synthetic=True)
            current.save(initial_path)
        current.save(active_path)
        _output(Path(str(active_path) + ".belief.json"), _json_bytes(BeliefState.from_model(current).to_dict()))
        initialized = True
    model_before = current.metadata["model_id"]

    atlas_path, evidence_path = output / "atlas.sqlite", output / "evidence.sqlite"
    ingestion = sync_inbox(inputs["inbox"], atlas_path=atlas_path, evidence_path=evidence_path)
    if ingestion["errors"]:
        raise ValueError("synthetic atlas ingestion failed: " + canonical_json(ingestion["errors"]))
    repeat_ingestion = sync_inbox(inputs["inbox"], atlas_path=atlas_path, evidence_path=evidence_path)
    if repeat_ingestion["errors"]:
        raise ValueError("repeated synthetic atlas ingestion failed: " + canonical_json(repeat_ingestion["errors"]))
    with AtlasStore(atlas_path) as atlas:
        summary = atlas.summary()
        states = atlas.pseudobulk_records()
    if set(summary["organs"]) != {organ} or any(row["organ"] != organ for row in states):
        raise ValueError("atlas contains a different organ; organ packages require independent output directories")
    state_belief = summarize_atlas_states(states)
    _output(output / "atlas_states.jsonl", _jsonl_bytes(states))
    _output(output / "atlas_belief.json", _json_bytes(state_belief))

    # State snapshots stay in the atlas. Only independently identified synthetic
    # response observations are offered to the response-learning journal.
    with EvidenceStore(evidence_path) as journal:
        if any(row["organ"] != organ or row["evidence_type"] != "synthetic"
               or row["source"] != f"synthetic_{organ}_response_fixture_v1" for row in journal.records()):
            raise ValueError("evidence journal does not belong to this organ's synthetic response demo")
        initial_append = journal.append(train)
        new_append = journal.append(new)
        journal_summary = journal.summary()
    cycle_arguments = {"model_path": active_path, "evidence_path": evidence_path,
                       "validation_path": output / "inputs/validation.jsonl", "output_dir": output / "learning",
                       "allow_synthetic": True, "max_relative_regression": 0.05, "promote": True}
    cycle = run_learning_cycle(**cycle_arguments)
    first_cycle_path = output / "learning/first_cycle.json"
    if first_cycle_path.is_file():
        first_cycle = json.loads(first_cycle_path.read_text(encoding="utf-8"))
    else:
        first_cycle = cycle
        _output(first_cycle_path, _json_bytes(first_cycle))
    repeated_cycle = run_learning_cycle(**cycle_arguments)
    active = BayesianWorldModel.load(active_path)
    _scope_model(active, organ)
    belief = BeliefState.from_model(active).to_dict()
    _output(Path(str(active_path) + ".belief.json"), _json_bytes(belief))
    evaluation = evaluate_model(active, test)
    plan = plan_experiment(active, candidates, outcome="response_a", budget=3)
    _output(output / "test_evaluation.json", _json_bytes(evaluation))
    _output(output / "experiment_plan.json", _json_bytes(plan))

    prediction = active.predict(test)
    prediction_rows = []
    for index, record in enumerate(test):
        for column, outcome in enumerate(prediction["outcome_names"]):
            prediction_rows.append(["SYNTHETIC", record["observation_id"], organ, outcome,
                                    record["outcomes"][outcome], float(prediction["mean"][index, column]),
                                    float(np.sqrt(prediction["epistemic_variance"][index, column])),
                                    float(np.sqrt(prediction["noise_variance"][index, column])),
                                    bool(prediction["in_domain"][index]), record["outcome_units"][outcome]])
    _output(output / "test_predictions.csv", _csv_bytes(
        ["data_type", "observation_id", "organ", "outcome", "observed_synthetic_value", "predicted_mean",
         "approximate_model_sd", "noise_sd", "in_domain", "unit"], prediction_rows))
    ranking_fields = ["candidate_id", "cost", "predicted_mean", "epistemic_sd", "noise_sd", "in_domain", "eligible",
                      "knowledge_gradient", "knowledge_gradient_per_cost"]
    _output(output / "candidate_rankings.csv", _csv_bytes(
        ["data_type", "organ", *ranking_fields],
        [["SYNTHETIC", organ, *[row.get(name) for name in ranking_fields]] for row in plan["rankings"]]))

    report = {"format": "virtual_t_cell.synthetic_organ_report.v1", "organ": organ, "evidence_type": "synthetic",
              "display_banner": "SYNTHETIC DEMONSTRATION — SOFTWARE TEST, NOT BIOLOGICAL VALIDATION",
              "biological_validation": False, "demonstrated_agi": False,
              "inputs": inputs, "demo_model_config": {**_MODEL_CONFIG, "model_family": "Bayesian linear regression",
                  "reason": "The explicitly synthetic numerical fixture is linear. Zero random features test fitting and the pipeline without claiming a nonlinear advantage."},
              "initialized_active_model_this_run": initialized, "model_id_before_run": model_before,
              "active_model_id": active.metadata["model_id"], "active_training_records": active.metadata["training_count"],
              "ingestion": ingestion, "repeat_ingestion": repeat_ingestion, "atlas": summary,
              "state_belief": state_belief, "response_belief": belief,
              "journal": {"initial_append": initial_append, "new_append": new_append, "summary": journal_summary},
              "atlas_states_used_as_response_labels": False,
              "first_learning_cycle": first_cycle, "learning_cycle": cycle, "repeat_learning_cycle": repeated_cycle,
              "test_evaluation": evaluation, "plan": plan,
              "artifacts": {"inputs": "inputs/organ_inputs.json", "active_model": "active_model.npz",
                            "response_belief": "active_model.npz.belief.json", "atlas_states": "atlas_states.jsonl",
                            "atlas_belief": "atlas_belief.json", "first_learning_cycle": "learning/first_cycle.json",
                            "test_evaluation": "test_evaluation.json", "experiment_plan": "experiment_plan.json",
                            "test_predictions": "test_predictions.csv", "candidate_rankings": "candidate_rankings.csv",
                            "report": "organ_report.json"},
              "limitations": ["All cells, counts and response labels in this package example are synthetic fixtures.",
                              "Reported accuracy is on a numerical software test, not organ-specific biological validation.",
                              "No sequencing, laboratory experiment, causal identification or AGI is performed.",
                              "Atlas state estimates are descriptive donor-weighted means, not a learned cell foundation model.",
                              "Model uncertainty is conditional and approximate; interval calibration is not established.",
                              "Separate biological validation and real, matched intervention evidence are required for real applications."]}
    _output(output / "organ_report.json", _json_bytes(report))
    return report
