"""Both standalone organ demonstrations are isolated, runnable and resumable."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from virtual_t_cell.learning.organ_demo import prepare_organ_inputs, run_organ_demo
from virtual_t_cell.learning.schema import read_jsonl, validate_dataset
from virtual_t_cell.learning.world_model import BayesianWorldModel


class OrganDemoTests(unittest.TestCase):
    def test_prepared_inputs_are_single_organ_and_have_independent_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for organ in ("liver", "spleen"):
                with self.subTest(organ=organ):
                    inputs = root / organ
                    report = prepare_organ_inputs(inputs, organ)
                    self.assertEqual(report["split_sizes"], {"train": 32, "new": 20, "validation": 14, "test": 14})
                    seen = set()
                    for name in ("train", "new", "validation", "test", "candidates"):
                        rows = read_jsonl(inputs / f"{name}.jsonl")
                        self.assertEqual({row["organ"] for row in rows}, {organ})
                        self.assertEqual({row["species"] for row in rows}, {"synthetic"})
                        groups = {row["group_id"] for row in rows}
                        self.assertFalse(seen & groups)
                        seen.update(groups)
                    for batch in (1, 2):
                        with (inputs / f"synthetic_metadata_{batch}.csv").open() as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual(len(rows), 8)
                        self.assertEqual({row["organ"] for row in rows}, {organ})
                    before = (inputs / "organ_inputs.json").read_bytes()
                    prepare_organ_inputs(inputs, organ)
                    self.assertEqual((inputs / "organ_inputs.json").read_bytes(), before)

    def test_full_organ_runs_fit_gate_plan_and_do_not_reset_promoted_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = {}
            for organ in ("liver", "spleen"):
                with self.subTest(organ=organ):
                    out = root / organ
                    # Inputs may be shipped in a ZIP before the first run.
                    prepare_organ_inputs(out / "inputs", organ)
                    first = run_organ_demo(out, organ)
                    self.assertEqual(first["evidence_type"], "synthetic")
                    self.assertFalse(first["biological_validation"])
                    self.assertFalse(first["demonstrated_agi"])
                    self.assertEqual(first["learning_cycle"]["status"], "promoted")
                    self.assertEqual(first["repeat_learning_cycle"]["status"], "no_new_evidence")
                    self.assertEqual(first["active_training_records"], 52)
                    self.assertEqual(first["atlas"]["n_cells"], 16)
                    self.assertEqual(first["atlas"]["organs"], {organ: 16})
                    self.assertEqual(first["journal"]["summary"]["total"], 52)
                    self.assertEqual(first["demo_model_config"]["random_features"], 0)
                    self.assertFalse(first["atlas_states_used_as_response_labels"])
                    self.assertIn(first["plan"]["decision"], {"experiment", "stop"})
                    self.assertEqual(len(first["plan"]["rankings"]), 6)
                    model = BayesianWorldModel.load(out / "active_model.npz")
                    self.assertEqual(set(model.metadata["training_coverage"]["organs"]), {organ})
                    states = read_jsonl(out / "atlas_states.jsonl")
                    self.assertEqual({row["organ"] for row in states}, {organ})
                    with self.assertRaisesRegex(ValueError, "atlas state"):
                        validate_dataset(states, allow_synthetic=True)
                    self.assertFalse(set(model.metadata["training_ids"]) & {row["observation_id"] for row in states})
                    active_bytes = (out / "active_model.npz").read_bytes()
                    second = run_organ_demo(out, organ)
                    self.assertFalse(second["initialized_active_model_this_run"])
                    self.assertEqual(second["learning_cycle"]["status"], "no_new_evidence")
                    self.assertEqual(second["repeat_learning_cycle"]["status"], "no_new_evidence")
                    self.assertEqual(second["first_learning_cycle"]["status"], "promoted")
                    self.assertEqual(second["active_model_id"], first["active_model_id"])
                    self.assertEqual((out / "active_model.npz").read_bytes(), active_bytes)
                    self.assertEqual(second["journal"]["initial_append"]["inserted"], 0)
                    self.assertEqual(second["journal"]["new_append"]["inserted"], 0)
                    self.assertEqual(second["atlas"]["n_cells"], 16)
                    stored = json.loads((out / "organ_report.json").read_text())
                    self.assertEqual(stored["active_model_id"], model.metadata["model_id"])
                    for name in ("test_predictions.csv", "candidate_rankings.csv"):
                        with (out / name).open() as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertTrue(rows)
                        self.assertEqual({row["organ"] for row in rows}, {organ})
                        self.assertEqual({row["data_type"] for row in rows}, {"SYNTHETIC"})
                        if name == "candidate_rankings.csv":
                            self.assertTrue(all(row["predicted_mean"] for row in rows))
                    ids[organ] = first["active_model_id"]
            self.assertNotEqual(ids["liver"], ids["spleen"])

    def test_other_organ_and_edited_inputs_are_rejected_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            prepare_organ_inputs(out, "liver")
            original = (out / "train.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "different organ"):
                prepare_organ_inputs(out, "spleen")
            self.assertEqual((out / "train.jsonl").read_bytes(), original)
            edited = original + b"\n"
            (out / "train.jsonl").write_bytes(edited)
            with self.assertRaisesRegex(ValueError, "modified"):
                prepare_organ_inputs(out, "liver")
            self.assertEqual((out / "train.jsonl").read_bytes(), edited)

    def test_invalid_organ_is_rejected_before_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "unused"
            with self.assertRaisesRegex(ValueError, "organ"):
                run_organ_demo(out, "blood")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
