"""Integration checks for immutable evidence, repeated sequencing batches, and CLI."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from virtual_t_cell.learning.cli import main, run_demo
from virtual_t_cell.learning.demo import make_response_fixture, write_demo_inputs
from virtual_t_cell.learning.evidence import EvidenceStore
from virtual_t_cell.learning.pipeline import sync_inbox
from virtual_t_cell.learning.schema import validate_dataset, validate_record, write_jsonl
from virtual_t_cell.learning.state import summarize_atlas_states


class EvidenceContractTests(unittest.TestCase):
    def test_units_nonfinite_and_evidence_types(self):
        row = make_response_fixture(groups=1)[0]
        with self.assertRaisesRegex(ValueError, "synthetic"):
            validate_dataset([row])
        validate_dataset([row], allow_synthetic=True)
        bad = copy.deepcopy(row)
        bad["features"]["descriptor_x"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_record(bad)
        bad = copy.deepcopy(row)
        bad["outcome_units"] = {}
        with self.assertRaisesRegex(ValueError, "units"):
            validate_record(bad)
        bad = copy.deepcopy(row)
        bad["evidence_type"] = "observational"
        with self.assertRaisesRegex(ValueError, "observational"):
            validate_dataset([bad], allow_synthetic=True)

    def test_journal_idempotence_alias_conflict_and_atomicity(self):
        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                rows = make_response_fixture(groups=2)
                self.assertEqual(store.append(rows[:1])["inserted"], 1)
                self.assertEqual(store.append(rows[:1])["repeated"], 1)
                alias = copy.deepcopy(rows[0])
                alias["observation_id"] = "renamed_original"
                self.assertEqual(store.append([alias])["repeated"], 1)
                conflict = copy.deepcopy(rows[0])
                conflict["outcomes"]["response_a"] += 1
                before = store.summary()["snapshot_hash"]
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    store.append([rows[1], conflict])
                self.assertEqual(store.summary()["snapshot_hash"], before)
                self.assertEqual(store.summary()["total"], 1)
                forecast = copy.deepcopy(rows[1])
                forecast["evidence_type"] = "model_prediction"
                with self.assertRaisesRegex(ValueError, "predictions"):
                    store.append([forecast])

    def test_initial_fit_cannot_recount_renamed_source_without_source_record_id(self):
        row = make_response_fixture(groups=1)[0]
        alias = copy.deepcopy(row)
        alias["observation_id"] = "renamed"
        with self.assertRaisesRegex(ValueError, "source evidence"):
            validate_dataset([row, alias], allow_synthetic=True)


class PipelineTests(unittest.TestCase):
    def test_state_uncertainty_uses_donors_not_cells(self):
        def snapshot(identifier, donor, value):
            return {"source_record_id": identifier, "snapshot_hash": identifier, "source": "synthetic",
                    "evidence_type": "synthetic", "state_snapshot": True, "response_training_eligible": False,
                    "organ": "liver", "cell_type": "CD8_T", "condition": "test", "gene_panel_id": "panel",
                    "donor_id": donor, "sample_id": identifier, "n_cells": 1000,
                    "outcomes": {"G": value}, "outcome_units": {"G": "log1p_cp10k"}}
        first = snapshot("s1", "d1", 2.0)
        another_same_donor = snapshot("s2", "d1", 4.0)
        single = summarize_atlas_states([first, another_same_donor])["profiles"][0]
        self.assertEqual(single["n_donors"], 1)
        self.assertIsNone(single["mean_standard_error"])
        second = snapshot("s3", "d2", 7.0)
        profile = summarize_atlas_states([first, another_same_donor, second])["profiles"][0]
        self.assertEqual(profile["n_donors"], 2)
        self.assertAlmostEqual(profile["gene_means"]["G"], 5.0)
        self.assertAlmostEqual(profile["mean_standard_error"]["G"], 2.0)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            summarize_atlas_states([first, first])

    def test_liver_spleen_batches_idempotent_and_input_changes_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_demo_inputs(root / "inputs")
            kwargs = {"atlas_path": root / "atlas.sqlite", "evidence_path": root / "evidence.sqlite"}
            first = sync_inbox(root / "inputs/inbox", **kwargs)
            self.assertFalse(first["errors"])
            self.assertEqual(first["processed"], 2)
            second = sync_inbox(root / "inputs/inbox", **kwargs)
            self.assertTrue(all(item["status"] == "already_complete" for item in second["reports"]))
            from virtual_t_cell.learning.atlas import AtlasStore
            with AtlasStore(root / "atlas.sqlite") as atlas:
                report = atlas.summary()
                self.assertEqual(report["n_cells"], 16)
                self.assertEqual(set(report["organs"]), {"liver", "spleen"})
                self.assertFalse(report["response_model_updated"])
            counts = root / "inputs/synthetic_counts_1.csv"
            counts.write_text(counts.read_text() + "new-cell,1,1,0,0\n")
            changed = sync_inbox(root / "inputs/inbox", **kwargs)
            self.assertEqual(len(changed["errors"]), 1)

    def test_rebuilt_atlas_reingests_and_bad_manifest_isolated(self):
        from virtual_t_cell.learning.atlas import AtlasStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_demo_inputs(root / "inputs")
            inbox = root / "inputs/inbox"
            journal = root / "evidence.sqlite"
            sync_inbox(inbox, atlas_path=root / "first.sqlite", evidence_path=journal)
            second = sync_inbox(inbox, atlas_path=root / "second.sqlite", evidence_path=journal)
            self.assertFalse(second["errors"])
            with AtlasStore(root / "second.sqlite") as atlas:
                self.assertEqual(atlas.summary()["n_cells"], 16)
            (inbox / "bad.json").write_text("[]")
            result = sync_inbox(inbox, atlas_path=root / "second.sqlite", evidence_path=journal)
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(result["processed"], 2)

    def test_full_synthetic_cli_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_demo(directory)
            self.assertEqual(report["evidence_type"], "synthetic")
            self.assertFalse(report["biological_validation"])
            self.assertIn(report["update_gate"]["decision"], {"accept", "reject"})
            self.assertIn(report["plan"]["decision"], {"experiment", "stop"})
            self.assertFalse(report["ingestion"]["errors"])
            self.assertEqual(report["continual_cycle"]["status"], "promoted")
            self.assertEqual(report["repeated_continual_cycle"]["status"], "no_new_evidence")
            self.assertTrue((Path(directory) / "demo_report.json").is_file())
            # Replay never becomes an independent evaluation split.
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                code = main(["evaluate", "--model", str(Path(directory) / "initial_model.npz"),
                             "--records", str(Path(directory) / "inputs/train.jsonl"),
                             "--out", str(Path(directory) / "leaky.json")])
            self.assertEqual(code, 2)
            self.assertFalse((Path(directory) / "leaky.json").exists())

    def test_inbox_cli_triggers_gated_learning_from_relative_config(self):
        from virtual_t_cell.learning.world_model import BayesianWorldModel
        from virtual_t_cell.learning.schema import read_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_demo_inputs(root / "inputs")
            model = BayesianWorldModel.fit(read_jsonl(root / "inputs/train.jsonl"), random_features=8, allow_synthetic=True)
            model.save(root / "current.npz")
            config = {"model_path": "current.npz", "replay_path": "inputs/train.jsonl",
                      "validation_path": "inputs/validation.jsonl", "output_dir": "updates",
                      "allow_synthetic": True, "promote": True}
            (root / "learning.json").write_text(json.dumps(config))
            manifest = {"schema_version": 1, "batch_id": "responses_001",
                        "source": "synthetic_numerical_fixture_v1", "responses_jsonl": "../new.jsonl"}
            (root / "inputs/inbox/responses.json").write_text(json.dumps(manifest))
            arguments = ["sync-inbox", "--inbox", str(root / "inputs/inbox"),
                         "--atlas", str(root / "atlas.sqlite"), "--evidence", str(root / "evidence.sqlite"),
                         "--learning-config", str(root / "learning.json"),
                         "--out", str(root / "last.json")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
                first = json.loads((root / "last.json").read_text())
                self.assertEqual(first["continual_learning"]["status"], "promoted")
                self.assertTrue(first["response_model_automatically_promoted"])
                self.assertEqual(main(arguments), 0)
            second = json.loads((root / "last.json").read_text())
            self.assertEqual(second["continual_learning"]["status"], "no_new_evidence")


if __name__ == "__main__":
    unittest.main()
