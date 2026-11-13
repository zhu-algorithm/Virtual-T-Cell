"""End-to-end response-cycle tests using synthetic software fixtures only."""
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from virtual_t_cell.learning.belief import BeliefState
from virtual_t_cell.learning.continual import propose_update
from virtual_t_cell.learning.evidence import EvidenceStore
from virtual_t_cell.learning.runner import _model_lock, run_learning_cycle
from virtual_t_cell.learning.schema import write_jsonl
from virtual_t_cell.learning.world_model import BayesianWorldModel


def records(prefix, inputs, *, shift=0.0, evidence_type="synthetic"):
    rows = []
    for index, x in enumerate(inputs):
        for organ in ("liver", "spleen"):
            identity = f"{prefix}:{organ}:{index}"
            rows.append({
                "observation_id": identity, "experiment_id": f"experiment:{identity}",
                "group_id": f"independent_group:{identity}",
                "organ": organ, "species": "synthetic", "cell_type": "CD8_T",
                "condition": "runner_software_fixture", "intervention": "abstract_input",
                "source": "synthetic_runner_unittest_not_biological_data",
                "evidence_type": evidence_type, "features": {"x": float(x)},
                "outcomes": {"response": float(2.0 * x + (0.4 if organ == "spleen" else 0.0) + shift)},
                "outcome_units": {"response": "synthetic_units"},
                "measurement_sd": {"response": 0.05},
            })
    return rows


def _hold_model_lock_in_process(model_path, connection):
    """Top-level spawn target, exercising the actual platform's lock backend."""
    try:
        with _model_lock(Path(model_path)):
            connection.send("locked")
            command = connection.recv()
            if command == "abort":
                # Deliberately bypass Python finally blocks: the process handle
                # lifecycle must not leave a stale persistent-file lock.
                os._exit(23)
    except BaseException as error:
        try:
            connection.send({"error": repr(error)})
        finally:
            raise
    finally:
        connection.close()


def _close_lock_holder(process, connection):
    if process.is_alive():
        try:
            connection.send("release")
        except (OSError, EOFError):
            pass
        process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    connection.close()
    process.close()


class LearningCycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "current.npz"
        self.evidence_path = self.root / "evidence.sqlite"
        self.validation_path = self.root / "validation.jsonl"
        self.replay_path = self.root / "replay.jsonl"
        self.output_dir = self.root / "learning"
        self.initial = records("old", [-2.0, -1.0, 0.0, 1.0, 2.0])
        self.new = records("new", np.linspace(-1.8, 1.8, 8))
        self.validation = records("validation", [-1.6, -0.5, 0.5, 1.6])
        self.current = BayesianWorldModel.fit(self.initial, random_features=0, allow_synthetic=True)
        self.current.save(self.model_path)
        self.original_bytes = self.model_path.read_bytes()
        self.belief_path = Path(str(self.model_path) + ".belief.json")
        self.belief_path.write_text(json.dumps(BeliefState.from_model(self.current).to_dict()), encoding="utf-8")
        write_jsonl(self.replay_path, self.initial)
        write_jsonl(self.validation_path, self.validation)

    def append(self, rows):
        with EvidenceStore(self.evidence_path) as journal:
            return journal.append(rows)

    def cycle(self, *, seed=True, **options):
        settings = dict(model_path=self.model_path, evidence_path=self.evidence_path,
                        validation_path=self.validation_path, output_dir=self.output_dir,
                        allow_synthetic=True, replay_path=self.replay_path if seed else None)
        settings.update(options)
        return run_learning_cycle(**settings)

    def lock_holder(self):
        # Always use spawn, even on Linux, so these tests use the same import and
        # process isolation behavior as native Windows CI. No fcntl is imported
        # by the test; the runner must select its real platform backend.
        context = multiprocessing.get_context("spawn")
        connection, child = context.Pipe()
        process = context.Process(target=_hold_model_lock_in_process,
                                  args=(str(self.model_path), child), daemon=True)
        process.start()
        child.close()
        self.addCleanup(_close_lock_holder, process, connection)
        self.assertTrue(connection.poll(15), "lock holder did not become ready")
        self.assertEqual(connection.recv(), "locked")
        return process, connection

    def test_accepted_cycle_promotes_and_refreshes_belief_then_has_no_pending_labels(self):
        self.append(self.new)
        report = self.cycle()
        self.assertEqual(report["status"], "promoted")
        self.assertTrue(report["promoted"])
        self.assertTrue(report["accepted"])
        self.assertEqual(report["new_evidence_count"], len(self.new))
        self.assertEqual(report["replay_count"], len(self.initial))
        active = BayesianWorldModel.load(self.model_path)
        self.assertEqual(active.metadata["model_id"], report["candidate_model_id"])
        self.assertEqual(set(active.metadata["training_ids"]),
                         {row["observation_id"] for row in self.initial + self.new})
        self.assertEqual(Path(report["previous_model_path"]).read_bytes(), self.original_bytes)
        self.assertEqual(json.loads(self.belief_path.read_text())["model_id"], active.metadata["model_id"])
        self.assertTrue(Path(report["previous_belief_path"]).is_file())
        self.assertTrue(Path(report["gate_report_path"]).is_file())
        self.assertTrue(Path(report["attempt_report_path"]).is_file())
        with patch.object(BayesianWorldModel, "fit", side_effect=AssertionError("no retraining expected")):
            repeated = self.cycle(seed=False)
        self.assertEqual(repeated["status"], "no_new_evidence")
        self.assertFalse(repeated["promoted"])
        self.assertEqual(repeated["new_evidence_count"], 0)

    def test_rejected_candidate_is_kept_and_repeated_poll_does_not_retrain(self):
        bad = records("bad", np.linspace(-1.8, 1.8, 8), shift=8.0)
        self.append(bad)
        rejected = self.cycle()
        self.assertEqual(rejected["status"], "rejected")
        self.assertFalse(rejected["accepted"])
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)
        candidate = BayesianWorldModel.load(rejected["candidate_model_path"])
        self.assertEqual(len(candidate.metadata["training_ids"]), len(self.initial + bad))
        gate = json.loads(Path(rejected["gate_report_path"]).read_text())
        self.assertFalse(gate["accepted"])
        with patch.object(BayesianWorldModel, "fit", side_effect=AssertionError("same rejected evidence was refitted")):
            repeated = self.cycle(seed=False)
        self.assertEqual(repeated["status"], "already_attempted")
        self.assertEqual(repeated["previous_status"], "rejected")
        self.assertEqual(repeated["attempt_key"], rejected["attempt_key"])
        with EvidenceStore(self.evidence_path) as journal:
            self.assertEqual(journal.summary()["total"], len(self.initial + bad))

    def test_next_batch_combines_all_pending_evidence_after_a_rejection(self):
        bad = records("bad", [-1.8, -0.6, 0.6, 1.8], shift=8.0)
        self.append(bad)
        first = self.cycle()
        fresh = records("later", [-1.5, 1.5])
        self.append(fresh)
        second = self.cycle(seed=False)
        self.assertNotEqual(first["attempt_key"], second["attempt_key"])
        self.assertEqual(second["new_evidence_count"], len(bad + fresh))
        candidate = BayesianWorldModel.load(second["candidate_model_path"])
        self.assertEqual(set(candidate.metadata["training_ids"]),
                         {row["observation_id"] for row in self.initial + bad + fresh})

    def test_missing_replay_is_explicit_and_recoverable_without_losing_new_records(self):
        self.append(self.new)
        with patch.object(BayesianWorldModel, "fit", side_effect=AssertionError("incomplete replay must not train")):
            missing = self.cycle(seed=False)
        self.assertEqual(missing["status"], "rejected")
        self.assertIn("missing_replay_training_records", missing["gate"]["reasons"])
        self.assertEqual(set(missing["missing_replay_ids"]), {row["observation_id"] for row in self.initial})
        self.assertIsNone(missing["candidate_model_path"])
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)
        recovered = self.cycle()
        self.assertEqual(recovered["status"], "promoted")
        self.assertNotEqual(missing["attempt_key"], recovered["attempt_key"])

    def test_observational_states_and_prediction_rows_are_explicitly_excluded(self):
        observational = records("atlas", [0.2], evidence_type="observational")
        snapshots = records("snapshot", [0.1])
        for row in snapshots:
            row["response_training_eligible"] = False
            row["state_snapshot"] = True
        prediction = records("prediction", [0.3], evidence_type="model_prediction")
        for row in prediction:
            row.update(outcomes={}, outcome_units={}, measurement_sd={})
        self.append(self.new + observational + snapshots)
        # The journal itself rejects predictions.  The optional seed reader also
        # reports exclusions, so historical mixed export files cannot train them.
        with self.assertRaisesRegex(ValueError, "predictions"):
            self.append(records("fake_prediction", [0.4], evidence_type="model_prediction"))
        write_jsonl(self.replay_path, self.initial + prediction)
        result = self.cycle()
        self.assertEqual(result["status"], "promoted")
        self.assertEqual(result["new_evidence_count"], len(self.new))
        ignored = result["ignored_evidence"]["by_reason"]
        self.assertEqual(ignored["model_predictions_are_not_observed_evidence"], 2)
        self.assertEqual(ignored["observational_atlas_is_not_response_training_evidence"], 2)
        self.assertEqual(ignored["state_snapshot_is_not_response_training_evidence"], 2)
        active = BayesianWorldModel.load(self.model_path)
        self.assertEqual(set(active.metadata["training_ids"]),
                         {row["observation_id"] for row in self.initial + self.new})

    def test_validation_leakage_is_rejected_and_cached_before_training(self):
        self.append(self.new)
        validation = deepcopy(self.validation)
        validation[0]["group_id"] = self.initial[0]["group_id"]
        write_jsonl(self.validation_path, validation)
        with patch.object(BayesianWorldModel, "fit", side_effect=AssertionError("leaked validation must not train")):
            result = self.cycle()
            repeated = self.cycle(seed=False)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("overlap", result["gate"]["error"])
        self.assertEqual(repeated["status"], "already_attempted")
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)

    def test_reviewed_accepted_candidate_can_be_promoted_without_refitting(self):
        self.append(self.new)
        proposal = self.cycle(promote=False)
        self.assertEqual(proposal["status"], "accepted_not_promoted")
        self.assertTrue(proposal["accepted"])
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)
        with patch.object(BayesianWorldModel, "fit", side_effect=AssertionError("reuse the accepted candidate")):
            promoted = self.cycle(seed=False, promote=True)
        self.assertEqual(promoted["status"], "promoted")
        self.assertTrue(promoted["reused_candidate_without_training"])
        self.assertEqual(promoted["attempt_key"], proposal["attempt_key"])
        self.assertEqual(json.loads(self.belief_path.read_text())["model_id"], promoted["candidate_model_id"])

    def test_model_changed_during_fit_is_not_overwritten(self):
        self.append(self.new)
        other = BayesianWorldModel.fit(self.initial, seed=99, random_features=0, allow_synthetic=True)

        def concurrent_writer(*args, **kwargs):
            candidate, gate = propose_update(*args, **kwargs)
            self.assertTrue(gate["accepted"])
            other.save(self.model_path)
            return candidate, gate

        with patch("virtual_t_cell.learning.runner.propose_update", side_effect=concurrent_writer):
            result = self.cycle()
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejection_stage"], "promotion")
        self.assertIn("changed", result["promotion_error"])
        self.assertEqual(BayesianWorldModel.load(self.model_path).metadata["model_id"], other.metadata["model_id"])
        self.assertTrue(Path(result["candidate_model_path"]).is_file())
        self.assertTrue(Path(result["gate_report_path"]).is_file())

    def test_belief_write_failure_rolls_model_back_and_preserves_candidate(self):
        self.append(self.new)
        old_belief = self.belief_path.read_bytes()
        replace = os.replace
        failed = False

        def sidecar_failure(source, destination):
            nonlocal failed
            # macOS may expose a temporary path through /var while the runner
            # resolves the same file under /private/var.
            if Path(destination).resolve() == self.belief_path.resolve() and not failed:
                failed = True
                raise OSError("synthetic sidecar write failure")
            return replace(source, destination)

        with patch("virtual_t_cell.learning.runner.os.replace", side_effect=sidecar_failure):
            result = self.cycle()
        self.assertTrue(failed)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejection_stage"], "promotion")
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)
        self.assertEqual(self.belief_path.read_bytes(), old_belief)
        self.assertTrue(Path(result["candidate_model_path"]).is_file())
        # A later successful write retries promotion, never training.
        with patch.object(BayesianWorldModel, "fit", side_effect=AssertionError("candidate should be reused")):
            recovered = self.cycle(seed=False)
        self.assertEqual(recovered["status"], "promoted")

    def test_parallel_cycle_lock_prevents_a_second_writer(self):
        process, connection = self.lock_holder()
        lock_path = Path(str(self.model_path) + ".learning.lock")
        started = time.monotonic()
        with self.assertRaisesRegex(BlockingIOError, "another learning cycle"):
            self.cycle()
        self.assertLess(time.monotonic() - started, 3.0, "lock contention must not use a blocking retry mode")
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)
        connection.send("release")
        process.join(timeout=15)
        self.assertFalse(process.is_alive(), "lock holder did not exit after release")
        self.assertEqual(process.exitcode, 0)
        self.assertTrue(lock_path.is_file(), "persistent lock files must not be unlinked")
        with _model_lock(self.model_path):
            pass

    def test_model_lock_is_released_when_body_raises(self):
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with _model_lock(self.model_path):
                raise RuntimeError("fixture failure")
        with _model_lock(self.model_path):
            pass
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)

    def test_abrupt_process_exit_does_not_leave_a_stale_lock(self):
        process, connection = self.lock_holder()
        connection.send("abort")
        process.join(timeout=15)
        self.assertFalse(process.is_alive(), "aborted lock holder did not exit")
        self.assertEqual(process.exitcode, 23)
        # Some filesystems can finish releasing process locks just after process
        # termination is observed.  Bound the recovery check, never delete the
        # lock file to force acquisition.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                with _model_lock(self.model_path):
                    break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.fail("process exit left an unreleased model lock")
                time.sleep(0.02)
        self.assertEqual(self.model_path.read_bytes(), self.original_bytes)

    def test_training_record_fingerprints_prevent_rewritten_replay(self):
        altered = deepcopy(self.initial)
        altered[0]["outcomes"]["response"] += 1.0
        self.append(altered + self.new)
        with patch.object(BayesianWorldModel, "fit", side_effect=AssertionError("rewritten evidence must not train")):
            result = self.cycle(seed=False)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("replay_content_differs_from_incorporated_evidence", result["gate"]["reasons"])
        self.assertEqual(result["changed_replay_ids"], [self.initial[0]["observation_id"]])

    def test_synthetic_permission_and_experimental_mode_are_kept_separate(self):
        self.append(self.new)
        result = self.cycle(allow_synthetic=False)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("synthetic_model_requires_allow_synthetic", result["gate"]["reasons"])
        # These are software type-contract fixtures, not empirical observations.
        initial = deepcopy(self.initial)
        new = deepcopy(self.new)
        validation = deepcopy(self.validation)
        for row in initial + new + validation:
            row["evidence_type"] = "experimental_summary"
        real_mode_model = BayesianWorldModel.fit(initial, random_features=0)
        real_mode_path = self.root / "experimental_mode.npz"
        real_mode_model.save(real_mode_path)
        real_journal = self.root / "experimental_mode.sqlite"
        with EvidenceStore(real_journal) as journal:
            journal.append(initial + new + records("synthetic_extra", [0.2]))
        real_validation = self.root / "experimental_validation.jsonl"
        write_jsonl(real_validation, validation)
        report = run_learning_cycle(model_path=real_mode_path, evidence_path=real_journal,
                                    validation_path=real_validation, output_dir=self.root / "experimental_runs",
                                    allow_synthetic=True)
        self.assertEqual(report["status"], "promoted")
        self.assertEqual(report["new_evidence_count"], len(new))
        self.assertEqual(report["ignored_evidence"]["by_reason"],
                         {"synthetic_evidence_cannot_update_an_experimental_model": 2})
        self.assertEqual(BayesianWorldModel.load(real_mode_path).metadata["evidence_types"], ["experimental_summary"])


if __name__ == "__main__":
    unittest.main()
