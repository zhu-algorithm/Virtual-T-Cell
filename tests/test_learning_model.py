"""Numerical and provenance checks for the response learning core.

The fixtures below are explicitly synthetic software tests.  Their successful
prediction does not constitute performance on biological data.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from virtual_t_cell.learning.belief import BeliefState
from virtual_t_cell.learning.world_model import BayesianWorldModel


def record(index, x, *, organ="liver", value=None, second=None, sd=None,
           evidence_type="synthetic", cell_type="CD8_T", condition="reference"):
    outcomes = {"response": float(2.0 * x if value is None else value)}
    if second is not None:
        outcomes["second"] = float(second)
    row = {
        "observation_id": f"obs-{index}",
        "experiment_id": f"exp-{index}",
        "group_id": f"donor-{index}",
        "organ": organ,
        "cell_type": cell_type,
        "condition": condition,
        "intervention": "fixture-action",
        "evidence_type": evidence_type,
        "source": "synthetic-unittest-fixture-not-biological-data",
        "features": {"action_descriptor": float(x)},
        "outcomes": outcomes,
        "outcome_units": {key: "synthetic_unit" for key in outcomes},
    }
    if sd is not None:
        row["measurement_sd"] = {"response": float(sd)}
    return row


def fixture(start=0, count=40):
    rows = []
    for index in range(count):
        x = -1.0 + 2.0 * (index // 2) / max(1, count // 2 - 1)
        organ = "liver" if index % 2 == 0 else "spleen"
        value = 2.0 * x + (0.8 if organ == "spleen" else -0.3) + 0.35 * np.sin(2.0 * x)
        rows.append(record(start + index, x, organ=organ, value=value, sd=0.03))
    return rows


def query(row):
    row = deepcopy(row)
    row["evidence_type"] = "model_prediction"
    row["outcomes"] = {}
    row["outcome_units"] = {}
    row.pop("measurement_sd", None)
    return row


class BayesianWorldModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = fixture()
        cls.model = BayesianWorldModel.fit(cls.records, seed=11, random_features=16,
                                          allow_synthetic=True)

    def test_learns_response_on_held_out_records_and_both_organs(self):
        held_out = []
        for index, x in enumerate(np.linspace(-0.91, 0.91, 22)):
            organ = "liver" if index % 2 == 0 else "spleen"
            value = 2.0 * x + (0.8 if organ == "spleen" else -0.3) + 0.35 * np.sin(2.0 * x)
            held_out.append(record(100 + index, x, organ=organ, value=value))
        prediction = self.model.predict(held_out)
        actual = np.asarray([row["outcomes"]["response"] for row in held_out])
        mse = np.mean((prediction["mean"][:, 0] - actual) ** 2)
        baseline = np.mean((actual - self.model.metadata["outcome_means"]["response"]) ** 2)
        self.assertLess(mse, baseline * 0.02)
        self.assertTrue(prediction["in_domain"].all())
        self.assertEqual(self.model.metadata["training_coverage"]["organs"], {"liver": 20, "spleen": 20})
        self.assertEqual(len(self.model.metadata["training_record_hashes"]), 40)
        self.assertEqual(len(self.model.metadata["training_experiments"]), 40)

    def test_missing_outputs_are_masks_and_true_zero_is_observed(self):
        rows = [record(index, index / 11.0, value=0.0 if index == 0 else index / 11.0,
                       second=(3.0 * index / 11.0 if index % 2 else None))
                for index in range(12)]
        model = BayesianWorldModel.fit(rows, random_features=4, allow_synthetic=True)
        self.assertEqual(model.metadata["output_counts"], {"response": 12, "second": 6})
        self.assertEqual(model.metadata["outcome_means"]["response"], 0.5)
        predicted = model.predict([query(rows[0]), query(rows[-1])])
        self.assertEqual(predicted["mean"].shape, (2, 2))
        self.assertTrue(np.isfinite(predicted["mean"]).all())
        self.assertAlmostEqual(predicted["mean"][1, 1], 3.0, places=4)

    def test_known_heteroscedastic_noise_downweights_unreliable_outlier(self):
        rows = [record(index, 0.0, value=0.0, sd=0.01) for index in range(10)]
        rows.append(record(10, 0.0, value=50.0, sd=100.0))
        weighted = BayesianWorldModel.fit(rows, random_features=0, allow_synthetic=True)
        unweighted_rows = deepcopy(rows)
        for row in unweighted_rows:
            row.pop("measurement_sd")
        unweighted = BayesianWorldModel.fit(unweighted_rows, random_features=0, allow_synthetic=True)
        weighted_mean = weighted.predict([query(rows[0])])["mean"][0, 0]
        unweighted_mean = unweighted.predict([query(rows[0])])["mean"][0, 0]
        self.assertLess(abs(weighted_mean), 0.01)
        self.assertGreater(abs(unweighted_mean), 1.0)

    def test_zero_noise_and_constant_features_remain_finite(self):
        rows = [record(index, 0.0, value=2.0, sd=0.0) for index in range(5)]
        model = BayesianWorldModel.fit(rows, random_features=8, allow_synthetic=True)
        prediction = model.predict(rows)
        np.testing.assert_allclose(prediction["mean"], 2.0)
        self.assertTrue(np.isfinite(prediction["epistemic_variance"]).all())
        self.assertTrue((prediction["measurement_variance"] == 0.0).all())
        self.assertTrue((prediction["noise_variance"] > 0.0).all())
        self.assertEqual(model.metadata["noise_model"]["normalized_residual_variance_floor"], 1e-8)

    def test_query_measurement_noise_is_separate_from_model_uncertainty(self):
        clean = deepcopy(self.records[5])
        clean["measurement_sd"] = {"response": 0.0}
        noisy = deepcopy(clean)
        noisy["measurement_sd"] = {"response": 3.0}
        prediction = self.model.predict([clean, noisy])
        np.testing.assert_array_equal(prediction["mean"][0], prediction["mean"][1])
        np.testing.assert_array_equal(prediction["epistemic_variance"][0], prediction["epistemic_variance"][1])
        self.assertAlmostEqual(prediction["noise_variance"][1, 0] - prediction["noise_variance"][0, 0], 9.0)
        np.testing.assert_allclose(prediction["noise_variance"],
                                   prediction["residual_variance"] + prediction["measurement_variance"])

    def test_covariance_agrees_with_epistemic_diagonal_and_is_psd(self):
        queries = [query(row) for row in self.records[::5]]
        covariance = self.model.covariance(queries, "response")
        prediction = self.model.predict(queries)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(covariance), prediction["epistemic_variance"][:, 0],
                                   rtol=1e-8, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-10)
        with self.assertRaisesRegex(ValueError, "unknown outcome"):
            self.model.covariance(queries, "never_measured")

    def test_unknown_contexts_and_numeric_extrapolation_are_flagged(self):
        rows = [query(self.records[10]) for _ in range(5)]
        rows[0]["organ"] = "blood"
        rows[1]["cell_type"] = "unseen_T"
        rows[2]["condition"] = "unseen_condition"
        rows[3]["features"]["action_descriptor"] = 2.0
        prediction = self.model.predict(rows)
        self.assertEqual(prediction["in_domain"].tolist(), [False, False, False, False, True])
        for index, token in enumerate(("unknown_organ", "unknown_cell_type", "unknown_condition",
                                       "feature_outside_training_range")):
            self.assertTrue(any(token in reason for reason in prediction["domain_reasons"][index]))

    def test_unseen_context_combination_is_not_silent_interpolation(self):
        rows = [record(i, i / 3, organ="liver", condition="A") for i in range(4)]
        rows += [record(10 + i, i / 3, organ="spleen", condition="B") for i in range(4)]
        model = BayesianWorldModel.fit(rows, random_features=0, allow_synthetic=True)
        candidate = query(rows[0])
        candidate["condition"] = "B"
        prediction = model.predict([candidate])
        self.assertFalse(prediction["in_domain"][0])
        self.assertIn("unseen_organ_cell_type_condition_species_combination", prediction["domain_reasons"][0])

    def test_species_is_encoded_and_unspecified_is_not_assumed_human(self):
        rows = []
        for index in range(10):
            for species, shift in (("human", 0.0), ("mouse", 4.0)):
                row = record(f"{species}-{index}", index / 9.0, value=index / 9.0 + shift)
                row["species"] = species
                rows.append(row)
        model = BayesianWorldModel.fit(rows, random_features=0, allow_synthetic=True)
        human, mouse = query(rows[0]), query(rows[1])
        unknown = deepcopy(human)
        unknown["species"] = "rat"
        unspecified = deepcopy(human)
        unspecified.pop("species")
        prediction = model.predict([human, mouse, unknown, unspecified])
        self.assertAlmostEqual(prediction["mean"][1, 0] - prediction["mean"][0, 0], 4.0, places=4)
        self.assertEqual(prediction["in_domain"].tolist(), [True, True, False, False])
        self.assertIn("unknown_species:rat", prediction["domain_reasons"][2])
        self.assertIn("unknown_species:unspecified", prediction["domain_reasons"][3])
        self.assertEqual(model.metadata["training_coverage"]["species"], {"human": 10, "mouse": 10})

    def test_missing_organ_evidence_for_output_head_is_flagged(self):
        rows = [record(i, i / 3, organ="liver") for i in range(4)]
        rows += [record(10 + i, i / 3, organ="spleen", second=float(i)) for i in range(4)]
        model = BayesianWorldModel.fit(rows, random_features=0, allow_synthetic=True)
        prediction = model.predict([query(rows[0]), query(rows[-1])])
        self.assertEqual(prediction["in_domain"].tolist(), [False, True])
        self.assertIn("outcome_without_organ_evidence:second:liver", prediction["domain_reasons"][0])

    def test_strict_feature_names_units_and_nonfinite_inputs(self):
        extra = query(self.records[0])
        extra["features"]["untrained"] = 1.0
        with self.assertRaisesRegex(ValueError, "feature contract mismatch"):
            self.model.predict([extra])
        units = deepcopy(self.records[0])
        units["outcome_units"]["response"] = "different_unit"
        with self.assertRaisesRegex(ValueError, "inconsistent units"):
            self.model.predict([units])
        bad = query(self.records[0])
        bad["features"]["action_descriptor"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite number"):
            self.model.predict([bad])
        overflowing_noise = deepcopy(self.records[0])
        overflowing_noise["measurement_sd"]["response"] = 1e200
        with self.assertRaisesRegex(ValueError, "measurement variance overflow"):
            self.model.predict([overflowing_noise])
        with self.assertRaisesRegex(ValueError, "measurement variance overflow"):
            BayesianWorldModel.fit([overflowing_noise], allow_synthetic=True)
        self.assertEqual(self.model.predict([])["mean"].shape, (0, 1))

    def test_predictions_observations_and_synthetic_evidence_are_not_interchangeable(self):
        with self.assertRaisesRegex(ValueError, "synthetic cannot"):
            BayesianWorldModel.fit(self.records)
        for kind in ("observational", "model_prediction"):
            row = deepcopy(self.records[0])
            row["evidence_type"] = kind
            with self.assertRaisesRegex(ValueError, "cannot be used"):
                BayesianWorldModel.fit([row], allow_synthetic=True)
        mixed = deepcopy(self.records[:2])
        mixed[0]["evidence_type"] = "experimental"
        with self.assertRaisesRegex(ValueError, "separate datasets"):
            BayesianWorldModel.fit(mixed, allow_synthetic=True)

    def test_save_load_is_pickle_free_and_preserves_predictions_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posterior.model"
            self.model.save(path)
            self.assertTrue(path.is_file())
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype.kind != "O" for key in archive.files))
            loaded = BayesianWorldModel.load(path)
            self.assertEqual(loaded.metadata, self.model.metadata)
            before, after = self.model.predict(self.records), loaded.predict(self.records)
            for key in ("mean", "epistemic_variance", "noise_variance", "in_domain"):
                np.testing.assert_array_equal(before[key], after[key])
            np.testing.assert_array_equal(self.model.covariance(self.records, "response"),
                                          loaded.covariance(self.records, "response"))
            with np.load(path, allow_pickle=False) as archive:
                content = {key: archive[key].copy() for key in archive.files}
            content["posterior_mean"][0, 0] += 1.0
            corrupt = Path(directory) / "corrupt.npz"
            np.savez_compressed(corrupt, **content)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                BayesianWorldModel.load(corrupt)


class BeliefStateTests(unittest.TestCase):
    def setUp(self):
        self.records = fixture(count=12)
        self.model = BayesianWorldModel.fit(self.records, random_features=4, allow_synthetic=True)
        self.state = BeliefState.from_model(self.model)

    def test_pending_evidence_does_not_silently_train_and_duplicates_are_idempotent(self):
        new = record(100, 0.1, value=0.5, sd=0.03)
        before = self.state.predict([new])["mean"].copy()
        result = self.state.record_observation(new)
        self.assertEqual(result["status"], "recorded")
        self.assertFalse(result["posterior_updated"])
        summary = self.state.to_dict()
        self.assertEqual(summary["evidence_counts"]["incorporated"], {"synthetic": 12})
        self.assertEqual(summary["evidence_counts"]["pending"], {"synthetic": 1})
        self.assertEqual(summary["pending_observation_ids"], ["obs-100"])
        np.testing.assert_array_equal(self.state.predict([new])["mean"], before)
        self.assertAlmostEqual(summary["residuals"][0]["residual"], 0.5 - before[0, 0])
        self.assertIn("not established", summary["uncertainty_calibration"])
        json.dumps(summary, allow_nan=False)
        self.assertEqual(self.state.record_observation(new)["status"], "duplicate")
        self.assertEqual(self.state.to_dict()["evidence_counts"]["total_unique"], 13)
        changed = deepcopy(new)
        changed["outcomes"]["response"] += 1.0
        with self.assertRaisesRegex(ValueError, "changed evidence"):
            self.state.record_observation(changed)
        refitted = BayesianWorldModel.fit(self.records + [new], random_features=4, allow_synthetic=True)
        self.state.bind_model(refitted)
        self.assertEqual(self.state.to_dict()["evidence_counts"]["pending"], {})
        self.assertEqual(self.state.to_dict()["evidence_counts"]["incorporated"], {"synthetic": 13})
        self.assertEqual(self.state.to_dict()["model_id"], refitted.metadata["model_id"])

    def test_ledger_rejects_fabricated_prediction_evidence_and_mismatched_predictions(self):
        candidate = query(self.records[0])
        candidate["observation_id"] = "prediction-not-evidence"
        # A prediction with fabricated labels is still not experimental evidence.
        candidate["outcomes"] = {"response": 0.0}
        candidate["outcome_units"] = {"response": "synthetic_unit"}
        with self.assertRaisesRegex(ValueError, "cannot be recorded"):
            self.state.record_observation(candidate)
        new = record(100, 0.1, value=0.5)
        prediction = self.model.predict([new])
        prediction["mean"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.state.record_observation(new, prediction=prediction)
        self.assertEqual(self.state.to_dict()["evidence_counts"]["total_unique"], 12)


if __name__ == "__main__":
    unittest.main()
