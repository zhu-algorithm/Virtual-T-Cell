"""Scientific failure-mode tests for planning and gated continual learning."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from virtual_t_cell.learning.continual import evaluate_model, propose_update
from virtual_t_cell.learning.planner import plan_experiment
from virtual_t_cell.learning.schema import content_hash
from virtual_t_cell.learning.world_model import BayesianWorldModel


def record(identity, x, *, organ="liver", outcome=None, group=None):
    return {
        "observation_id": identity, "experiment_id": f"experiment:{identity}",
        "group_id": group or f"group:{identity}", "organ": organ, "cell_type": "CD8_T",
        "condition": "synthetic_fixture", "intervention": "abstract_input",
        "source": "synthetic_unit_test", "evidence_type": "synthetic",
        "features": {"x": float(x)}, "outcomes": {"response": float(2 * x if outcome is None else outcome)},
        "outcome_units": {"response": "synthetic_units"}, "measurement_sd": {"response": 0.05},
    }


def candidate(index, *, cost=1.0, organ="liver"):
    result = record(f"candidate:{index}", index, organ=organ)
    result.update(candidate_id=f"candidate:{index}", cost=cost, outcomes={}, outcome_units={},
                  measurement_sd={}, evidence_type="model_prediction")
    return result


class GaussianFixture:
    def __init__(self, means, covariance, noise=None):
        self.means = np.asarray(means, dtype=float)
        self.cov = np.asarray(covariance, dtype=float)
        self.noise = np.zeros(len(means)) if noise is None else np.asarray(noise, dtype=float)
        self.metadata = {"outcome_units": {"response": "synthetic_units"}, "model_id": "gaussian_fixture"}

    def predict(self, records):
        indices = [int(row["features"]["x"]) for row in records]
        domain = np.asarray([row["organ"] != "unsupported" for row in records])
        return {"outcome_names": ["response"], "mean": self.means[indices, None],
                "epistemic_variance": np.diag(self.cov)[indices, None],
                "noise_variance": self.noise[indices, None], "in_domain": domain,
                "domain_reasons": [[] if supported else ["unknown_organ"] for supported in domain]}

    def covariance(self, records, outcome):
        indices = [int(row["features"]["x"]) for row in records]
        return self.cov[np.ix_(indices, indices)]


class ResponseFixture:
    def __init__(self, training, organ_bias):
        self.organ_bias = dict(organ_bias)
        self.metadata = {
            "training_ids": [row["observation_id"] for row in training],
            "training_groups": [row["group_id"] for row in training],
            "training_experiments": [row["experiment_id"] for row in training],
            "training_record_hashes": {row["observation_id"]: content_hash(row) for row in training},
            "outcome_units": {"response": "synthetic_units"},
            "outcome_means": {"response": float(np.mean([row["outcomes"]["response"] for row in training]))},
            "evidence_types": ["synthetic"], "model_id": content_hash([training, organ_bias]),
            "fit_config": {"seed": 3, "random_features": 0, "prior_precision": 1.0},
        }

    def predict(self, records):
        means = [[2 * row["features"]["x"] + self.organ_bias.get(row["organ"], 0.0)] for row in records]
        return {"outcome_names": ["response"], "mean": np.asarray(means),
                "epistemic_variance": np.full((len(records), 1), 0.1),
                "noise_variance": np.full((len(records), 1), 0.2),
                "in_domain": np.ones(len(records), dtype=bool), "domain_reasons": [[] for _ in records]}


class TwoOutputFixture(ResponseFixture):
    def __init__(self, training, biases):
        super().__init__(training, {organ: value[0] for organ, value in biases.items()})
        self.biases = biases
        self.metadata["outcome_units"]["second"] = "synthetic_units"
        self.metadata["outcome_means"]["second"] = self.metadata["outcome_means"]["response"]

    def predict(self, records):
        result = super().predict(records)
        result["outcome_names"] = ["response", "second"]
        result["mean"] = np.asarray([[2 * row["features"]["x"] + bias
                                      for bias in self.biases[row["organ"]]] for row in records])
        result["epistemic_variance"] = np.full((len(records), 2), 0.1)
        result["noise_variance"] = np.full((len(records), 2), 0.2)
        return result


class KnowledgeGradientTests(unittest.TestCase):
    def test_exact_independent_knowledge_gradient(self):
        result = plan_experiment(GaussianFixture([0, 0], np.eye(2)), [candidate(0), candidate(1)],
                                 outcome="response", budget=1.0)
        self.assertEqual(result["decision"], "experiment")
        for rank in result["rankings"]:
            self.assertAlmostEqual(rank["knowledge_gradient"], 1.0 / math.sqrt(2.0 * math.pi), places=12)
        json.dumps(result, allow_nan=False)

    def test_correlated_common_shift_has_zero_decision_value(self):
        result = plan_experiment(GaussianFixture([0, 1], np.ones((2, 2))), [candidate(0), candidate(1)],
                                 outcome="response", budget=2)
        self.assertEqual(result["decision"], "stop")
        self.assertTrue(all(row["knowledge_gradient"] < 1e-12 for row in result["rankings"]))

    def test_negative_correlation_and_minimization(self):
        model = GaussianFixture([0, 0], [[1, -1], [-1, 1]])
        for direction in ("maximize", "minimize"):
            result = plan_experiment(model, [candidate(0), candidate(1)], outcome="response", budget=2,
                                     direction=direction)
            self.assertAlmostEqual(result["expected_decision_gain"], math.sqrt(2 / math.pi), places=12)

    def test_cost_noise_and_budget_have_distinct_roles(self):
        model = GaussianFixture([0, 0], np.eye(2), noise=[0, 3])
        choices = [candidate(0, cost=4), candidate(1, cost=1)]
        result = plan_experiment(model, choices, outcome="response", budget=4)
        self.assertEqual(result["selected_candidate"]["candidate_id"], "candidate:1")
        ranking = {row["candidate_id"]: row for row in result["rankings"]}
        self.assertAlmostEqual(ranking["candidate:0"]["knowledge_gradient"],
                               2 * ranking["candidate:1"]["knowledge_gradient"])
        # Only one affordable terminal alternative remains; information cannot change a choice.
        constrained = plan_experiment(model, choices, outcome="response", budget=1)
        self.assertEqual(constrained["decision"], "stop")
        self.assertIn("over_budget", constrained["rankings"][-1]["exclusion_reasons"])

    def test_out_of_domain_rejected_unless_explicitly_allowed(self):
        model = GaussianFixture([0, 0], np.eye(2))
        choices = [candidate(0), candidate(1, organ="unsupported")]
        result = plan_experiment(model, choices, outcome="response", budget=2)
        self.assertEqual(result["decision"], "stop")
        self.assertIn("out_of_domain", result["rankings"][-1]["exclusion_reasons"])
        allowed = plan_experiment(model, choices, outcome="response", budget=2, allow_extrapolation=True)
        self.assertEqual(allowed["decision"], "experiment")
        self.assertTrue(any(not row["in_domain"] for row in allowed["rankings"]))

    def test_invalid_inputs_and_zero_uncertainty(self):
        model = GaussianFixture([0, 0], np.zeros((2, 2)))
        result = plan_experiment(model, [candidate(0), candidate(1)], outcome="response", budget=2,
                                 min_information_value=0)
        self.assertEqual(result["decision"], "stop")
        for cost in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                plan_experiment(model, [candidate(0, cost=cost)], outcome="response", budget=2)
        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            plan_experiment(model, [candidate(0), candidate(0)], outcome="response", budget=2)
        with self.assertRaisesRegex(ValueError, "no trained output"):
            plan_experiment(model, [candidate(0)], outcome="missing", budget=2)
        with self.assertRaisesRegex(ValueError, "model_prediction"):
            wrong = candidate(0)
            wrong["evidence_type"] = "experimental"
            plan_experiment(model, [wrong], outcome="response", budget=2)


class ContinualLearningTests(unittest.TestCase):
    def setUp(self):
        self.replay = [record(f"old:{i}", i - 2, organ="liver" if i % 2 == 0 else "spleen") for i in range(6)]
        self.new = [record(f"new:{i}", i - 1.5, organ="liver" if i % 2 == 0 else "spleen") for i in range(4)]
        self.validation = [record(f"validation:{i}", i - 2.5, organ="liver" if i % 2 == 0 else "spleen") for i in range(6)]
        self.current = ResponseFixture(self.replay, {"liver": 1, "spleen": 2})

    def test_evaluation_uses_training_mean_and_discloses_groups(self):
        metrics = evaluate_model(self.current, self.validation)
        self.assertEqual(metrics["per_output"]["response"]["training_mean_baseline"],
                         self.current.metadata["outcome_means"]["response"])
        self.assertAlmostEqual(metrics["per_organ"]["liver"]["rmse"], 1)
        self.assertAlmostEqual(metrics["per_organ"]["spleen"]["rmse"], 2)
        self.assertEqual(metrics["overall"]["n_groups"], 6)
        self.assertTrue(metrics["independent_validation"])
        json.dumps(metrics, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_model(self.current, self.replay)

    def test_group_balancing_does_not_treat_many_cells_as_many_groups(self):
        records = [record(f"many:{i}", 0, outcome=0, group="many") for i in range(9)]
        records.append(record("one", 0, outcome=10, group="one"))
        model = ResponseFixture(self.replay, {"liver": 0})
        output = evaluate_model(model, records)["per_output"]["response"]
        self.assertAlmostEqual(output["rmse"], math.sqrt(10))
        self.assertAlmostEqual(output["group_balanced_rmse"], math.sqrt(50))

    def test_improving_proposal_is_returned_without_mutating_current(self):
        candidate_model = ResponseFixture(self.replay + self.new, {"liver": 0.5, "spleen": 1})
        identity = copy.deepcopy(self.current.metadata)
        with patch.object(BayesianWorldModel, "fit", return_value=candidate_model):
            proposed, report = propose_update(self.current, self.replay, self.new, self.validation,
                                               allow_synthetic=True)
        self.assertIs(proposed, candidate_model)
        self.assertTrue(report["accepted"])
        self.assertFalse(report["promoted"])
        self.assertEqual(self.current.metadata, identity)
        json.dumps(report, allow_nan=False)

    def test_organ_regression_blocks_better_overall_score(self):
        current = ResponseFixture(self.replay, {"liver": 10, "spleen": 1})
        candidate_model = ResponseFixture(self.replay + self.new, {"liver": 5, "spleen": 2})
        with patch.object(BayesianWorldModel, "fit", return_value=candidate_model):
            _, report = propose_update(current, self.replay, self.new, self.validation, allow_synthetic=True)
        self.assertTrue(report["regression_checks"]["overall"]["passed"])
        self.assertFalse(report["regression_checks"]["per_organ"]["spleen"]["passed"])
        self.assertFalse(report["accepted"])

    def test_missing_tissue_validation_blocks_promotion(self):
        candidate_model = ResponseFixture(self.replay + self.new, {"liver": 0, "spleen": 0})
        validation = [row for row in self.validation if row["organ"] == "liver"]
        with patch.object(BayesianWorldModel, "fit", return_value=candidate_model):
            _, report = propose_update(self.current, self.replay, self.new, validation, allow_synthetic=True)
        self.assertFalse(report["accepted"])
        self.assertIn("missing_independent_validation_organ:spleen", report["reasons"])

    def test_tissue_output_regression_cannot_hide_in_better_averages(self):
        replay, new, validation = copy.deepcopy((self.replay, self.new, self.validation))
        for row in replay + new + validation:
            row["outcomes"]["second"] = row["outcomes"]["response"]
            row["outcome_units"]["second"] = "synthetic_units"
        current = TwoOutputFixture(replay, {"liver": (10, 1), "spleen": (1, 10)})
        candidate_model = TwoOutputFixture(replay + new, {"liver": (5, 2), "spleen": (2, 5)})
        with patch.object(BayesianWorldModel, "fit", return_value=candidate_model):
            _, report = propose_update(current, replay, new, validation, allow_synthetic=True)
        self.assertTrue(report["regression_checks"]["overall"]["passed"])
        self.assertTrue(all(check["passed"] for check in report["regression_checks"]["per_output"].values()))
        self.assertFalse(report["accepted"])
        self.assertIn("organ_output_gate_failed:liver:second", report["reasons"])
        self.assertIn("organ_output_gate_failed:spleen:response", report["reasons"])

    def test_new_tissue_must_beat_mean_baseline_not_only_old_extrapolation(self):
        replay = [record(f"liver:{i}", x) for i, x in enumerate((-1, 1))]
        new = [record(f"spleen:{i}", x, organ="spleen") for i, x in enumerate((-1, 1))]
        validation = [record(f"v:{organ}:{i}", x, organ=organ)
                      for organ in ("liver", "spleen") for i, x in enumerate((-0.5, 0.5))]
        current = ResponseFixture(replay, {"liver": 0, "spleen": 100})
        candidate_model = ResponseFixture(replay + new, {"liver": 0, "spleen": 10})
        with patch.object(BayesianWorldModel, "fit", return_value=candidate_model):
            _, report = propose_update(current, replay, new, validation, allow_synthetic=True)
        self.assertTrue(report["regression_checks"]["overall"]["passed"])
        self.assertFalse(report["accepted"])
        self.assertIn("organ_output_gate_failed:spleen:response", report["reasons"])

    def test_species_need_independent_in_domain_validation(self):
        replay, new, validation = copy.deepcopy((self.replay, self.new, self.validation))
        for row in replay + validation:
            row["species"] = "human"
        for row in new:
            row["species"] = "mouse"
        current = ResponseFixture(replay, {"liver": 1, "spleen": 2})
        candidate_model = ResponseFixture(replay + new, {"liver": 0, "spleen": 0})
        with patch.object(BayesianWorldModel, "fit", return_value=candidate_model):
            _, report = propose_update(current, replay, new, validation, allow_synthetic=True)
        self.assertFalse(report["accepted"])
        self.assertIn("missing_independent_validation_species:mouse", report["reasons"])
        original_predict = candidate_model.predict

        def extrapolate(rows):
            prediction = original_predict(rows)
            prediction["in_domain"][:] = False
            prediction["domain_reasons"] = [["out_of_support"] for _ in rows]
            return prediction

        with patch.object(BayesianWorldModel, "fit", return_value=candidate_model), \
                patch.object(candidate_model, "predict", side_effect=extrapolate):
            _, report = propose_update(current, replay, new, validation, allow_synthetic=True)
        self.assertIn("no_in_domain_validation_species:human", report["reasons"])

    def test_leakage_duplicates_rewritten_replay_and_units_rejected(self):
        cases = []
        val = copy.deepcopy(self.validation)
        val[0]["group_id"] = self.replay[0]["group_id"]
        cases.append((self.replay, self.new, val, "overlap"))
        val = copy.deepcopy(self.validation)
        val[0]["experiment_id"] = self.replay[0]["experiment_id"]
        cases.append((self.replay, self.new, val, "overlap|source evidence"))
        cases.append((self.replay[:-1], self.new, self.validation, "replay must cover"))
        altered = copy.deepcopy(self.replay)
        altered[0]["outcomes"]["response"] += 10
        cases.append((altered, self.new, self.validation, "changed previously"))
        repeated = copy.deepcopy(self.replay[0])
        repeated["observation_id"], repeated["group_id"] = "renamed_observation", "renamed_group"
        cases.append((self.replay, self.new + [repeated], self.validation, "repeated evidence|source evidence"))
        val = copy.deepcopy(self.validation)
        for row in val:
            row["outcome_units"]["response"] = "different_units"
        cases.append((self.replay, self.new, val, "units"))
        for replay, new, validation, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                propose_update(self.current, replay, new, validation, allow_synthetic=True)

    def test_prediction_and_observational_labels_cannot_train_responses(self):
        for kind in ("model_prediction", "observational", "experimental"):
            changed = copy.deepcopy(self.new)
            changed[0]["evidence_type"] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                propose_update(self.current, self.replay, changed, self.validation, allow_synthetic=True)

    def test_original_source_evidence_cannot_be_renamed_into_new_evidence(self):
        replay = copy.deepcopy(self.replay)
        for row in replay:
            row["source_record_id"] = row["observation_id"]
        current = ResponseFixture(replay, {"liver": 1, "spleen": 2})
        duplicate = copy.deepcopy(replay[0])
        duplicate.update(observation_id="renamed", group_id="regrouped", experiment_id="new_experiment_name")
        with self.assertRaisesRegex(ValueError, "source evidence"):
            propose_update(current, replay, self.new + [duplicate], self.validation, allow_synthetic=True)

    def test_standalone_evaluation_rejects_renamed_training_source_record(self):
        training = copy.deepcopy(self.replay)
        for row in training:
            row["source_record_id"] = row["observation_id"]
        model = BayesianWorldModel.fit(training, random_features=0, allow_synthetic=True)
        self.assertIn("training_source_identities", model.metadata)
        renamed = copy.deepcopy(training[0])
        renamed.update(observation_id="pretend_holdout", group_id="pretend_holdout_group",
                       experiment_id="pretend_holdout_experiment")
        with self.assertRaisesRegex(ValueError, "source identity overlap"):
            evaluate_model(model, [renamed])

    def test_real_fit_update_preserves_old_evidence_and_independent_holdout(self):
        replay = [record(f"train:{i}", float(x)) for i, x in enumerate(np.linspace(-2, 2, 14))]
        new = [record(f"fresh:{i}", float(x)) for i, x in enumerate(np.linspace(-1.8, 1.8, 8))]
        validation = [record(f"holdout:{i}", float(x)) for i, x in enumerate(np.linspace(-1.7, 1.7, 6))]
        model = BayesianWorldModel.fit(replay, random_features=0, allow_synthetic=True)
        old_identity = model.metadata["model_id"]
        updated, gate = propose_update(model, replay, new, validation, allow_synthetic=True)
        self.assertEqual(model.metadata["model_id"], old_identity)
        self.assertEqual(set(updated.metadata["training_ids"]),
                         {row["observation_id"] for row in replay + new})
        self.assertTrue(gate["accepted"])
        self.assertLess(gate["candidate_metrics"]["overall"]["rmse"], 0.1)


if __name__ == "__main__":
    unittest.main()
