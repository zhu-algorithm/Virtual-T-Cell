import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from virtual_t_cell.cli import parse_perturbations, pathway_scores, predict


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "gse92872_virtual_t_cell.npz"
PRIMARY_MODEL = ROOT / "models" / "gse314342_primary_cd4_virtual_t_cell.npz"
TCR_DATABASE = ROOT / "models" / "vdjdb_2026_06_tcr_evidence.npz"
CONTEXT_MODEL = ROOT / "models" / "gse278572_primary_context_virtual_t_cell.npz"


class VirtualTCellTests(unittest.TestCase):
    def test_primary_cd4_model_meets_scale_requirement(self):
        model = np.load(PRIMARY_MODEL, allow_pickle=False)
        self.assertGreaterEqual(len(model["targets"]), 200)
        self.assertGreaterEqual(len(model["genes"]), 200)
        self.assertEqual(model["source"].item(), "GSE314342")
        self.assertEqual(model["conditions"].tolist(), ["Rest", "Stim8hr", "Stim48hr"])

    def test_tcr_database_schema_and_scale(self):
        database = np.load(TCR_DATABASE, allow_pickle=False)
        self.assertEqual(database["source"].item(), "VDJdb")
        self.assertGreater(len(database["field_cdr3"]), 100000)
        self.assertIn("TRA", database["field_gene"])
        self.assertIn("TRB", database["field_gene"])

    def test_gse278572_is_primary_context_model(self):
        model = np.load(CONTEXT_MODEL, allow_pickle=False)
        self.assertEqual(model["source"].item(), "GSE278572_primary")
        self.assertGreaterEqual(len(model["targets"]), 500)
        self.assertEqual(model["conditions"].tolist(),
                         ["Teff_Resting", "Teff_Stimulated", "Treg_Resting", "Treg_Stimulated"])
        self.assertIn("MED12", model["targets"])
        self.assertGreater(len(model["screen_targets"]), 19000)

    def test_bundled_model_schema(self):
        model = np.load(MODEL, allow_pickle=False)
        self.assertIn("stimulated", model["conditions"].tolist())
        self.assertIn("unstimulated", model["conditions"].tolist())
        self.assertIn("LCK", model["targets"].tolist())
        self.assertEqual(model["baseline"].shape[-1], len(model["genes"]))
        self.assertEqual(model["effects"].shape[-1], len(model["genes"]))

    def test_end_to_end_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            predict(MODEL, "stimulated", [("LCK", 1.0)], out)
            genes = pd.read_csv(out / "gene_predictions.csv")
            pathways = pd.read_csv(out / "pathway_predictions.csv")
            metadata = json.loads((out / "prediction_metadata.json").read_text())
            self.assertGreater(len(genes), 1000)
            self.assertEqual(len(pathways), 10)
            self.assertEqual(metadata["perturbations"][0]["mode"], "observed")

    def test_multi_target_parser(self):
        self.assertEqual(parse_perturbations(["LCK:0.8", "PTPN11:0.3"]), [("LCK", 0.8), ("PTPN11", 0.3)])

    def test_pathway_scores(self):
        result = pathway_scores(np.array(["LCK", "ZAP70", "FOS"]), np.array([-1.0, -0.5, 0.2]))
        self.assertIn("TCR_SIGNALING", result["pathway"].tolist())


if __name__ == "__main__":
    unittest.main()
