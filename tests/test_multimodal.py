import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from virtual_t_cell.multimodal import (
    generate_simulated_example, predict_multimodal, train_multimodal,
    validate_dataset,
)


class MultimodalVirtualTCellTests(unittest.TestCase):
    def test_schema_train_and_counterfactual_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            model = root / "model.npz"
            summary = generate_simulated_example(dataset, n_cells=48, seed=3)
            self.assertEqual(summary["data_origin"], "simulated")
            self.assertEqual(set(summary["modalities"]), {"rna", "atac", "protein", "tcr"})
            report = train_multimodal(dataset, model, latent_dim=3, ridge=2.0)
            self.assertEqual(report["model_type"], "masked_multiview_svd_ridge_v1")
            output = root / "prediction"
            metadata = predict_multimodal(model, dataset, "sim_cell_0000", "LCK", 0.8, 24, output)
            self.assertEqual(metadata["counterfactual"]["perturbation"], "LCK")
            for modality in ("rna", "atac", "protein", "tcr"):
                frame = pd.read_csv(output / f"{modality}_predictions.csv")
                self.assertGreater(len(frame), 0)
                self.assertTrue(np.isfinite(frame["predicted_value"]).all())
            saved = json.loads((output / "prediction_metadata.json").read_text())
            self.assertEqual(saved["data_origin"], "simulated")

    def test_validator_rejects_unlabelled_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            generate_simulated_example(dataset, n_cells=24)
            manifest = json.loads((dataset / "manifest.json").read_text())
            manifest.pop("data_origin")
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "data_origin"):
                validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
