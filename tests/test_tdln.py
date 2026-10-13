import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from virtual_t_cell.tdln.model import CELL_TYPES, default_state, run_simulation, simulate
from virtual_t_cell.tdln.download import FILES, _url
from virtual_t_cell.tdln.spatial import COMMUNICATION, ZONES


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "models" / "tdln_public_calibration_v1.json"


class TDLNTests(unittest.TestCase):
    def test_download_manifest_uses_geo_and_pinned_checksums(self):
        self.assertEqual(len(FILES), 6)
        self.assertTrue(all(len(checksum) == 64 for checksum in FILES.values()))
        self.assertTrue(all(_url(name).startswith("https://ftp.ncbi.nlm.nih.gov/geo/") for name in FILES))

    def test_bundled_public_calibration_schema(self):
        payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "tdln-public-calibration-v1")
        self.assertEqual(payload["cohort"]["lymph_node_cells"], 32509)
        self.assertEqual(payload["protein"]["matched_cells"], 32509)
        self.assertEqual(set(payload["baseline_abundance"]), set(CELL_TYPES))
        self.assertIn("leave-one-patient-out", payload["validation"]["method"])

    def test_calibration_initializes_organ(self):
        state = default_state(CALIBRATION)
        self.assertAlmostEqual(state.abundance.sum(), 1.0)
        self.assertGreater(state.loc[state.cell_type.eq("CD4_naive"), "abundance"].item(), .4)

    def test_mass_conservation_and_outputs(self):
        cells, programs = simulate(days=2, therapy="anti_PD1", calibration_model=CALIBRATION)
        self.assertTrue(((cells.groupby("day").abundance.sum() - 1).abs() < 1e-6).all())
        self.assertEqual(cells.zone.nunique(), len(ZONES))
        with tempfile.TemporaryDirectory() as tmp:
            metadata = run_simulation(Path(tmp), days=2, tumor_burden=.5, therapy="none",
                                      metastasis=True, seed=1, calibration_model=CALIBRATION)
            self.assertEqual(metadata["model"], "multiscale_spatial_communication_v3")
            communication = pd.read_csv(Path(tmp) / "cell_communication_trajectory.csv")
            self.assertEqual(len(communication), 3 * len(ZONES) * len(COMMUNICATION))


if __name__ == "__main__":
    unittest.main()
