import tempfile
import unittest
from pathlib import Path

from virtual_t_cell.immune_cycle import STAGES, run_immune_cycle, simulate_immune_cycle


class ImmuneCycleTests(unittest.TestCase):
    def test_complete_cycle_has_all_stage_fluxes(self):
        trajectory, flux = simulate_immune_cycle(days=42)
        self.assertEqual(set(flux.stage), set(STAGES))
        self.assertEqual(trajectory.cycle_completion.iloc[-1], 1)
        self.assertGreater(trajectory.tumor_infiltrating_effector.max(), 0)
        self.assertGreater(trajectory.daily_tumor_killing.sum(), 0)

    def test_recognition_requires_hla_and_tumor_mhc(self):
        matched, _ = simulate_immune_cycle(days=42, hla_match=.9, tumor_mhc_i=.9)
        no_hla, _ = simulate_immune_cycle(days=42, hla_match=0, tumor_mhc_i=.9)
        no_mhc, _ = simulate_immune_cycle(days=42, hla_match=.9, tumor_mhc_i=0)
        self.assertGreater(matched.daily_tumor_killing.sum(), 0)
        self.assertEqual(no_hla.daily_tumor_killing.sum(), 0)
        self.assertEqual(no_mhc.daily_tumor_killing.sum(), 0)

    def test_checkpoint_blockade_increases_killing(self):
        untreated, _ = simulate_immune_cycle(days=42, therapy="none")
        treated, _ = simulate_immune_cycle(days=42, therapy="anti_PD1")
        self.assertGreater(treated.daily_tumor_killing.sum(), untreated.daily_tumor_killing.sum())
        self.assertLess(treated.tumor.iloc[-1], untreated.tumor.iloc[-1])

    def test_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_immune_cycle(Path(tmp), days=7)
            self.assertFalse(result["clinical_use"])
            self.assertTrue((Path(tmp) / "immune_cycle_trajectory.csv").exists())


if __name__ == "__main__": unittest.main()
