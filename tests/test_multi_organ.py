import tempfile
import unittest
from pathlib import Path

from virtual_t_cell.multi_organ import ORGANS, SITE_TO_DRAINING_LN, run_multi_organ, simulate_multi_organ


class MultiOrganTests(unittest.TestCase):
    def test_all_organs_and_nonnegative_states(self):
        organs, migration, readouts = simulate_multi_organ(days=20)
        self.assertEqual(set(organs.organ), set(ORGANS))
        self.assertTrue((organs.abundance >= 0).all())
        self.assertGreater(len(migration), 0)
        self.assertTrue((readouts.tumor_burden.between(0, 1)).all())

    def test_anatomical_drainage(self):
        for site, expected_node in SITE_TO_DRAINING_LN.items():
            organs, _, readouts = simulate_multi_organ(days=8, tumor_site=site)
            self.assertEqual(readouts.draining_ln.iloc[0], expected_node)
            final = organs[organs.day.eq(8)].groupby("organ").antigen_load.first()
            other_nodes = [x for x in SITE_TO_DRAINING_LN.values() if x != expected_node]
            self.assertGreater(final[expected_node], max(final[x] for x in other_nodes))

    def test_recognition_and_combination_response(self):
        _, _, no_hla = simulate_multi_organ(days=40, hla_match=0)
        _, _, untreated = simulate_multi_organ(days=40, therapy="none")
        _, _, combination = simulate_multi_organ(days=40, therapy="combination")
        self.assertEqual(no_hla.daily_tumor_killing.sum(), 0)
        self.assertGreater(combination.daily_tumor_killing.sum(), untreated.daily_tumor_killing.sum())
        self.assertLess(combination.tumor_burden.iloc[-1], untreated.tumor_burden.iloc[-1])

    def test_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_multi_organ(Path(tmp), days=3, tumor_site="lung")
            self.assertEqual(result["draining_ln"], "mediastinal_ln")
            self.assertTrue((Path(tmp) / "multi_organ_cell_trajectory.csv").exists())


if __name__ == "__main__": unittest.main()
