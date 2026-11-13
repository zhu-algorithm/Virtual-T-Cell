"""Data provenance, tissue separation and missing-outcome regression tests."""
import csv
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from virtual_t_cell.learning.adapters import import_reference_npz
from virtual_t_cell.learning.atlas import AtlasStore
from virtual_t_cell.learning.schema import validate_dataset


class AtlasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.atlas = AtlasStore(self.root / "atlas.sqlite")

    def tearDown(self):
        self.atlas.close()
        self.temp.cleanup()

    def write_batch(self, name="one", rows=None, metadata=None, genes=None, extra_columns=None):
        genes = genes or ["CD3D", "IFNG", "ALB", "LYZ", "MT-CO1"]
        rows = rows if rows is not None else [
            ["c1", 3, 1, 0, 0, 0], ["c2", 2, 2, 0, 0, 0],
            ["c3", 5, 1, 0, 0, 0], ["c4", 0, 0, 0, 10, 1],
            ["c5", 0, 0, 0, 0, 0]]
        metadata = metadata if metadata is not None else [
            ["c1", "liver-sample", "donor-a", "liver", "CD8 T cell", "rest", 0],
            ["c2", "liver-sample", "donor-a", "liver", "CD8 T cell", "rest", 0],
            ["c3", "spleen-sample", "donor-a", "spleen", "CD8 T cell", "rest", 0],
            ["c4", "liver-sample", "donor-a", "liver", "macrophage", "rest", 0],
            ["c5", "liver-sample", "donor-a", "liver", "CD8 T cell", "rest", 0]]
        counts, meta = self.root / (name + ".counts.csv"), self.root / (name + ".metadata.csv")
        with counts.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_id", *genes])
            writer.writerows(rows)
        with meta.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_id", "sample_id", "donor_id", "organ", "cell_type", "condition", "biological_time_hours", *(extra_columns or [])])
            writer.writerows(metadata)
        return counts, meta

    def test_liver_spleen_context_and_pseudobulk_units(self):
        paths = self.write_batch()
        report = self.atlas.ingest_counts(*paths, batch_id="b1", source="fixture", evidence_type="synthetic")
        self.assertEqual(report["added_cells"], 5)
        self.assertEqual(report["qc_failed"], 1)
        self.assertFalse(report["response_model_updated"])
        states = self.atlas.pseudobulk_records()
        self.assertEqual(len(states), 3)
        liver_t = next(state for state in states if state["organ"] == "liver" and state["cell_type"] == "CD8 T cell")
        self.assertEqual(liver_t["raw_counts"]["CD3D"], 5)
        self.assertEqual(liver_t["n_cells"], 2)
        self.assertEqual(liver_t["n_donors"], 1)
        self.assertAlmostEqual(liver_t["outcomes"]["CD3D"], math.log1p(5 / 8 * 1e4))
        self.assertEqual(liver_t["outcome_units"]["CD3D"], "log1p_cp10k")
        self.assertEqual(liver_t["normalization"]["denominator_counts"], 8)
        self.assertIn("supplied gene panel", liver_t["normalization"]["denominator"])
        self.assertFalse(liver_t["normalization"]["whole_transcriptome_library_size_inferred"])
        self.assertFalse(liver_t["response_training_eligible"])
        self.assertTrue(any(state["cell_type"] == "macrophage" for state in states))
        summary = self.atlas.summary()
        self.assertEqual(summary["organs"], {"liver": 4, "spleen": 1})
        self.assertEqual(summary["evidence_counts"], {"synthetic": 5})

    def test_repeated_batches_idempotent_and_restart_persistent(self):
        paths = self.write_batch()
        self.atlas.ingest_counts(*paths, batch_id="b1", source="fixture")
        before = self.atlas.summary()["state_version"]
        repeated = self.atlas.ingest_counts(*paths, batch_id="b1", source="fixture")
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(repeated["added_cells"], 0)
        repeated_other_batch = self.atlas.ingest_counts(*paths, batch_id="b2", source="fixture")
        self.assertEqual(repeated_other_batch["duplicate_cells"], 5)
        self.assertEqual(repeated_other_batch["added_cells"], 0)
        self.assertEqual(self.atlas.summary()["state_version"], before)
        with AtlasStore(self.root / "atlas.sqlite") as reopened:
            self.assertEqual(reopened.summary()["n_cells"], 5)
            self.assertEqual(reopened.summary()["n_batches"], 2)
            self.assertTrue(all(state["batch_ids"] == ["b1", "b2"] for state in reopened.pseudobulk_records()))

    def test_persistent_store_identity_changes_for_rebuilt_database(self):
        original = self.atlas.store_identity()
        with AtlasStore(self.atlas.path) as reopened:
            self.assertEqual(original, reopened.store_identity())
        with AtlasStore(self.root / "different.sqlite") as different:
            self.assertNotEqual(original, different.store_identity())
        self.atlas.close()
        self.atlas.path.unlink()
        self.atlas = AtlasStore(self.root / "atlas.sqlite")
        self.assertNotEqual(original, self.atlas.store_identity())

    def test_interventions_and_doses_remain_separate_state_contexts(self):
        paths = self.write_batch("treatments", genes=["CD3D"],
                                 rows=[["c1", 10], ["c2", 20], ["c3", 30], ["c4", 40]],
                                 extra_columns=["intervention", "dose", "dose_unit"],
                                 metadata=[
                                     ["c1", "sample", "donor", "liver", "CD8 T cell", "treated", 0, "A", 1, "nM"],
                                     ["c2", "sample", "donor", "liver", "CD8 T cell", "treated", 0, "A", 2, "nM"],
                                     ["c3", "sample", "donor", "liver", "CD8 T cell", "treated", 0, "B", 1, "nM"],
                                     ["c4", "sample", "donor", "liver", "CD8 T cell", "treated", 0, "A", 1, "nM"]])
        self.atlas.ingest_counts(*paths, batch_id="b1", source="fixture")
        states = self.atlas.pseudobulk_records()
        self.assertEqual(len(states), 3)
        self.assertEqual({(r["intervention"], r["dose"], r["dose_unit"]) for r in states},
                         {("A", 1.0, "nM"), ("A", 2.0, "nM"), ("B", 1.0, "nM")})
        group = next(r for r in states if r["intervention"] == "A" and r["dose"] == 1.0)
        self.assertEqual(group["n_cells"], 2)
        self.assertEqual(group["raw_counts"]["CD3D"], 50)
        self.assertFalse(group["response_training_eligible"])

    def test_metadata_dose_requires_numeric_nonnegative_value_and_unit(self):
        for dose, unit in ((-1, "nM"), ("unknown-dose", "nM"), (1, "")):
            with self.subTest(dose=dose, unit=unit):
                paths = self.write_batch("bad-dose", genes=["CD3D"], rows=[["c1", 10]],
                                         extra_columns=["dose", "dose_unit"],
                                         metadata=[["c1", "s", "d", "liver", "CD8 T cell", "treated", 0, dose, unit]])
                with self.assertRaisesRegex(ValueError, "dose"):
                    self.atlas.ingest_counts(*paths, batch_id="bad", source="fixture")
        self.assertEqual(self.atlas.summary()["n_cells"], 0)

    def test_changed_cell_conflict_rolls_back_new_cells(self):
        self.atlas.ingest_counts(*self.write_batch(), batch_id="b1", source="fixture")
        counts, meta = self.write_batch("conflict", rows=[["new", 1, 0, 0, 0, 0], ["c1", 99, 0, 0, 0, 0]], metadata=[
            ["new", "liver-sample", "donor-a", "liver", "CD8 T cell", "rest", 0],
            ["c1", "liver-sample", "donor-a", "liver", "CD8 T cell", "rest", 0]])
        with self.assertRaisesRegex(ValueError, "existing cell identity"):
            self.atlas.ingest_counts(counts, meta, batch_id="bad", source="fixture")
        self.assertEqual(self.atlas.summary()["n_cells"], 5)
        self.assertEqual(self.atlas.summary()["n_batches"], 1)
        with self.assertRaisesRegex(ValueError, "batch_id already exists"):
            self.atlas.ingest_counts(counts, meta, batch_id="b1", source="fixture")

    def test_barcodes_may_repeat_across_samples_not_within_batch(self):
        self.atlas.ingest_counts(*self.write_batch(), batch_id="b1", source="fixture")
        paths = self.write_batch("sample2", rows=[["c1", 1, 0, 0, 0, 0]], metadata=[
            ["c1", "other-sample", "donor-b", "spleen", "CD8 T cell", "rest", 24]])
        self.atlas.ingest_counts(*paths, batch_id="b2", source="fixture")
        self.assertEqual(self.atlas.summary()["n_cells"], 6)
        self.assertEqual(self.atlas.summary()["n_pseudobulk_groups"], 4)
        paths = self.write_batch("dup", rows=[["c1", 1, 0, 0, 0, 0], ["c1", 1, 0, 0, 0, 0]], metadata=[
            ["c1", "third-sample", "donor-c", "liver", "CD8 T cell", "rest", 0]])
        with self.assertRaisesRegex(ValueError, "duplicate cell_id in counts"):
            self.atlas.ingest_counts(*paths, batch_id="b3", source="fixture")

    def test_gene_panels_and_timepoints_do_not_create_unmeasured_zeros(self):
        self.atlas.ingest_counts(*self.write_batch(), batch_id="b1", source="fixture")
        paths = self.write_batch("panel2", genes=["CD3D"], rows=[["new", 8]], metadata=[
            ["new", "liver-sample", "donor-a", "liver", "CD8 T cell", "rest", 0]])
        self.atlas.ingest_counts(*paths, batch_id="b2", source="fixture")
        states = self.atlas.pseudobulk_records()
        self.assertEqual(len(states), 4)
        new_state = next(state for state in states if list(state["outcomes"]) == ["CD3D"])
        self.assertNotIn("IFNG", new_state["outcomes"])
        self.assertEqual(new_state["n_cells"], 1)

    def test_raw_counts_and_metadata_required_before_commit(self):
        for bad in (-1, 1.5, float("nan"), float("inf")):
            with self.subTest(value=bad):
                paths = self.write_batch("invalid", rows=[["c1", bad, 0, 0, 0, 0]], metadata=[
                    ["c1", "s", "d", "liver", "CD8 T cell", "rest", 0]])
                with self.assertRaisesRegex(ValueError, "raw counts"):
                    self.atlas.ingest_counts(*paths, batch_id="bad", source="fixture")
        self.assertEqual(self.atlas.summary()["n_cells"], 0)
        paths = self.write_batch("missing", rows=[["c1", 1, 0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "metadata contains cells absent"):
            self.atlas.ingest_counts(*paths, batch_id="bad", source="fixture")
        self.assertEqual(self.atlas.summary()["n_cells"], 0)

    def test_mito_qc_and_observational_labels_are_not_responses(self):
        paths = self.write_batch()
        report = self.atlas.ingest_counts(*paths, batch_id="b1", source="fixture", max_mito_fraction=0.05)
        self.assertEqual(report["qc_failed"], 2)
        with self.assertRaisesRegex(ValueError, "atlas|state|cannot|response"):
            validate_dataset(self.atlas.pseudobulk_records())
        with self.assertRaisesRegex(ValueError, "observational or synthetic"):
            self.atlas.ingest_counts(*paths, batch_id="b2", source="fixture", evidence_type="experimental")

    def test_h5ad_dependency_error_and_explicit_layer(self):
        with self.assertRaisesRegex(ValueError, "counts_layer"):
            self.atlas.ingest_h5ad(self.root / "fake.h5ad", counts_layer="", batch_id="b", source="fixture")
        with mock.patch.dict("sys.modules", {"anndata": None}):
            with self.assertRaisesRegex(ImportError, "optional dependencies"):
                self.atlas.ingest_h5ad(self.root / "fake.h5ad", counts_layer="counts", batch_id="b", source="fixture")

    @unittest.skipUnless(importlib.util.find_spec("anndata"), "optional anndata is not installed")
    def test_h5ad_explicit_counts_layer(self):
        import anndata as ad
        import pandas as pd
        matrix = ad.AnnData(np.array([[0.2, 0.3]]),
                            obs=pd.DataFrame({"sample_id": ["s"], "donor_id": ["d"], "organ": ["liver"],
                                              "cell_type": ["CD8 T cell"], "condition": ["rest"]}, index=["c"]),
                            var=pd.DataFrame(index=["CD3D", "IFNG"]))
        matrix.layers["counts"] = np.array([[2, 3]])
        path = self.root / "input.h5ad"
        matrix.write_h5ad(path)
        with self.assertRaisesRegex(ValueError, "raw counts"):
            self.atlas.ingest_h5ad(path, counts_layer="X", batch_id="bad", source="fixture")
        report = self.atlas.ingest_h5ad(path, counts_layer="counts", batch_id="good", source="fixture")
        self.assertEqual(report["added_cells"], 1)


class ReferenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, name="summary", *, unit="log2_fold_change", source="GSE314342", multiplier=1):
        path = self.root / (name + ".npz")
        effects = np.array([[[0.0, 4.0, 8.0], [-1.0, 5.0, np.nan]],
                            [[2.0, 3.0, 8.0], [-2.0, 1.0, 7.0]]]) * multiplier
        se = np.array([[[0.5, 0.0, 1.0], [0.2, 0.3, 0.4]],
                       [[0.4, 0.3, 0.4], [0.2, 0.1, 0.0]]])
        kwargs = {"genes": np.array(["IL2", "IFNG", "TOX"]), "targets": np.array(["IL2", "UNKNOWN"]),
                  "conditions": np.array(["Rest", "Stim8hr"]), "effects": effects, "uncertainty": se,
                  "baseline": np.full((2, 3), 999.0 * multiplier), "counts": np.full((2, 2), 777),
                  "source": np.array(source)}
        if unit is not None:
            kwargs["effect_unit"] = np.array(unit)
        np.savez_compressed(path, **kwargs)
        return path

    def test_mask_units_provenance_and_groups(self):
        records, report = import_reference_npz(self.archive(), organ="blood", cell_type="CD4 T cell", outcome_genes=["IL2", "IFNG", "TOX", "ABSENT"])
        self.assertEqual(len(records), 4)
        self.assertEqual(report["masked_outcome_entries"], 3)
        self.assertEqual(report["missing_requested_genes"], ["ABSENT"])
        first = records[0]
        self.assertEqual(first["outcomes"]["IL2"], 0.0)  # A measured zero is legitimate.
        self.assertNotIn("IFNG", first["outcomes"])  # Missing zero-SE entries are not.
        self.assertEqual(first["outcome_units"]["IL2"], "log2_fold_change")
        self.assertEqual(first["measurement_sd"]["IL2"], 0.5)
        self.assertEqual(first["evidence_type"], "experimental_summary")
        self.assertEqual(first["source_record_id"], first["experiment_id"])
        self.assertEqual(len({record["group_id"] for record in records}), 2)
        self.assertEqual(len(first["provenance"]["archive_sha256"]), 64)
        self.assertNotIn("dose", first)
        self.assertNotIn("donor_id", first)
        self.assertNotIn("biological_time_hours", first)
        validate_dataset(records)

    def test_features_do_not_use_effects_baseline_or_assay_counts(self):
        a, _ = import_reference_npz(self.archive("a"), organ="blood", cell_type="CD4 T cell")
        b, _ = import_reference_npz(self.archive("b", multiplier=17), organ="blood", cell_type="CD4 T cell")
        self.assertEqual([row["features"] for row in a], [row["features"] for row in b])
        self.assertNotEqual(a[-1]["outcomes"], b[-1]["outcomes"])
        self.assertFalse(any("baseline" in name or "count" in name for name in a[0]["features"]))

    def test_units_are_required_and_conflicting_override_rejected(self):
        path = self.archive(unit=None)
        with self.assertRaisesRegex(ValueError, "effect_unit"):
            import_reference_npz(path, organ="blood", cell_type="CD4 T cell")
        records, _ = import_reference_npz(path, organ="blood", cell_type="CD4 T cell", effect_unit_override="log2_fold_change")
        self.assertFalse(records[0]["provenance"]["effect_unit_declared"])
        with self.assertRaisesRegex(ValueError, "conflicting declared unit"):
            import_reference_npz(self.archive("declared"), organ="blood", cell_type="CD4 T cell", effect_unit_override="log1p_cp10k")

    def test_blood_evidence_cannot_be_relabelled_as_liver_or_spleen(self):
        path = self.archive()
        for organ in ("liver", "spleen"):
            with self.subTest(organ=organ):
                with self.assertRaisesRegex(ValueError, "blood-derived"):
                    import_reference_npz(path, organ=organ, cell_type="CD4 T cell")
        with self.assertRaisesRegex(ValueError, "CD4"):
            import_reference_npz(path, organ="blood", cell_type="hepatocyte")
        with self.assertRaisesRegex(ValueError, "none of the requested"):
            import_reference_npz(path, organ="blood", cell_type="CD4 T cell", outcome_genes=["MISSING"])

    def test_source_is_not_inferred_from_filename(self):
        path = self.root / "GSE314342.npz"
        np.savez(path, genes=["IL2"], conditions=["Rest"], targets=["IL2"],
                 effects=np.zeros((1, 1, 1)), uncertainty=np.ones((1, 1, 1)), effect_unit="log2_fold_change")
        with self.assertRaisesRegex(ValueError, "source provenance"):
            import_reference_npz(path, organ="blood", cell_type="CD4 T cell")


if __name__ == "__main__":
    unittest.main()
