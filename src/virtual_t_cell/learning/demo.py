"""Small explicitly synthetic fixtures for exercising the research pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .schema import write_jsonl


def make_response_fixture(*, seed: int = 7, groups: int = 80) -> list[dict]:
    """Numerical test law, not a claim about real liver or spleen cell biology."""
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(groups):
        x = float(rng.uniform(-1, 1))
        z = float(rng.uniform(-1, 1))
        for organ in ("liver", "spleen"):
            effect = 0.35 if organ == "liver" else -0.35
            outcomes = {"response_a": float(1.8 * x - 0.9 * z + effect + rng.normal(0, 0.08)),
                        "response_b": float(-0.7 * x + 0.5 * z - effect + rng.normal(0, 0.08))}
            rows.append({"observation_id": f"synthetic:{seed}:{group}:{organ}",
                         "experiment_id": f"synthetic:{seed}:{group}:{organ}",
                         "group_id": f"synthetic:{seed}:{group}",
                         "organ": organ, "cell_type": "CD8_T", "condition": "fixture_context", "species": "synthetic",
                         "intervention": f"fixture_target_{group}",
                         "evidence_type": "synthetic", "source": "synthetic_numerical_fixture_v1",
                         "features": {"descriptor_x": x, "descriptor_z": z}, "outcomes": outcomes,
                         "outcome_units": {name: "synthetic_score" for name in outcomes},
                         "measurement_sd": {name: 0.08 for name in outcomes}})
    return rows


def write_demo_inputs(directory: str | Path) -> dict:
    """Prepare reproducible inputs; never fetch, sequence, or simulate real patients."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = make_response_fixture()
    splits = {"train": rows[:64], "new": rows[64:104], "validation": rows[104:132], "test": rows[132:]}
    for name, split in splits.items():
        write_jsonl(directory / f"{name}.jsonl", split)
    candidates = []
    for index in range(6):
        organ = "liver" if index % 2 == 0 else "spleen"
        candidates.append({"observation_id": f"candidate:{index}", "candidate_id": f"candidate:{index}",
                           "experiment_id": f"future:{index}", "group_id": f"future:{index}",
                           "organ": organ, "cell_type": "CD8_T", "condition": "fixture_context", "species": "synthetic",
                           "intervention": f"hypothetical_target_{index}", "evidence_type": "model_prediction",
                           "source": "synthetic_candidate_specification", "features": {
                               "descriptor_x": -0.65 + index * 0.25, "descriptor_z": 0.4 - index * 0.12},
                           "outcomes": {}, "outcome_units": {}, "cost": float(1 + index % 3)})
    write_jsonl(directory / "candidates.jsonl", candidates)
    inbox = directory / "inbox"
    inbox.mkdir(exist_ok=True)
    genes = ["CD3D", "CD8A", "MS4A1", "MT-CO1"]
    for batch_number in (1, 2):
        count_path = directory / f"synthetic_counts_{batch_number}.csv"
        meta_path = directory / f"synthetic_metadata_{batch_number}.csv"
        with count_path.open("w", newline="", encoding="utf-8") as cfile, meta_path.open("w", newline="", encoding="utf-8") as mfile:
            cwriter = csv.writer(cfile)
            mwriter = csv.writer(mfile)
            cwriter.writerow(["cell_id", *genes])
            mwriter.writerow(["cell_id", "sample_id", "donor_id", "organ", "cell_type", "condition", "biological_time_hours"])
            for index in range(8):
                organ = "liver" if index < 4 else "spleen"
                cell_type = "CD8_T" if index % 4 < 3 else "B_cell"
                counts = [5 + index, 4, 0, 1] if cell_type == "CD8_T" else [0, 0, 10 + index, 1]
                # Sample-scoped barcodes intentionally recur in a different batch.
                cell_id = f"barcode-{index}"
                cwriter.writerow([cell_id, *counts])
                mwriter.writerow([cell_id, f"synthetic_sample_{batch_number}_{organ}",
                                  f"synthetic_donor_{batch_number}", organ, cell_type,
                                  "synthetic_unperturbed", (batch_number - 1) * 24])
        manifest = {"schema_version": 1, "batch_id": f"synthetic_batch_{batch_number}",
                    "source": "synthetic_atlas_fixture_v1", "atlas_evidence_type": "synthetic",
                    "atlas": {"counts_csv": f"../{count_path.name}", "metadata_csv": f"../{meta_path.name}",
                              "min_counts": 1, "min_genes": 1, "max_mito_fraction": 0.5}}
        (inbox / f"batch_{batch_number}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"evidence_type": "synthetic", "description": "Numerical integration fixture; not real sequencing or biology.",
            "split_sizes": {name: len(split) for name, split in splits.items()},
            "inbox": str(inbox)}
