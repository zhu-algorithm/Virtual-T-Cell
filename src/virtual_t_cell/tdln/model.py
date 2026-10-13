"""Transparent compartment model for a tumor-draining lymph node.

This is an executable mechanistic baseline. Parameters are qualitative and are
not fitted clinical effect estimates; public omics are used as evidence layers
and future calibration inputs, never silently treated as paired measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CELL_TYPES = (
    "CD8_naive", "CD8_effector", "CD8_exhausted", "CD4_naive", "Tfh", "Treg",
    "B_naive", "GC_B", "plasma", "NK", "cDC1", "cDC2", "pDC", "macrophage",
    "monocyte", "lymphatic_endothelial", "blood_endothelial", "fibroblast",
    "tumor_cell",
)

PROGRAMS = ("TCR_activation", "cytotoxicity", "exhaustion", "antigen_presentation",
            "germinal_center", "IFN_response", "immunosuppression", "lymphangiogenesis")


def default_state(calibration_model: Path | None = None) -> pd.DataFrame:
    abundance = np.asarray([.12, .04, .02, .12, .035, .04, .20, .06, .015, .035,
                            .025, .03, .01, .035, .025, .035, .02, .06, .02])
    activation = np.asarray([.1, .55, .35, .08, .45, .35, .1, .5, .65, .45,
                             .5, .4, .35, .4, .25, .2, .15, .25, .45])
    if calibration_model is not None:
        payload = json.loads(Path(calibration_model).read_text(encoding="utf-8"))
        abundance = np.asarray([payload["baseline_abundance"][c] for c in CELL_TYPES])
        activation = np.asarray([payload["activation"][c] for c in CELL_TYPES])
    return pd.DataFrame({"cell_type": CELL_TYPES, "abundance": abundance / abundance.sum(),
                         "activation": activation, "proliferation": .08, "apoptosis": .03})


def _signals(state: pd.DataFrame, tumor_burden: float, therapy: str) -> dict[str, float]:
    a = dict(zip(state.cell_type, state.activation * state.abundance))
    checkpoint = 0.28 * a["Treg"] + 0.30 * a["CD8_exhausted"] + 0.22 * tumor_burden
    if therapy == "anti_PD1":
        checkpoint *= 0.45
    radiation_danger = .28 if therapy == "tumor_radiation" else 0.0
    return {
        "antigen": np.clip(.15 + .75 * tumor_burden + radiation_danger, 0, 1),
        "IL2": np.clip(.2 + 1.7 * a["CD4_naive"] + .8 * a["CD8_effector"], 0, 1),
        "IFNG": np.clip(.1 + 2.0 * a["CD8_effector"] + 1.2 * a["NK"], 0, 1),
        "TGFb": np.clip(.12 + 2.0 * a["Treg"] + .45 * tumor_burden, 0, 1),
        "checkpoint": np.clip(checkpoint, 0, 1),
        "VEGF_C": np.clip(.08 + .7 * tumor_burden, 0, 1),
    }


def simulate(days: int = 14, tumor_burden: float = .35, therapy: str = "none",
             metastasis: bool = False, seed: int = 13,
             calibration_model: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate spatial cell states; retained as the compact two-table API."""
    if therapy not in {"none", "anti_PD1", "tumor_radiation"}:
        raise ValueError("therapy must be none, anti_PD1, or tumor_radiation")
    if not 0 <= tumor_burden <= 1 or days < 1:
        raise ValueError("tumor_burden must be in [0,1] and days must be positive")
    from .spatial import simulate_multiscale
    cells, programs, _, _ = simulate_multiscale(default_state(calibration_model), days, tumor_burden, therapy, metastasis, seed)
    return cells, programs


def run_simulation(out_dir: Path, **kwargs) -> dict:
    from .spatial import ZONES, COMMUNICATION, simulate_multiscale
    calibration_model = kwargs.pop("calibration_model", None)
    cells, programs, communications, organ = simulate_multiscale(default_state(calibration_model), **kwargs)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out_dir / "cell_state_trajectory.csv", index=False)
    programs.to_csv(out_dir / "pathway_trajectory.csv", index=False)
    communications.to_csv(out_dir / "cell_communication_trajectory.csv", index=False)
    organ.to_csv(out_dir / "organ_readouts.csv", index=False)
    summary = {"model": "multiscale_spatial_communication_v3", **kwargs,
               "calibration_model": str(calibration_model) if calibration_model else None,
               "cell_types": len(CELL_TYPES), "programs": len(PROGRAMS),
               "zones": len(ZONES), "communication_axes": len(COMMUNICATION),
               "data_boundary": "Initial state is public-cohort calibrated when a calibration model is supplied; dynamics remain mechanistic and non-clinical.",
               "clinical_use": False}
    (out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
