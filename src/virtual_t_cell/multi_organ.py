"""Connected multi-organ virtual immune system.

This is a transparent compartment network joining hematopoiesis, thymic
maturation, circulating PBMC, spleen, anatomical lymph nodes, thyroid and a
tumor site. It is designed for research simulation and explicit replacement of
mechanistic priors with matched human measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ORGANS = (
    "bone_marrow", "thymus", "blood_pbmc", "spleen", "cervical_ln",
    "axillary_ln", "mediastinal_ln", "mesenteric_ln", "pelvic_ln",
    "thyroid", "tumor",
)
CELL_TYPES = ("progenitor", "CD8_naive", "CD4_naive", "antigen_specific_CD8",
              "CD8_exhausted", "B_cell", "NK", "myeloid_APC")
SITE_TO_DRAINING_LN = {
    "thyroid": "cervical_ln", "lung": "mediastinal_ln", "breast": "axillary_ln",
    "colorectal": "mesenteric_ln", "pelvic": "pelvic_ln", "generic": "cervical_ln",
}


def _initial_state() -> pd.DataFrame:
    profiles = {
        "bone_marrow": [1.0, .03, .04, 0, 0, .15, .04, .22],
        "thymus": [.12, .38, .34, .001, 0, .01, .01, .08],
        "blood_pbmc": [.01, .30, .28, .002, .002, .20, .10, .10],
        "spleen": [.01, .18, .20, .002, .002, .36, .08, .16],
        "cervical_ln": [0, .30, .28, .001, .001, .30, .03, .09],
        "axillary_ln": [0, .30, .28, .001, .001, .30, .03, .09],
        "mediastinal_ln": [0, .30, .28, .001, .001, .30, .03, .09],
        "mesenteric_ln": [0, .30, .28, .001, .001, .30, .03, .09],
        "pelvic_ln": [0, .30, .28, .001, .001, .30, .03, .09],
        "thyroid": [0, .015, .02, .0001, .0001, .01, .006, .025],
        "tumor": [0, .006, .008, .0005, .001, .003, .008, .035],
    }
    return pd.DataFrame([{"organ": organ, "cell_type": cell, "abundance": value}
                         for organ, values in profiles.items()
                         for cell, value in zip(CELL_TYPES, values)])


def simulate_multi_organ(
    days: int = 60,
    tumor_site: str = "thyroid",
    tumor_burden: float = .55,
    antigen_immunogenicity: float = .70,
    hla_match: float = .85,
    tcr_affinity: float = .70,
    tumor_mhc_i: float = .75,
    therapy: str = "none",
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if days < 1: raise ValueError("days must be positive")
    if tumor_site not in SITE_TO_DRAINING_LN: raise ValueError(f"tumor_site must be one of {sorted(SITE_TO_DRAINING_LN)}")
    if therapy not in {"none", "anti_PD1", "tumor_radiation", "combination"}: raise ValueError("unsupported therapy")
    for name, value in (("tumor_burden", tumor_burden), ("antigen_immunogenicity", antigen_immunogenicity),
                        ("hla_match", hla_match), ("tcr_affinity", tcr_affinity), ("tumor_mhc_i", tumor_mhc_i)):
        if not 0 <= value <= 1: raise ValueError(f"{name} must be in [0, 1]")
    rng = np.random.default_rng(seed)
    state = _initial_state().set_index(["organ", "cell_type"])["abundance"].to_dict()
    draining_ln = SITE_TO_DRAINING_LN[tumor_site]
    antigen = {organ: 0.0 for organ in ORGANS}; antigen["tumor"] = .02 * tumor_burden
    tumor = float(tumor_burden)
    organ_rows, migration_rows, readout_rows = [], [], []

    def get(organ, cell): return state[(organ, cell)]
    def transfer(day, source, target, cell, fraction, route):
        amount = min(get(source, cell), max(0.0, fraction * get(source, cell)))
        state[(source, cell)] -= amount; state[(target, cell)] += amount
        migration_rows.append({"day": day, "source": source, "target": target,
                               "cell_type": cell, "amount": amount, "route": route})
        return amount

    for day in range(days + 1):
        checkpoint = np.clip(.15 + .60 * tumor + .35 * get("tumor", "CD8_exhausted"), 0, .95)
        if therapy in {"anti_PD1", "combination"}: checkpoint *= .35
        radiation = therapy in {"tumor_radiation", "combination"} and 7 <= day <= 11
        recognition_gate = hla_match * tcr_affinity * tumor_mhc_i * antigen_immunogenicity
        recognition = get("tumor", "antigen_specific_CD8") * recognition_gate
        killing = min(tumor, .65 * recognition * (1 - checkpoint) * tumor)
        radiation_kill = .035 * tumor if radiation else 0.0
        readout_rows.append({"day": day, "tumor_site": tumor_site, "draining_ln": draining_ln,
            "tumor_burden": tumor, "checkpoint_suppression": checkpoint,
            "recognition_gate": recognition_gate, "tumor_recognition": recognition,
            "daily_tumor_killing": killing, "circulating_specific_cd8": get("blood_pbmc", "antigen_specific_CD8"),
            "tumor_infiltrating_specific_cd8": get("tumor", "antigen_specific_CD8")})
        for organ in ORGANS:
            for cell in CELL_TYPES:
                organ_rows.append({"day": day, "organ": organ, "cell_type": cell,
                                   "abundance": get(organ, cell), "antigen_load": antigen[organ]})
        if day == days: break

        # Bone marrow output and thymic education.
        progenitor_out = .018 * get("bone_marrow", "progenitor")
        state[("bone_marrow", "progenitor")] += .015 - progenitor_out
        state[("thymus", "progenitor")] += progenitor_out
        mature = .055 * get("thymus", "progenitor") * (.25 + .75 * hla_match)
        state[("thymus", "progenitor")] -= mature
        state[("thymus", "CD8_naive")] += .55 * mature
        state[("thymus", "CD4_naive")] += .45 * mature
        transfer(day, "thymus", "blood_pbmc", "CD8_naive", .045, "thymic_egress")
        transfer(day, "thymus", "blood_pbmc", "CD4_naive", .045, "thymic_egress")

        # Baseline PBMC recirculation through spleen and every anatomical LN.
        for organ in ("spleen", "cervical_ln", "axillary_ln", "mediastinal_ln", "mesenteric_ln", "pelvic_ln"):
            for cell in ("CD8_naive", "CD4_naive", "B_cell"):
                transfer(day, "blood_pbmc", organ, cell, .008, "immune_surveillance")
                transfer(day, organ, "blood_pbmc", cell, .007, "lymphatic_return")

        # Tumor antigen drains preferentially to the anatomically linked LN; a
        # smaller blood-borne fraction reaches spleen and other nodes.
        released = tumor * (.020 + (.070 if radiation else 0)) + .55 * (killing + radiation_kill)
        antigen["tumor"] = max(0, .66 * antigen["tumor"] + released)
        antigen[draining_ln] = .82 * antigen[draining_ln] + .30 * antigen["tumor"]
        antigen["spleen"] = .84 * antigen["spleen"] + .035 * antigen["tumor"]
        antigen["blood_pbmc"] = .70 * antigen["blood_pbmc"] + .025 * antigen["tumor"]
        for node in ("cervical_ln", "axillary_ln", "mediastinal_ln", "mesenteric_ln", "pelvic_ln"):
            if node != draining_ln: antigen[node] = .90 * antigen[node] + .005 * antigen["blood_pbmc"]
        antigen["thyroid"] = .88 * antigen["thyroid"] + (.04 * antigen["tumor"] if tumor_site == "thyroid" else .002 * antigen["blood_pbmc"])

        # Antigen-specific priming and expansion occur mainly in the draining LN.
        priming = min(get(draining_ln, "CD8_naive"),
                      .40 * get(draining_ln, "CD8_naive") * antigen[draining_ln] * hla_match * tcr_affinity * antigen_immunogenicity)
        state[(draining_ln, "CD8_naive")] -= priming
        state[(draining_ln, "antigen_specific_CD8")] += priming
        state[(draining_ln, "antigen_specific_CD8")] *= 1 + .12 * (1 - .35 * checkpoint)
        transfer(day, draining_ln, "blood_pbmc", "antigen_specific_CD8", .22, "tdln_effector_egress")
        transfer(day, "blood_pbmc", "tumor", "antigen_specific_CD8",
                 .18 * (.25 + .75 * antigen_immunogenicity), "tumor_homing")
        # A small memory-like pool surveys spleen and the other nodes.
        transfer(day, "blood_pbmc", "spleen", "antigen_specific_CD8", .02, "systemic_memory")

        exhausted = .05 * get("tumor", "antigen_specific_CD8") * checkpoint
        state[("tumor", "antigen_specific_CD8")] -= exhausted
        state[("tumor", "CD8_exhausted")] += exhausted
        if therapy in {"anti_PD1", "combination"}:
            revived = .04 * get("tumor", "CD8_exhausted")
            state[("tumor", "CD8_exhausted")] -= revived
            state[("tumor", "antigen_specific_CD8")] += revived
        tumor = float(np.clip(tumor + .033 * tumor * (1 - tumor) - killing - radiation_kill, 0, 1))
        # Keep every state non-negative and add only tiny reproducible biological noise.
        for key in state: state[key] = max(0.0, state[key] + rng.normal(0, 1e-6))
    return pd.DataFrame(organ_rows), pd.DataFrame(migration_rows), pd.DataFrame(readout_rows)


def run_multi_organ(out_dir: Path, **kwargs) -> dict:
    organs, migration, readouts = simulate_multi_organ(**kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    organs.to_csv(out_dir / "multi_organ_cell_trajectory.csv", index=False)
    migration.to_csv(out_dir / "immune_cell_migration.csv", index=False)
    readouts.to_csv(out_dir / "tumor_immunity_readouts.csv", index=False)
    summary = {"model": "multi_organ_virtual_immune_system_v1", **kwargs,
        "organs": list(ORGANS), "cell_types": list(CELL_TYPES),
        "draining_ln": readouts.draining_ln.iloc[0],
        "initial_tumor_burden": float(readouts.tumor_burden.iloc[0]),
        "final_tumor_burden": float(readouts.tumor_burden.iloc[-1]),
        "cumulative_tumor_killing": float(readouts.daily_tumor_killing.sum()),
        "clinical_use": False,
        "boundary": "Connected mechanistic compartments; not a patient-specific clinical digital twin."}
    (out_dir / "multi_organ_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
