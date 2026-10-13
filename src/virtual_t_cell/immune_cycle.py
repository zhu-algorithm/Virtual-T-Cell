"""Executable whole-body tumor-immunity cycle.

The model links thymic T-cell maturation to antigen-specific priming in a
tumor-draining lymph node, blood trafficking, tumor infiltration, recognition,
killing and antigen re-release. Parameters are mechanistic research priors, not
clinical effect estimates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STAGES = (
    "bone_marrow_output", "thymic_positive_selection", "thymic_negative_selection",
    "naive_repertoire", "tumor_antigen_release", "dendritic_cell_capture",
    "tdln_priming", "clonal_expansion", "blood_trafficking", "tumor_infiltration",
    "tcr_pmhc_recognition", "tumor_killing", "antigen_feedback",
)


def _bounded(name: str, value: float) -> float:
    if not 0 <= value <= 1: raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def simulate_immune_cycle(
    days: int = 42,
    tumor_burden: float = .55,
    antigen_immunogenicity: float = .70,
    hla_match: float = .85,
    tcr_affinity: float = .70,
    tumor_mhc_i: float = .75,
    self_similarity: float = .20,
    therapy: str = "none",
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run maturation, antigen recognition, trafficking and killing dynamics."""
    if days < 1: raise ValueError("days must be positive")
    if therapy not in {"none", "anti_PD1", "tumor_radiation", "combination"}:
        raise ValueError("unsupported therapy")
    tumor = _bounded("tumor_burden", tumor_burden)
    immunogenicity = _bounded("antigen_immunogenicity", antigen_immunogenicity)
    hla = _bounded("hla_match", hla_match)
    affinity = _bounded("tcr_affinity", tcr_affinity)
    mhc = _bounded("tumor_mhc_i", tumor_mhc_i)
    self_like = _bounded("self_similarity", self_similarity)
    rng = np.random.default_rng(seed)

    state = {
        "marrow_progenitor": .80, "thymus_DN": .10, "thymus_DP": .18,
        "thymus_CD8_SP": .035, "naive_tumor_specific": .015,
        "dc_antigen_loaded": 0.0, "tdln_activated_clone": 0.0,
        "blood_effector": 0.0, "tumor_infiltrating_effector": 0.0,
        "tumor_infiltrating_exhausted": 0.0, "tumor": tumor,
        "extracellular_antigen": .01 * tumor,
    }
    cumulative = {name: 0.0 for name in STAGES}
    rows, stage_rows = [], []
    for day in range(days + 1):
        checkpoint = np.clip(.12 + .62 * state["tumor"] +
                             .35 * state["tumor_infiltrating_exhausted"], 0, .95)
        if therapy in {"anti_PD1", "combination"}: checkpoint *= .35
        radiation = therapy in {"tumor_radiation", "combination"} and 7 <= day <= 11

        marrow_out = .025 * state["marrow_progenitor"]
        dn_to_dp = .12 * state["thymus_DN"]
        positive = .075 * state["thymus_DP"] * (.25 + .75 * hla)
        # High self similarity removes clones during negative selection; tumor-distinct
        # antigens retain a small cognate precursor pool.
        negative_loss = positive * (.25 + .70 * self_like)
        mature_cognate = (positive - negative_loss) * (.01 + .09 * immunogenicity * affinity)
        shedding = state["tumor"] * (.018 + (.07 if radiation else 0))
        dc_capture = .30 * state["extracellular_antigen"] * immunogenicity
        dc_migration = .22 * state["dc_antigen_loaded"]
        recognition_gate = hla * affinity * immunogenicity
        priming = min(state["naive_tumor_specific"],
                      3.0 * state["naive_tumor_specific"] * dc_migration * recognition_gate)
        expansion = state["tdln_activated_clone"] * (.30 + .75 * dc_migration) * (1 - .35 * checkpoint)
        egress = .25 * state["tdln_activated_clone"]
        infiltration = .25 * state["blood_effector"] * (.25 + .75 * immunogenicity)
        recognize = state["tumor_infiltrating_effector"] * affinity * hla * mhc
        killing = min(state["tumor"], 1.2 * recognize * (1 - checkpoint) * state["tumor"])
        direct_radiation_kill = .035 * state["tumor"] if radiation else 0.0
        exhaustion = .055 * state["tumor_infiltrating_effector"] * checkpoint
        reinvigoration = (.045 * state["tumor_infiltrating_exhausted"]
                          if therapy in {"anti_PD1", "combination"} else 0.0)
        tumor_growth = .035 * state["tumor"] * (1 - state["tumor"])
        feedback = .55 * (killing + direct_radiation_kill)

        flows = {
            "bone_marrow_output": marrow_out, "thymic_positive_selection": positive,
            "thymic_negative_selection": negative_loss, "naive_repertoire": mature_cognate,
            "tumor_antigen_release": shedding, "dendritic_cell_capture": dc_capture,
            "tdln_priming": priming, "clonal_expansion": expansion,
            "blood_trafficking": egress, "tumor_infiltration": infiltration,
            "tcr_pmhc_recognition": recognize, "tumor_killing": killing,
            "antigen_feedback": feedback,
        }
        for name, value in flows.items():
            cumulative[name] += value
            stage_rows.append({"day": day, "stage": name, "flux": value,
                               "cumulative_flux": cumulative[name]})
        rows.append({"day": day, **state, "checkpoint_suppression": checkpoint,
                     "recognition_gate": recognition_gate, "daily_tumor_killing": killing,
                     "cycle_completion": sum(v > 0 for v in cumulative.values()) / len(STAGES)})
        if day == days: break

        state["marrow_progenitor"] = max(.1, state["marrow_progenitor"] + .02 - marrow_out)
        state["thymus_DN"] = max(0, state["thymus_DN"] + marrow_out - dn_to_dp)
        state["thymus_DP"] = max(0, state["thymus_DP"] + dn_to_dp - positive)
        state["thymus_CD8_SP"] = max(0, state["thymus_CD8_SP"] + positive - negative_loss - mature_cognate)
        state["naive_tumor_specific"] = max(0, state["naive_tumor_specific"] + mature_cognate - priming)
        state["dc_antigen_loaded"] = max(0, .86 * state["dc_antigen_loaded"] + dc_capture - dc_migration)
        state["tdln_activated_clone"] = max(0, state["tdln_activated_clone"] + priming + expansion - egress)
        state["blood_effector"] = max(0, .90 * state["blood_effector"] + egress - infiltration)
        state["tumor_infiltrating_effector"] = max(0, .94 * state["tumor_infiltrating_effector"] + infiltration - exhaustion + reinvigoration)
        state["tumor_infiltrating_exhausted"] = max(0, .97 * state["tumor_infiltrating_exhausted"] + exhaustion - reinvigoration)
        state["tumor"] = float(np.clip(state["tumor"] + tumor_growth - killing - direct_radiation_kill, 0, 1))
        noise = rng.normal(0, .0002)
        state["extracellular_antigen"] = max(0, .65 * state["extracellular_antigen"] + shedding + feedback + noise - dc_capture)
    return pd.DataFrame(rows), pd.DataFrame(stage_rows)


def run_immune_cycle(out_dir: Path, **kwargs) -> dict:
    trajectory, flux = simulate_immune_cycle(**kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(out_dir / "immune_cycle_trajectory.csv", index=False)
    flux.to_csv(out_dir / "immune_cycle_stage_flux.csv", index=False)
    summary = {
        "model": "whole_body_tumor_immunity_cycle_v1", **kwargs,
        "stages": list(STAGES), "cycle_completed": bool(trajectory.cycle_completion.iloc[-1] == 1),
        "initial_tumor_burden": float(trajectory.tumor.iloc[0]),
        "final_tumor_burden": float(trajectory.tumor.iloc[-1]),
        "cumulative_tumor_killing": float(trajectory.daily_tumor_killing.sum()),
        "boundary": "Mechanistic research simulation; antigen/HLA/TCR inputs are abstract scores, not patient predictions.",
        "clinical_use": False,
    }
    (out_dir / "immune_cycle_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
