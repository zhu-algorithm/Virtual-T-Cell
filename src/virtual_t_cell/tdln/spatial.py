"""Multi-scale spatial and cell-communication layer for the TDLN model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import CELL_TYPES, PROGRAMS


ZONES = ("subcapsular_sinus", "t_cell_zone", "b_cell_follicle", "medulla",
         "vascular_stroma", "metastatic_niche")

HOME_ZONE = {
    "CD8_naive": "t_cell_zone", "CD8_effector": "t_cell_zone", "CD8_exhausted": "metastatic_niche",
    "CD4_naive": "t_cell_zone", "Tfh": "b_cell_follicle", "Treg": "t_cell_zone",
    "B_naive": "b_cell_follicle", "GC_B": "b_cell_follicle", "plasma": "medulla",
    "NK": "subcapsular_sinus", "cDC1": "t_cell_zone", "cDC2": "t_cell_zone",
    "pDC": "t_cell_zone", "macrophage": "subcapsular_sinus", "monocyte": "subcapsular_sinus",
    "lymphatic_endothelial": "subcapsular_sinus", "blood_endothelial": "vascular_stroma",
    "fibroblast": "vascular_stroma", "tumor_cell": "metastatic_niche",
}

COMMUNICATION = (
    ("cDC1", "CD8_naive", "antigen-MHC_I", "TCR", "antigen"),
    ("cDC2", "CD4_naive", "antigen-MHC_II", "TCR", "antigen"),
    ("Tfh", "GC_B", "IL21", "IL21R", "IL2"),
    ("Treg", "CD8_effector", "TGFB1", "TGFBR", "TGFb"),
    ("CD8_exhausted", "cDC1", "PDCD1", "CD274", "checkpoint"),
    ("CD8_effector", "tumor_cell", "IFNG", "IFNGR", "IFNG"),
    ("tumor_cell", "lymphatic_endothelial", "VEGFC", "FLT4", "VEGF_C"),
    ("fibroblast", "CD8_effector", "CXCL12", "CXCR4", "TGFb"),
)


def spatialize(base: pd.DataFrame, metastasis: bool) -> pd.DataFrame:
    rows = []
    for cell in base.itertuples(index=False):
        home = HOME_ZONE[cell.cell_type]
        home_share = .82 if cell.cell_type != "tumor_cell" else (.96 if metastasis else .75)
        for zone in ZONES:
            share = home_share if zone == home else (1 - home_share) / (len(ZONES) - 1)
            rows.append({**cell._asdict(), "zone": zone, "abundance": cell.abundance * share})
    state = pd.DataFrame(rows)
    state.abundance /= state.abundance.sum()
    return state


def zone_signals(state: pd.DataFrame, tumor_burden: float, therapy: str) -> pd.DataFrame:
    rows = []
    for zone, frame in state.groupby("zone"):
        weighted = (frame.activation * frame.abundance).groupby(frame.cell_type).sum().to_dict()
        local_tumor = float(frame.loc[frame.cell_type == "tumor_cell", "abundance"].sum())
        antigen = np.clip(.08 + .65 * tumor_burden + 2.5 * local_tumor +
                          (.25 if therapy == "tumor_radiation" else 0), 0, 1)
        checkpoint = .2 * weighted.get("Treg", 0) + .3 * weighted.get("CD8_exhausted", 0) + .2 * tumor_burden
        if therapy == "anti_PD1": checkpoint *= .45
        rows.append({"zone": zone, "antigen": antigen,
            "IL2": np.clip(.15 + 3 * weighted.get("CD4_naive", 0) + weighted.get("CD8_effector", 0), 0, 1),
            "IFNG": np.clip(.08 + 4 * weighted.get("CD8_effector", 0) + 2 * weighted.get("NK", 0), 0, 1),
            "TGFb": np.clip(.1 + 4 * weighted.get("Treg", 0) + .4 * tumor_burden, 0, 1),
            "checkpoint": np.clip(checkpoint, 0, 1),
            "VEGF_C": np.clip(.05 + .65 * tumor_burden + 2 * local_tumor, 0, 1)})
    return pd.DataFrame(rows)


def communication_scores(state: pd.DataFrame, signals: pd.DataFrame, day: int) -> pd.DataFrame:
    rows = []
    sig = signals.set_index("zone")
    for zone, frame in state.groupby("zone"):
        local = frame.set_index("cell_type")
        for sender, receiver, ligand, receptor, signal in COMMUNICATION:
            s = float(local.at[sender, "abundance"] * local.at[sender, "activation"])
            r = float(local.at[receiver, "abundance"] * (.5 + local.at[receiver, "activation"] / 2))
            score = s * r * float(sig.at[zone, signal]) * 1000
            rows.append({"day": day, "zone": zone, "sender": sender, "receiver": receiver,
                         "ligand": ligand, "receptor": receptor, "score": score})
    return pd.DataFrame(rows)


def simulate_multiscale(base: pd.DataFrame, days: int, tumor_burden: float,
                        therapy: str, metastasis: bool, seed: int):
    rng = np.random.default_rng(seed)
    state = spatialize(base, metastasis)
    cell_rows, program_rows, comm_rows, organ_rows = [], [], [], []
    for day in range(days + 1):
        signals = zone_signals(state, tumor_burden, therapy)
        snap = state.merge(signals, on="zone")
        snap.insert(0, "day", day)
        cell_rows.append(snap.copy())
        comm_rows.append(communication_scores(state, signals, day))
        for row in signals.itertuples(index=False):
            values = {"TCR_activation": row.antigen * (1 - row.checkpoint),
                      "cytotoxicity": row.IFNG * (1 - row.checkpoint),
                      "exhaustion": row.antigen * row.checkpoint,
                      "antigen_presentation": row.antigen * (.5 + row.IFNG / 2),
                      "germinal_center": np.clip(.35 + row.IL2 - .4 * row.TGFb, 0, 1),
                      "IFN_response": row.IFNG, "immunosuppression": row.TGFb,
                      "lymphangiogenesis": row.VEGF_C}
            program_rows.extend({"day": day, "zone": row.zone, "program": k, "score": float(v)}
                                for k, v in values.items())
        tumor = float(state.loc[state.cell_type == "tumor_cell", "abundance"].sum())
        eff = float((state.loc[state.cell_type == "CD8_effector", "abundance"] *
                     state.loc[state.cell_type == "CD8_effector", "activation"]).sum())
        treg = float(state.loc[state.cell_type == "Treg", "abundance"].sum())
        gc = float(state.loc[state.cell_type == "GC_B", "abundance"].sum())
        organ_rows.append({"day": day, "tumor_occupancy": tumor, "effective_cytotoxicity": eff,
                           "immune_suppression_index": treg + signals.TGFb.mean(),
                           "germinal_center_mass": gc, "metastatic_risk_proxy": tumor * signals.VEGF_C.mean()})
        if day == days: break
        local = state.merge(signals, on="zone")
        growth = np.zeros(len(state))
        mask = state.cell_type == "CD8_effector"
        growth[mask] = (.12 * local.antigen * (1 - local.checkpoint))[mask]
        mask = state.cell_type == "CD8_exhausted"
        growth[mask] = (.07 * local.antigen * local.checkpoint)[mask]
        mask = state.cell_type == "Treg"
        growth[mask] = (.05 * local.TGFb)[mask]
        mask = state.cell_type == "GC_B"
        growth[mask] = (.04 * np.clip(.35 + local.IL2 - .4 * local.TGFb, 0, 1))[mask]
        mask = state.cell_type == "cDC1"
        growth[mask] = (.035 * local.antigen)[mask]
        mask = state.cell_type == "lymphatic_endothelial"
        growth[mask] = (.04 * local.VEGF_C)[mask]
        mask = state.cell_type == "tumor_cell"
        growth[mask] = (.07 * tumor_burden - .08 * local.IFNG * (1 - local.checkpoint))[mask]
        state.abundance = np.clip(state.abundance * (1 + growth + rng.normal(0, .0015, len(state))), 1e-7, None)
        # Homing is an explicit spatial transition, not a cell-type relabeling.
        for cell_type, indices in state.groupby("cell_type").groups.items():
            total = state.loc[indices, "abundance"].sum()
            desired = np.asarray([.82 if z == HOME_ZONE[cell_type] else .18 / (len(ZONES) - 1)
                                  for z in state.loc[indices, "zone"]]) * total
            state.loc[indices, "abundance"] = .94 * state.loc[indices, "abundance"] + .06 * desired
        state.abundance /= state.abundance.sum()
        immune = ~state.cell_type.isin(["fibroblast", "blood_endothelial", "lymphatic_endothelial", "tumor_cell"])
        target = np.clip(.12 + .65 * local.antigen - .4 * local.checkpoint, 0, 1)
        state.loc[immune, "activation"] = .75 * state.loc[immune, "activation"] + .25 * target[immune]
        state["proliferation"] = np.clip(.04 + growth, 0, 1)
        state["apoptosis"] = np.clip(.025 + .05 * local.checkpoint, 0, 1)
    return (pd.concat(cell_rows, ignore_index=True), pd.DataFrame(program_rows),
            pd.concat(comm_rows, ignore_index=True), pd.DataFrame(organ_rows))
