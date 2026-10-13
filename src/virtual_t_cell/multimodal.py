"""Lightweight multimodal Virtual T Cell training and inference.

The implementation deliberately uses only NumPy/Pandas.  Each modality has an
independent SVD encoder, missing modalities are represented by an explicit mask,
and ridge decoders predict RNA, ATAC, protein and TCR feature space after adding
perturbation and continuous-time covariates.  It is a reproducible baseline,
not a claim that synthetic examples or unpaired public cohorts are biological
ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = "virtual-t-cell.multimodal.v1"
MODALITIES = ("rna", "atac", "protein", "tcr")
REQUIRED_OBS = (
    "cell_id", "donor_id", "batch", "cell_type", "condition", "time_hours",
    "perturbation", "perturbation_strength", "is_control", "split",
)


def _load_dataset(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    manifest_path = path / "manifest.json"
    arrays_path = path / "arrays.npz"
    if not manifest_path.exists() or not arrays_path.exists():
        raise ValueError("A dataset must contain manifest.json and arrays.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays_file = np.load(arrays_path, allow_pickle=False)
    arrays = {key: arrays_file[key] for key in arrays_file.files}
    return manifest, arrays


def validate_dataset(path: Path) -> dict:
    """Validate the portable NPZ/JSON schema and return a compact summary."""
    manifest, arrays = _load_dataset(path)
    errors = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if manifest.get("data_origin") not in {"real", "simulated", "mixed"}:
        errors.append("data_origin must be real, simulated, or mixed")
    missing_obs = [name for name in REQUIRED_OBS if f"obs__{name}" not in arrays]
    errors.extend(f"missing obs column: {name}" for name in missing_obs)
    n_cells = len(arrays.get("obs__cell_id", []))
    if len(set(arrays.get("obs__cell_id", []).astype(str))) != n_cells:
        errors.append("obs__cell_id values must be unique")
    present = []
    for modality in MODALITIES:
        xkey, fkey, mkey = f"X__{modality}", f"var__{modality}", f"mask__{modality}"
        if xkey not in arrays:
            continue
        present.append(modality)
        x = arrays[xkey]
        if x.ndim != 2 or x.shape[0] != n_cells:
            errors.append(f"{xkey} must have shape (n_cells, n_features)")
        if fkey not in arrays or len(arrays[fkey]) != x.shape[1]:
            errors.append(f"{fkey} must name every feature")
        if mkey not in arrays or arrays[mkey].shape != (n_cells,):
            errors.append(f"{mkey} must have shape (n_cells,)")
        elif not np.isfinite(x[arrays[mkey].astype(bool)]).all():
            errors.append(f"{xkey} contains non-finite observed values")
    if not present:
        errors.append("at least one modality is required")
    for name in REQUIRED_OBS:
        key = f"obs__{name}"
        if key in arrays and len(arrays[key]) != n_cells:
            errors.append(f"{key} length does not match cell count")
    if "obs__time_hours" in arrays and not np.isfinite(arrays["obs__time_hours"].astype(float)).all():
        errors.append("obs__time_hours must be finite")
    if "obs__perturbation_strength" in arrays:
        strength = arrays["obs__perturbation_strength"].astype(float)
        if not np.isfinite(strength).all() or np.any((strength < 0) | (strength > 1)):
            errors.append("obs__perturbation_strength must be finite and between 0 and 1")
    if "obs__split" in arrays:
        unknown = set(arrays["obs__split"].astype(str)) - {"train", "validation", "test"}
        if unknown:
            errors.append(f"obs__split contains unsupported labels: {sorted(unknown)}")
    if errors:
        raise ValueError("Invalid multimodal dataset:\n- " + "\n- ".join(errors))
    return {"schema_version": SCHEMA_VERSION, "cells": n_cells,
            "modalities": present, "data_origin": manifest["data_origin"]}


def _svd_encoder(x: np.ndarray, observed: np.ndarray, latent_dim: int):
    fitted = x[observed]
    mean = fitted.mean(axis=0)
    scale = fitted.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (fitted - mean) / scale
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    k = min(latent_dim, vt.shape[0], vt.shape[1])
    components = vt[:k].astype(np.float32)
    encoded = np.zeros((len(x), k), dtype=np.float32)
    encoded[observed] = ((x[observed] - mean) / scale) @ components.T
    return mean.astype(np.float32), scale.astype(np.float32), components, encoded


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0  # do not regularize the intercept
    return np.linalg.solve(x.T @ x + penalty, x.T @ y).astype(np.float32)


def train_multimodal(dataset: Path, out: Path, latent_dim: int = 8,
                     ridge: float = 1.0) -> dict:
    """Train modality encoders and conditional decoders from a schema dataset."""
    summary = validate_dataset(dataset)
    manifest, a = _load_dataset(dataset)
    n = summary["cells"]
    train = a["obs__split"].astype(str) == "train"
    if train.sum() < 2:
        raise ValueError("At least two cells with split='train' are required")
    perturb = a["obs__perturbation"].astype(str)
    perturbations = sorted(set(perturb[train]) | {"CTRL"})
    cell_types = sorted(set(a["obs__cell_type"].astype(str)[train]))
    conditions = sorted(set(a["obs__condition"].astype(str)[train]))

    model: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "model_type": np.asarray("masked_multiview_svd_ridge_v1"),
        "data_origin": np.asarray(manifest["data_origin"]),
        "modalities": np.asarray(summary["modalities"], dtype=str),
        "perturbations": np.asarray(perturbations, dtype=str),
        "cell_types": np.asarray(cell_types, dtype=str),
        "conditions_vocab": np.asarray(conditions, dtype=str),
        "training_dataset": np.asarray(str(manifest.get("dataset_name", Path(dataset).name))),
    }
    latent_parts = []
    for modality in summary["modalities"]:
        observed = a[f"mask__{modality}"].astype(bool) & train
        if observed.sum() < 2:
            raise ValueError(f"Modality {modality} needs at least two observed training cells")
        mean, scale, components, encoded = _svd_encoder(
            a[f"X__{modality}"].astype(np.float32), observed, latent_dim)
        model[f"mean__{modality}"] = mean
        model[f"scale__{modality}"] = scale
        model[f"components__{modality}"] = components
        model[f"features__{modality}"] = a[f"var__{modality}"].astype(str)
        latent_parts.extend([encoded, a[f"mask__{modality}"].astype(np.float32)[:, None]])

    fused = np.concatenate(latent_parts, axis=1)
    strength = a["obs__perturbation_strength"].astype(np.float32)
    time = a["obs__time_hours"].astype(np.float32)
    time_scale = max(float(np.max(time[train])), 1.0)
    covariates = []
    for name in perturbations:
        covariates.append(((perturb == name).astype(np.float32) * strength)[:, None])
    cell_values = a["obs__cell_type"].astype(str)
    condition_values = a["obs__condition"].astype(str)
    covariates.extend((cell_values == name).astype(np.float32)[:, None] for name in cell_types)
    covariates.extend((condition_values == name).astype(np.float32)[:, None] for name in conditions)
    design = np.concatenate([fused, *covariates, (time / time_scale)[:, None],
                             np.ones((n, 1), dtype=np.float32)], axis=1)
    model["time_scale"] = np.asarray(time_scale, dtype=np.float32)
    model["fused_width"] = np.asarray(fused.shape[1], dtype=np.int32)

    metrics = {}
    for modality in summary["modalities"]:
        observed = a[f"mask__{modality}"].astype(bool) & train
        y = a[f"X__{modality}"].astype(np.float32)
        weights = _ridge(design[observed], y[observed], ridge)
        model[f"decoder__{modality}"] = weights
        pred = design[observed] @ weights
        metrics[modality] = {"observed_train_cells": int(observed.sum()),
                             "train_rmse": float(np.sqrt(np.mean((pred - y[observed]) ** 2))),
                             "features": int(y.shape[1])}

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **model)
    report = {"model_type": "masked_multiview_svd_ridge_v1", "schema": summary,
              "latent_dim_requested": latent_dim, "ridge": ridge, "metrics": metrics,
              "limitations": [
                  "Linear baseline; no nonlinear gene-regulatory or dose interactions.",
                  "TCR output is a feature embedding, not a generated receptor sequence.",
                  "Causal claims require held-out real perturbation experiments.",
              ]}
    out.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _encode_row(model, arrays, row: int) -> np.ndarray:
    parts = []
    for modality in model["modalities"].astype(str):
        present = bool(arrays[f"mask__{modality}"][row])
        k = model[f"components__{modality}"].shape[0]
        if present:
            x = arrays[f"X__{modality}"][row].astype(np.float32)
            z = ((x - model[f"mean__{modality}"]) / model[f"scale__{modality}"]) @ model[f"components__{modality}"].T
        else:
            z = np.zeros(k, dtype=np.float32)
        parts.extend([z, np.asarray([float(present)], dtype=np.float32)])
    return np.concatenate(parts)


def predict_multimodal(model_path: Path, dataset: Path, cell_id: str,
                       perturbation: str, strength: float, time_hours: float,
                       out_dir: Path, condition: str | None = None) -> dict:
    """Predict every trained modality for one cell under a counterfactual."""
    validate_dataset(dataset)
    _, a = _load_dataset(dataset)
    m = np.load(model_path, allow_pickle=False)
    ids = a["obs__cell_id"].astype(str).tolist()
    if cell_id not in ids:
        raise ValueError(f"Unknown cell_id {cell_id!r}")
    perturbations = m["perturbations"].astype(str).tolist()
    if perturbation not in perturbations:
        raise ValueError(f"perturbation must be one of {perturbations}")
    row = ids.index(cell_id)
    cell_type = str(a["obs__cell_type"][row])
    condition = condition or str(a["obs__condition"][row])
    if cell_type not in m["cell_types"].astype(str).tolist():
        raise ValueError(f"Unseen cell_type {cell_type!r}")
    if condition not in m["conditions_vocab"].astype(str).tolist():
        raise ValueError(f"Unseen condition {condition!r}")
    design = [_encode_row(m, a, row)]
    design.extend(np.asarray([strength if perturbation == name else 0.0], dtype=np.float32)
                  for name in m["perturbations"].astype(str))
    design.extend(np.asarray([float(cell_type == name)], dtype=np.float32)
                  for name in m["cell_types"].astype(str))
    design.extend(np.asarray([float(condition == name)], dtype=np.float32)
                  for name in m["conditions_vocab"].astype(str))
    design.extend([np.asarray([time_hours / float(m["time_scale"])], dtype=np.float32),
                   np.ones(1, dtype=np.float32)])
    x = np.concatenate(design)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for modality in m["modalities"].astype(str):
        pred = x @ m[f"decoder__{modality}"]
        frame = pd.DataFrame({"feature": m[f"features__{modality}"].astype(str),
                              "predicted_value": pred})
        name = f"{modality}_predictions.csv"
        frame.to_csv(out_dir / name, index=False)
        files[modality] = name
    metadata = {"model_type": str(m["model_type"]), "input_cell_id": cell_id,
                "cell_type": cell_type, "condition": condition,
                "counterfactual": {"perturbation": perturbation,
                                   "strength": float(strength), "time_hours": float(time_hours)},
                "outputs": files, "data_origin": str(m["data_origin"]),
                "interpretation": "Research baseline; predictions are not clinical or experimentally validated."}
    (out_dir / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def generate_simulated_example(out_dir: Path, n_cells: int = 120, seed: int = 7) -> dict:
    """Generate a tiny, explicitly simulated paired multi-omic T-cell dataset."""
    if n_cells < 24:
        raise ValueError("n_cells must be at least 24")
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genes = np.asarray(["CD3D", "CD3E", "LCK", "ZAP70", "IL7R", "CCR7", "NKG7", "GZMB", "IFNG", "PDCD1", "CTLA4", "MKI67"])
    peaks = np.asarray([f"chr{i % 3 + 1}:{1000 + i * 200}-{1100 + i * 200}" for i in range(16)])
    proteins = np.asarray(["CD3", "CD4", "CD8", "CD25", "CD69", "PD-1"])
    tcr_features = np.asarray([f"tcr_3mer_hash_{i}" for i in range(10)])
    perturbations = np.asarray(["CTRL", "LCK", "ZAP70", "CTLA4"])
    cell_types = np.asarray(["CD4_T", "CD8_T", "Treg"])
    perturb = np.resize(perturbations, n_cells)
    cell_type = np.resize(cell_types, n_cells)
    time = np.resize(np.asarray([0.0, 8.0, 24.0, 48.0]), n_cells)
    strength = (perturb != "CTRL").astype(np.float32)
    latent = rng.normal(size=(n_cells, 5)).astype(np.float32)
    latent[:, 0] += (cell_type == "CD8_T") * 1.5
    latent[:, 1] += (cell_type == "Treg") * 1.5
    latent[:, 2] += (perturb == "LCK") * -1.2 + (perturb == "ZAP70") * -0.8
    latent[:, 3] += time / 48.0
    def simulate(width, positive=True):
        values = latent @ rng.normal(scale=.5, size=(5, width)) + rng.normal(scale=.15, size=(n_cells, width))
        return np.log1p(np.exp(values)).astype(np.float32) if positive else values.astype(np.float32)
    arrays = {
        "obs__cell_id": np.asarray([f"sim_cell_{i:04d}" for i in range(n_cells)]),
        "obs__donor_id": np.asarray([f"sim_donor_{i % 4}" for i in range(n_cells)]),
        "obs__batch": np.asarray([f"sim_batch_{i % 2}" for i in range(n_cells)]),
        "obs__cell_type": cell_type, "obs__condition": np.where(time > 0, "stimulated", "resting"),
        "obs__time_hours": time, "obs__perturbation": perturb,
        "obs__perturbation_strength": strength, "obs__is_control": perturb == "CTRL",
        "obs__split": np.where(np.arange(n_cells) % 5 == 0, "test", "train"),
        "X__rna": simulate(len(genes)), "var__rna": genes,
        "X__atac": simulate(len(peaks)), "var__atac": peaks,
        "X__protein": simulate(len(proteins)), "var__protein": proteins,
        "X__tcr": simulate(len(tcr_features), positive=False), "var__tcr": tcr_features,
    }
    for i, modality in enumerate(MODALITIES):
        mask = rng.random(n_cells) > (0.05 + 0.05 * i)
        mask[:2] = True
        arrays[f"mask__{modality}"] = mask
        arrays[f"X__{modality}"][~mask] = 0.0
    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_name": "simulated_multimodal_t_cells",
        "data_origin": "simulated", "paired_measurements": True,
        "modalities": {
            "rna": {"representation": "simulated log1p-like expression"},
            "atac": {"representation": "simulated accessibility values"},
            "protein": {"representation": "simulated CLR-like CITE-seq values"},
            "tcr": {"representation": "simulated hashed 3-mer feature embedding"},
        },
        "warning": "No row or value in this example comes from a patient, donor, assay, or public dataset.",
        "seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validate_dataset(out_dir)
