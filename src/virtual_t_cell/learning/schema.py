"""Explicit observation, unit, provenance and group contracts for research learning."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


EVIDENCE_TYPES = {"experimental", "experimental_summary", "observational", "synthetic", "model_prediction"}
RESPONSE_EVIDENCE = {"experimental", "experimental_summary"}
REQUIRED_TEXT = ("observation_id", "experiment_id", "group_id", "organ", "cell_type", "condition", "intervention", "source")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evidence_identity(record: dict) -> str:
    """Stable source assay identity, independent of renamed observation/group IDs.

    A source's original record ID is preferred. Without it, provenance must keep
    stable experiment/sample/context identifiers. Cross-source aliases still
    require upstream reconciliation; no algorithm can infer missing provenance.
    """
    if record.get("source_record_id"):
        identity = {"source": record["source"], "source_record_id": record["source_record_id"]}
    else:
        names = ("source", "experiment_id", "sample_id", "donor_id", "organ", "species", "cell_type",
                 "condition", "intervention", "biological_time_hours", "dose", "dose_unit", "assay")
        identity = {name: record.get(name) for name in names}
    return content_hash(identity)


def _numeric_mapping(value: object, name: str, allow_empty: bool = False) -> dict:
    if not isinstance(value, dict) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else 'nonempty '}mapping")
    for key, number in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be nonempty strings")
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
            raise ValueError(f"{name}.{key} must be a finite number")
    return value


def validate_record(record: dict, require_outcomes: bool = True) -> dict:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    for name in REQUIRED_TEXT:
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ValueError(f"{name} is required and must be a nonempty string")
    if record.get("evidence_type") not in EVIDENCE_TYPES:
        raise ValueError("evidence_type must explicitly identify the observation's origin")
    for optional_text in ("source_record_id", "species"):
        if optional_text in record and (not isinstance(record[optional_text], str) or not record[optional_text].strip()):
            raise ValueError(f"{optional_text} must be a nonempty string when provided")
    _numeric_mapping(record.get("features"), "features")
    outcomes = _numeric_mapping(record.get("outcomes", {}), "outcomes", allow_empty=not require_outcomes)
    units = record.get("outcome_units", {})
    if not isinstance(units, dict) or set(units) != set(outcomes):
        raise ValueError("outcome_units must name exactly the observed outcomes")
    if any(not isinstance(unit, str) or not unit.strip() for unit in units.values()):
        raise ValueError("all observed outcomes require explicit units")
    noise = _numeric_mapping(record.get("measurement_sd", {}), "measurement_sd", allow_empty=True)
    if not set(noise).issubset(outcomes) or any(value < 0 for value in noise.values()):
        raise ValueError("measurement_sd must be nonnegative and refer to observed outcomes")
    for name in ("biological_time_hours", "dose"):
        value = record.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be null or a finite nonnegative number")
    if record.get("dose") is not None and not record.get("dose_unit"):
        raise ValueError("a dose requires dose_unit")
    return record


def validate_dataset(records: list[dict], allow_synthetic: bool = False) -> list[dict]:
    if not records:
        raise ValueError("dataset is empty")
    validate_record(records[0])
    ids: set[str] = set()
    source_ids: set[str] = set()
    features = set(records[0].get("features", {}))
    units: dict[str, str] = {}
    kinds: set[str] = set()
    for record in records:
        validate_record(record)
        identity = record["observation_id"]
        if identity in ids:
            raise ValueError(f"duplicate observation_id: {identity}")
        ids.add(identity)
        if record.get("response_training_eligible") is False or record.get("state_snapshot") is True:
            raise ValueError("atlas state snapshots cannot be used as experimental response evidence")
        source_identity = evidence_identity(record)
        if source_identity in source_ids:
            raise ValueError("duplicate original source evidence under different observation IDs")
        source_ids.add(source_identity)
        if set(record["features"]) != features:
            raise ValueError("all rows must have the same explicit numeric feature names")
        kind = record["evidence_type"]
        kinds.add(kind)
        if kind not in RESPONSE_EVIDENCE and not (allow_synthetic and kind == "synthetic"):
            raise ValueError(f"{kind} cannot be used as experimental response evidence")
        for outcome, unit in record["outcome_units"].items():
            if outcome in units and units[outcome] != unit:
                raise ValueError(f"inconsistent units for {outcome}: {units[outcome]} versus {unit}")
            units[outcome] = unit
    if "synthetic" in kinds and len(kinds) > 1:
        raise ValueError("synthetic and experimental evidence must remain separate datasets")
    return records


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on line {line_number}") from error
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
