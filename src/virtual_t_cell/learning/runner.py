"""One resumable response-learning cycle over an immutable evidence journal.

This module does not poll, sequence cells, or execute experiments.  A caller can
invoke it after each successful ingestion pass.  Accepted proposals alone may
replace the active model; rejected evidence remains in the journal for review.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .belief import BeliefState
from .continual import propose_update
from .evidence import EvidenceStore
from .schema import RESPONSE_EVIDENCE, canonical_json, content_hash, validate_record
from .world_model import BayesianWorldModel


_RUNNER_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_records(path: Path) -> tuple[list[dict], str]:
    # Read once: the validation digest describes exactly the bytes parsed here.
    data = path.read_bytes()
    records = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path.name} at line {line_number}") from error
    return records, hashlib.sha256(data).hexdigest()


def _temporary_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = _temporary_file(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def _copy_to_temporary(source: Path, destination: Path) -> Path:
    temporary = _temporary_file(destination)
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _model_lock(model_path: Path):
    """Nonblocking process lock shared by cycles for this active model path.

    The persistent lock file is deliberately not unlinked: unlinking it could
    let different callers lock different files.  POSIX uses ``flock``; native
    Windows uses ``msvcrt.locking`` on byte zero.  Both are held until this
    context exits, and the OS releases the handle if its process terminates.
    Writers outside this runner must also honor this lock; a content-hash check
    immediately before replacement detects other changes during fitting.
    """
    lock_path = model_path.with_name(model_path.name + ".learning.lock")
    # No byte needs to be written: Windows permits locking beyond EOF.  Avoiding
    # writes also avoids touching a byte another process might already hold.
    # https://docs.python.org/3/library/msvcrt.html#msvcrt.locking
    with lock_path.open("a+b", buffering=0) as handle:
        if os.name == "nt":
            import msvcrt

            def operate(acquire: bool) -> None:
                # locking() starts at the current descriptor position; every
                # acquisition and release must refer to the same byte range.
                handle.seek(0)
                mode = msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK
                msvcrt.locking(handle.fileno(), mode, 1)
        else:
            try:
                import fcntl
            except ImportError as error:
                raise ImportError("model locking requires Windows msvcrt or POSIX fcntl") from error

            def operate(acquire: bool) -> None:
                mode = fcntl.LOCK_EX | fcntl.LOCK_NB if acquire else fcntl.LOCK_UN
                fcntl.flock(handle.fileno(), mode)

        try:
            operate(True)
        except OSError as error:
            # The Microsoft CRT reports a locking violation as EACCES.  POSIX
            # flock uses EAGAIN/EWOULDBLOCK (and may use EACCES).  Preserve other
            # I/O errors rather than misreporting them as ordinary contention.
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                raise BlockingIOError(errno.EWOULDBLOCK,
                                      "another learning cycle is using this active model",
                                      str(lock_path)) from error
            raise
        try:
            yield
        finally:
            operate(False)


def _eligible_records(records: list[dict], *, model_kind: str, origin: str):
    eligible, ignored = [], []
    for row in records:
        validate_record(row, require_outcomes=False)
        kind = row["evidence_type"]
        if kind == "model_prediction":
            reason = "model_predictions_are_not_observed_evidence"
        elif row.get("response_training_eligible") is False or row.get("state_snapshot") is True:
            reason = "state_snapshot_is_not_response_training_evidence"
        elif kind == "observational":
            reason = "observational_atlas_is_not_response_training_evidence"
        elif model_kind == "synthetic" and kind != "synthetic":
            reason = "experimental_evidence_cannot_update_a_synthetic_model"
        elif model_kind == "experimental" and kind not in RESPONSE_EVIDENCE:
            reason = "synthetic_evidence_cannot_update_an_experimental_model"
        else:
            reason = None
        if reason is None:
            validate_record(row)
            eligible.append(row)
        else:
            ignored.append({"observation_id": row["observation_id"],
                            "evidence_type": kind, "reason": reason, "origin": origin})
    return eligible, ignored


def _save_candidate(model: BayesianWorldModel, path: Path) -> None:
    temporary = _temporary_file(path)
    try:
        model.save(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _PromotionConflict(ValueError):
    pass


def _promote_candidate(candidate: BayesianWorldModel, *, model_path: Path,
                       candidate_path: Path, expected_sha256: str, current_model_id: str,
                       output_dir: Path) -> dict:
    """Prepare both files, archive the incumbent, then replace with rollback.

    Model and belief files cannot be one filesystem rename.  Readers should
    compare their model IDs.  Both files are prepared first, and a sidecar write
    failure rolls the model back rather than leaving a stale belief on success.
    """
    if _file_hash(model_path) != expected_sha256:
        raise _PromotionConflict("current model changed during the learning cycle")
    belief_path = Path(str(model_path) + ".belief.json")
    belief = BeliefState.from_model(candidate).to_dict()
    encoded_belief = (canonical_json(belief) + "\n").encode("utf-8")
    previous = output_dir / "versions" / f"{current_model_id}.npz"
    previous.parent.mkdir(parents=True, exist_ok=True)
    old_belief = belief_path.read_bytes() if belief_path.is_file() else None
    previous_belief = Path(str(previous) + ".belief.json")
    backup = new_model = new_belief = None
    replaced_model = False
    try:
        backup = _copy_to_temporary(model_path, previous)
        new_model = _copy_to_temporary(candidate_path, model_path)
        new_belief = _temporary_file(belief_path)
        candidate_sha256 = _file_hash(new_model)
        if _file_hash(backup) != expected_sha256:
            raise _PromotionConflict("current model changed while its previous version was archived")
        with new_belief.open("wb") as handle:
            handle.write(encoded_belief)
            handle.flush()
            os.fsync(handle.fileno())
        if previous.exists():
            archived = BayesianWorldModel.load(previous)
            if archived.metadata["model_id"] != current_model_id:
                raise ValueError("previous-version archive conflicts with the current model identity")
        else:
            # Keep the temporary backup too, for an exact-byte rollback below.
            _atomic_bytes(previous, backup.read_bytes())
        if old_belief is not None and not previous_belief.exists():
            _atomic_bytes(previous_belief, old_belief)
        if _file_hash(model_path) != expected_sha256:
            raise _PromotionConflict("current model changed before atomic replacement")
        os.replace(new_model, model_path)
        replaced_model = True
        os.replace(new_belief, belief_path)
    except BaseException:
        if replaced_model:
            if _file_hash(model_path) != candidate_sha256:
                raise _PromotionConflict(
                    "another writer changed the promoted model; automatic rollback was not safe"
                )
            os.replace(backup, model_path)
            if old_belief is not None:
                _atomic_bytes(belief_path, old_belief)
            elif belief_path.is_file():
                belief_path.unlink()
        raise
    finally:
        for temporary in (backup, new_model, new_belief):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"previous_model_path": str(previous),
            "previous_belief_path": str(previous_belief) if old_belief is not None else None,
            "belief_path": str(belief_path), "model_sha256_after": candidate_sha256}


def _finish_promotion(report: dict, gate: dict, *, model_path: Path, candidate_path: Path,
                      expected_sha256: str, output_dir: Path, promote: bool) -> dict:
    report["accepted"] = bool(gate.get("accepted"))
    report["gate"] = {key: gate[key] for key in ("accepted", "decision", "reasons", "error") if key in gate}
    report["promoted"] = False
    report["promotion_requested"] = promote
    report["model_sha256_before"] = expected_sha256
    if not report["accepted"]:
        report["status"] = "rejected"
    elif not promote:
        report["status"] = "accepted_not_promoted"
    else:
        candidate = BayesianWorldModel.load(candidate_path)
        if candidate.metadata["model_id"] != report["candidate_model_id"]:
            raise ValueError("cached candidate does not match the attempt report")
        try:
            saved = _promote_candidate(
                candidate, model_path=model_path, candidate_path=candidate_path,
                expected_sha256=expected_sha256, current_model_id=report["current_model_id"],
                output_dir=output_dir,
            )
        except (OSError, ValueError) as error:
            report.update(status="rejected", rejection_stage="promotion", promotion_error=str(error))
            try:
                report["active_model_id"] = BayesianWorldModel.load(model_path).metadata["model_id"]
            except (OSError, ValueError):
                report["active_model_id"] = None
        else:
            report.update(saved)
            report.update(status="promoted", promoted=True, active_model_id=report["candidate_model_id"])
            report.pop("rejection_stage", None)
            report.pop("promotion_error", None)
    report["updated_at"] = _now()
    _write_json(Path(report["attempt_report_path"]), report)
    return report


def run_learning_cycle(*, model_path, evidence_path, validation_path, output_dir,
                       replay_path=None, allow_synthetic=False,
                       max_relative_regression=0.05, promote=True) -> dict:
    """Run one journal-snapshot update, with full replay and cached gate attempts.

    ``replay_path`` can seed initial training records into the journal.  Future
    calls reconstruct replay from the current model's immutable training IDs and
    use every compatible pending response observation.  Atlas snapshots and
    predictions are excluded and explicitly reported.  A synthetic model needs
    ``allow_synthetic=True`` and receives synthetic records only; a real model
    receives experimental/experimental-summary records only.

    Status is ``no_new_evidence``, ``promoted``, ``rejected``,
    ``already_attempted``, or ``accepted_not_promoted``.  A rejected candidate
    and its gate remain on disk.  Setting ``promote=True`` on a later call can
    promote an already accepted candidate without fitting it again.
    """
    if not isinstance(allow_synthetic, bool) or not isinstance(promote, bool):
        raise ValueError("allow_synthetic and promote must be booleans")
    if (isinstance(max_relative_regression, bool)
            or not isinstance(max_relative_regression, (int, float))
            or not math.isfinite(max_relative_regression) or max_relative_regression < 0):
        raise ValueError("max_relative_regression must be finite and nonnegative")
    model_path, evidence_path, validation_path, output_dir = (
        Path(value).expanduser().resolve()
        for value in (model_path, evidence_path, validation_path, output_dir)
    )
    if not model_path.is_file():
        raise ValueError("model_path must identify an existing fitted model")
    if model_path in {evidence_path, validation_path} or evidence_path == validation_path:
        raise ValueError("model, evidence journal and validation paths must be different")
    output_dir.mkdir(parents=True, exist_ok=True)
    with _model_lock(model_path):
        original_sha256 = _file_hash(model_path)
        current = BayesianWorldModel.load(model_path)
        if _file_hash(model_path) != original_sha256:
            raise _PromotionConflict("current model changed while it was loaded")
        metadata = current.metadata
        kinds = set(metadata["evidence_types"])
        if kinds == {"synthetic"}:
            model_kind = "synthetic"
        elif kinds and kinds <= RESPONSE_EVIDENCE:
            model_kind = "experimental"
        else:
            raise ValueError("current model has an unsupported mixture of evidence types")
        seed_result, seed_ignored = None, []
        with EvidenceStore(evidence_path) as journal:
            if replay_path is not None:
                seed_records, _ = _read_records(Path(replay_path).expanduser().resolve())
                seed_records, seed_ignored = _eligible_records(
                    seed_records, model_kind=model_kind, origin="replay_seed"
                )
                seed_result = journal.append(seed_records)
            # One read transaction gives records and summary the same snapshot,
            # while WAL ingestion writers can continue appending the next batch.
            journal.connection.execute("BEGIN")
            all_records = journal.records()
            journal_summary = journal.summary()
            journal.connection.commit()
        records, ignored = _eligible_records(all_records, model_kind=model_kind, origin="journal")
        ignored = seed_ignored + ignored
        available = {row["observation_id"]: row for row in records}
        training_ids = set(metadata["training_ids"])
        replay = [available[identity] for identity in sorted(training_ids) if identity in available]
        new = [row for row in records if row["observation_id"] not in training_ids]
        missing = sorted(training_ids - set(available))
        fingerprints = metadata.get("training_record_hashes", {})
        changed = sorted(row["observation_id"] for row in replay
                         if content_hash(row) != fingerprints.get(row["observation_id"]))
        base = {
            "runner_version": _RUNNER_VERSION, "current_model_id": metadata["model_id"],
            "candidate_model_id": None, "active_model_id": metadata["model_id"],
            "model_path": str(model_path), "model_sha256_before": original_sha256,
            "new_evidence_count": len(new), "replay_count": len(replay),
            "journal_snapshot_hash": journal_summary["snapshot_hash"],
            "ignored_evidence": {"count": len(ignored),
                                 "by_reason": dict(sorted(Counter(row["reason"] for row in ignored).items())),
                                 "records": ignored},
            "replay_seed": seed_result, "promoted": False,
            "promotion_requested": promote,
            "belief_path": str(model_path) + ".belief.json" if Path(str(model_path) + ".belief.json").is_file() else None,
            "attempt_report_path": None, "gate_report_path": None, "candidate_model_path": None,
            "limitations": [
                "This cycle consumes completed data; it does not sequence or execute experiments.",
                "Observational atlas updates do not train intervention-response heads.",
                "Repeated validation reuse can overfit the gate; keep an untouched final test set.",
                "Concurrent writers must honor the model lock; external changes are checked before replacement.",
            ],
        }
        preflight = []
        if model_kind == "synthetic" and not allow_synthetic:
            preflight.append("synthetic_model_requires_allow_synthetic")
        if missing:
            preflight.append("missing_replay_training_records")
        if changed:
            preflight.append("replay_content_differs_from_incorporated_evidence")
        if not new and not preflight:
            return {**base, "status": "no_new_evidence"}
        validation, validation_sha256 = _read_records(validation_path)
        parameters = {"allow_synthetic": allow_synthetic,
                      "max_relative_regression": float(max_relative_regression),
                      "fit_config": metadata.get("fit_config", {})}
        candidate_hashes = {row["observation_id"]: content_hash(row) for row in replay + new}
        attempt_identity = {
            "runner_version": _RUNNER_VERSION, "model_path": str(model_path),
            "current_model_id": metadata["model_id"],
            "candidate_evidence_hashes": candidate_hashes,
            "validation_hash": content_hash(validation), "parameters": parameters,
        }
        attempt_key = content_hash(attempt_identity)
        attempt_dir = output_dir / "attempts" / attempt_key
        report_path = attempt_dir / "attempt_report.json"
        gate_path = attempt_dir / "gate_report.json"
        candidate_path = attempt_dir / "candidate_model.npz"
        if report_path.is_file():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("attempt_key") != attempt_key:
                raise ValueError("cached attempt key does not match its content address")
            if previous.get("accepted") and promote and (
                previous.get("status") == "accepted_not_promoted"
                or previous.get("rejection_stage") == "promotion"
            ):
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                previous["reused_candidate_without_training"] = True
                previous["journal_snapshot_hash"] = journal_summary["snapshot_hash"]
                return _finish_promotion(previous, gate, model_path=model_path,
                                         candidate_path=candidate_path, expected_sha256=original_sha256,
                                         output_dir=output_dir, promote=promote)
            repeated = deepcopy(previous)
            repeated.update(status="already_attempted", previous_status=previous["status"],
                            promoted=False, active_model_id=metadata["model_id"],
                            journal_snapshot_hash=journal_summary["snapshot_hash"],
                            ignored_evidence=base["ignored_evidence"],
                            promotion_requested=promote, reused_candidate_without_training=True)
            return repeated
        report = {
            **base, "attempt_key": attempt_key, "created_at": _now(),
            "attempt_report_path": str(report_path), "gate_report_path": str(gate_path),
            "validation_path": str(validation_path), "validation_sha256": validation_sha256,
            "validation_hash": attempt_identity["validation_hash"],
            "candidate_evidence_hashes": candidate_hashes, "parameters": parameters,
            "missing_replay_ids": missing, "changed_replay_ids": changed,
            "reused_candidate_without_training": False,
        }
        candidate = None
        if preflight:
            gate = {"accepted": False, "decision": "reject", "reasons": preflight,
                    "current_model_id": metadata["model_id"], "candidate_model_id": None,
                    "missing_replay_ids": missing, "changed_replay_ids": changed,
                    "promoted": False}
        else:
            try:
                candidate, gate = propose_update(
                    current, replay, new, validation,
                    max_relative_regression=max_relative_regression,
                    allow_synthetic=allow_synthetic,
                )
            except (ValueError, KeyError) as error:
                gate = {"accepted": False, "decision": "reject",
                        "reasons": ["invalid_evidence_or_validation"], "error": str(error),
                        "error_type": type(error).__name__,
                        "current_model_id": metadata["model_id"], "candidate_model_id": None,
                        "promoted": False}
        if candidate is not None:
            _save_candidate(candidate, candidate_path)
            report["candidate_model_id"] = candidate.metadata["model_id"]
            report["candidate_model_path"] = str(candidate_path)
            report["candidate_model_sha256"] = _file_hash(candidate_path)
        _write_json(gate_path, gate)
        return _finish_promotion(report, gate, model_path=model_path,
                                 candidate_path=candidate_path, expected_sha256=original_sha256,
                                 output_dir=output_dir, promote=promote)
