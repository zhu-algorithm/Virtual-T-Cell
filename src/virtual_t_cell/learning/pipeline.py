"""Resumable local batch ingestion. Sequencing and lab instruments are external."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .atlas import AtlasStore
from .evidence import EvidenceStore
from .schema import content_hash, read_jsonl


def _input_path(manifest: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("input paths must be nonempty strings")
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_manifest(path: str | Path, *, atlas_path: str | Path, evidence_path: str | Path) -> dict:
    """A completed, immutable JSON manifest is the handoff from a sequencing pipeline.

    Individual stores commit idempotently; retrying a partial cross-store failure
    is safe. This is deliberately not a distributed transaction or a sequencer.
    """
    path = Path(path).resolve()
    data = path.read_bytes()
    manifest_digest = hashlib.sha256(data).hexdigest()
    manifest = json.loads(data)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if not isinstance(manifest.get("batch_id"), str) or not manifest["batch_id"].strip():
        raise ValueError("manifest needs a batch_id")
    if not isinstance(manifest.get("source"), str) or not manifest["source"].strip():
        raise ValueError("manifest needs a source")
    if not manifest.get("atlas") and not manifest.get("responses_jsonl"):
        raise ValueError("manifest must reference atlas input or response observations")
    input_paths = []
    if manifest.get("responses_jsonl"):
        input_paths.append(_input_path(path, manifest["responses_jsonl"]))
    if manifest.get("atlas"):
        atlas_input = manifest["atlas"]
        if not isinstance(atlas_input, dict):
            raise ValueError("atlas input must be an object")
        keys = ("h5ad",) if "h5ad" in atlas_input else ("counts_csv", "metadata_csv")
        input_paths.extend(_input_path(path, atlas_input[key]) for key in keys)
    input_hashes = {str(item.resolve()): _file_hash(item) for item in input_paths}
    atlas_identity = None
    if manifest.get("atlas"):
        with AtlasStore(atlas_path) as atlas:
            atlas_identity = atlas.store_identity()
    digest = content_hash({"manifest_sha256": manifest_digest, "input_sha256": input_hashes})
    # Namespace the ledger by the persistent destination store, not only its
    # filename. A newly created/rebuilt atlas must ingest the original batches.
    journal_key = path.with_name(path.name + ".store_" + (atlas_identity or "responses_only"))
    with EvidenceStore(evidence_path) as journal:
        prior = journal.batch_status(journal_key, digest)
        if prior and prior["status"] == "complete":
            return {"status": "already_complete", "batch_id": manifest["batch_id"], "previous": prior["report"]}
        report = {"batch_id": manifest["batch_id"], "manifest_sha256": manifest_digest,
                  "input_sha256": input_hashes, "content_sha256": digest, "status": "complete",
                  "atlas_store_identity": atlas_identity}
        try:
            response_records = None
            if manifest.get("responses_jsonl"):
                response_records = read_jsonl(_input_path(path, manifest["responses_jsonl"]))
                if any(not isinstance(row, dict) for row in response_records):
                    raise ValueError("each response JSONL row must be a JSON object")
                if any(row.get("source") != manifest["source"] for row in response_records):
                    raise ValueError("response source must match manifest provenance")
            if manifest.get("atlas"):
                atlas_input = manifest["atlas"]
                if not isinstance(atlas_input, dict):
                    raise ValueError("atlas input must be an object")
                arguments = {"batch_id": manifest["batch_id"], "source": manifest["source"],
                             "evidence_type": manifest.get("atlas_evidence_type", "observational")}
                for key in ("min_counts", "min_genes", "max_mito_fraction"):
                    if key in atlas_input:
                        arguments[key] = atlas_input[key]
                atlas = AtlasStore(atlas_path)
                try:
                    if "h5ad" in atlas_input:
                        report["atlas"] = atlas.ingest_h5ad(_input_path(path, atlas_input["h5ad"]),
                                                           counts_layer=atlas_input.get("counts_layer"), **arguments)
                    else:
                        report["atlas"] = atlas.ingest_counts(
                            _input_path(path, atlas_input["counts_csv"]),
                            _input_path(path, atlas_input["metadata_csv"]), **arguments)
                finally:
                    atlas.close()
            if response_records is not None:
                report["responses"] = journal.append(response_records)
            report["learning"] = "Atlas states updated; response model promotion requires an independent validation gate."
            journal.record_batch(journal_key, digest, "complete", report)
            return report
        except Exception as error:
            journal.record_batch(journal_key, digest, "error", {"error": str(error), "batch_id": manifest["batch_id"]})
            raise


def sync_inbox(inbox: str | Path, *, atlas_path: str | Path, evidence_path: str | Path) -> dict:
    """Process *.json manifests once. Publish manifests last via atomic rename."""
    inbox = Path(inbox)
    if not inbox.is_dir():
        raise ValueError("inbox must be an existing directory")
    reports = []
    errors = []
    for path in sorted(inbox.glob("*.json")):
        try:
            reports.append(ingest_manifest(path, atlas_path=atlas_path, evidence_path=evidence_path))
        except (ValueError, KeyError, OSError, ImportError) as error:
            errors.append({"manifest": str(path), "error": str(error)})
    return {"processed": len(reports), "errors": errors, "reports": reports,
            "response_model_automatically_promoted": False}
