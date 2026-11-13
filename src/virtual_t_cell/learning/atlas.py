"""Append-only single-cell state atlas with explicit observational provenance.

CSV counts are streamed; metadata and pseudobulk accumulators are held in memory.
This is a research-scale ingestion path, not a distributed atlas trainer.  A new
sequencing batch contains independently sampled cells, not a tracked cell movie.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

import numpy as np

from .schema import canonical_json, content_hash


REQUIRED_METADATA = ("cell_id", "sample_id", "donor_id", "organ", "cell_type", "condition")
_ATLAS_EVIDENCE = {"observational", "synthetic"}
_ORGAN_ALIASES = {"肝": "liver", "肝脏": "liver", "脾": "spleen", "脾脏": "spleen"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _metadata_record(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        if key is None:
            raise ValueError("metadata row has more values than its header")
        if value is not None and str(value).strip() and str(value).strip().lower() not in {"nan", "none"}:
            result[str(key)] = str(value).strip()
    for name in REQUIRED_METADATA:
        result[name] = _text(result.get(name), name)
    organ = result["organ"]
    result["organ"] = _ORGAN_ALIASES.get(organ, organ.lower())
    for name in ("biological_time_hours", "dose"):
        if name in result:
            try:
                value = float(result[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be numeric when supplied") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            result[name] = value
    if "dose" in result and "dose_unit" not in result:
        raise ValueError("a supplied dose requires dose_unit")
    return result


def _load_metadata(path: str | Path) -> dict[str, dict]:
    records = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not set(REQUIRED_METADATA).issubset(header):
            raise ValueError(f"metadata requires unique columns: {', '.join(REQUIRED_METADATA)}")
        for row in reader:
            record = _metadata_record(row)
            cell = record["cell_id"]
            if cell in records:
                raise ValueError(f"duplicate cell_id in metadata: {cell}; use unique row IDs within a batch")
            records[cell] = record
    if not records:
        raise ValueError("metadata is empty")
    return records


def _qc_policy(min_counts: int, min_genes: int, max_mito_fraction: float) -> dict:
    for name, value in (("min_counts", min_counts), ("min_genes", min_genes)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(max_mito_fraction, bool) or not math.isfinite(max_mito_fraction) or not 0 <= max_mito_fraction <= 1:
        raise ValueError("max_mito_fraction must be between 0 and 1")
    return {"min_counts": int(min_counts), "min_genes": int(min_genes),
            "max_mito_fraction": float(max_mito_fraction), "mitochondrial_prefix": "MT- (case insensitive)"}


class AtlasStore:
    """Persistent state observations; response-model training is a separate API.

    Cell identity is (stable source namespace, sample_id, cell_id). A repeated
    barcode in a different sample is allowed; changed values for the same cell
    identity are rejected. Keep source/sample IDs stable across reruns.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS atlas_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS atlas_panels (panel_id TEXT PRIMARY KEY, genes_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS atlas_batches (
                    batch_key TEXT PRIMARY KEY, source TEXT NOT NULL, batch_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL, report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS atlas_cells (
                    cell_key TEXT PRIMARY KEY, source TEXT NOT NULL, sample_id TEXT NOT NULL,
                    cell_id TEXT NOT NULL, payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL,
                    passed_qc INTEGER NOT NULL, qc_json TEXT NOT NULL, panel_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS atlas_cell_batches (
                    cell_key TEXT NOT NULL, batch_key TEXT NOT NULL,
                    PRIMARY KEY (cell_key, batch_key));
            """)
            row = connection.execute("SELECT value FROM atlas_meta WHERE key = 'schema_version'").fetchone()
            if row is not None and row[0] != "1":
                raise ValueError("unsupported atlas database schema")
            connection.execute("INSERT OR IGNORE INTO atlas_meta VALUES ('schema_version', '1')")
            connection.execute("INSERT OR IGNORE INTO atlas_meta VALUES ('store_uuid', ?)", (str(uuid.uuid4()),))

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def close(self) -> None:
        """No persistent handles are held; included for explicit caller cleanup."""

    def store_identity(self) -> str:
        """Stable database UUID for ingestion-ledger binding, independent of path.

        A database reopened or moved retains this identity; constructing a fresh
        database, including at a previously used path, creates a different UUID.
        """
        with self._connection() as connection:
            row = connection.execute("SELECT value FROM atlas_meta WHERE key = 'store_uuid'").fetchone()
        if row is None:
            raise ValueError("atlas database is missing its persistent identity")
        try:
            return str(uuid.UUID(row[0]))
        except (ValueError, AttributeError) as error:
            raise ValueError("atlas database has an invalid persistent identity") from error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def ingest_counts(self, counts_path: str | Path, metadata_path: str | Path, *,
                      batch_id: str, source: str, evidence_type: str = "observational",
                      min_counts: int = 1, min_genes: int = 1,
                      max_mito_fraction: float = 1.0) -> dict:
        """Validate and atomically append a wide *raw-count* CSV and its metadata.

        Zero-library and threshold-failing cells are retained for audit but are
        excluded from pseudobulk states. These basic count filters do not detect
        doublets, ambient RNA, annotation errors, or experimental confounding.
        """
        metadata = _load_metadata(metadata_path)
        provenance = {"format": "raw_counts_csv", "counts_sha256": file_sha256(counts_path),
                      "metadata_sha256": file_sha256(metadata_path),
                      "raw_count_assertion": "caller supplied raw counts; finite nonnegative integers checked"}
        with Path(counts_path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if len(header) < 2 or header[0] != "cell_id":
                raise ValueError("counts CSV must start with cell_id followed by gene columns")
            return self._ingest(header[1:], reader, metadata, batch_id=batch_id, source=source,
                                evidence_type=evidence_type, provenance=provenance,
                                qc_policy=_qc_policy(min_counts, min_genes, max_mito_fraction))

    def ingest_h5ad(self, path: str | Path, *, counts_layer: str, batch_id: str, source: str,
                    evidence_type: str = "observational", min_counts: int = 1, min_genes: int = 1,
                    max_mito_fraction: float = 1.0) -> dict:
        """Optional H5AD adapter; layer name or explicit 'X' must denote raw counts.

        Never implicitly reads log-normalized X. Integer validation cannot prove
        a matrix's biological units, so the explicit raw-count assertion is kept.
        AnnData backed files are read in row chunks, but a layer can still be
        materialized by AnnData; this is not a guarantee of bounded total memory.
        """
        counts_layer = _text(counts_layer, "counts_layer")
        try:
            import anndata as ad
        except ImportError as error:
            raise ImportError('H5AD input requires optional dependencies: pip install -e ".[atlas]"') from error
        data = ad.read_h5ad(path, backed="r")
        try:
            if counts_layer == "X":
                matrix = data.X
            elif counts_layer in data.layers:
                matrix = data.layers[counts_layer]
            else:
                raise ValueError(f"explicit raw-count layer not found: {counts_layer}")
            metadata, cell_order = {}, []
            for index, row in data.obs.iterrows():
                values = row.to_dict()
                values.setdefault("cell_id", str(index))
                record = _metadata_record(values)
                cell = record["cell_id"]
                if cell in metadata:
                    raise ValueError(f"duplicate cell_id in H5AD metadata: {cell}")
                metadata[cell] = record
                cell_order.append(cell)
            def rows():
                # Chunking limits temporary dense arrays, even for sparse input.
                chunk_size = max(1, min(256, 1_000_000 // max(1, data.n_vars)))
                for start in range(0, data.n_obs, chunk_size):
                    block = matrix[start:start + chunk_size, :]
                    if hasattr(block, "toarray"):
                        block = block.toarray()
                    block = np.asarray(block)
                    for offset, values in enumerate(block):
                        yield [cell_order[start + offset], *values.tolist()]
            provenance = {"format": "h5ad", "h5ad_sha256": file_sha256(path),
                          "counts_layer": counts_layer,
                          "raw_count_assertion": "caller designated layer as raw counts; finite nonnegative integers checked"}
            return self._ingest([str(gene) for gene in data.var_names], rows(), metadata,
                                batch_id=batch_id, source=source, evidence_type=evidence_type,
                                provenance=provenance, qc_policy=_qc_policy(min_counts, min_genes, max_mito_fraction))
        finally:
            data.file.close()

    def _ingest(self, genes, rows, metadata, *, batch_id, source, evidence_type, provenance, qc_policy):
        source, batch_id = _text(source, "source"), _text(batch_id, "batch_id")
        if evidence_type not in _ATLAS_EVIDENCE:
            raise ValueError("atlas evidence_type must be observational or synthetic; intervention labels use the response API")
        if not genes or any(not isinstance(gene, str) or not gene.strip() for gene in genes) or len(set(genes)) != len(genes):
            raise ValueError("gene columns must be nonempty and unique")
        if not metadata:
            raise ValueError("metadata is empty")
        panel = sorted(genes)
        panel_id = content_hash(panel)
        mitochondrial = [index for index, gene in enumerate(genes) if gene.upper().startswith("MT-")]
        payload = {"source": source, "batch_id": batch_id, "evidence_type": evidence_type,
                   "gene_panel_id": panel_id, "provenance": provenance, "qc_policy": qc_policy}
        batch_key, batch_hash = content_hash([source, batch_id]), content_hash(payload)
        added = duplicates = passed = failed = 0
        seen, organs, cell_types = set(), Counter(), Counter()
        with self._connection() as connection:
            previous = connection.execute("SELECT payload_hash, report_json FROM atlas_batches WHERE batch_key = ?", (batch_key,)).fetchone()
            if previous is not None:
                if previous[0] != batch_hash:
                    raise ValueError(f"batch_id already exists with different data or QC policy: {batch_id}")
                report = json.loads(previous[1])
                report.update(idempotent=True, added_cells=0, duplicate_cells=report["cells_total"])
                return report
            connection.execute("INSERT OR IGNORE INTO atlas_panels VALUES (?, ?)", (panel_id, canonical_json(panel)))
            for row_number, row in enumerate(rows, 2):
                if len(row) != len(genes) + 1:
                    raise ValueError(f"counts row {row_number} does not match the gene header")
                cell = _text(row[0], "cell_id")
                if cell in seen:
                    raise ValueError(f"duplicate cell_id in counts: {cell}")
                seen.add(cell)
                if cell not in metadata:
                    raise ValueError(f"counts cell is absent from metadata: {cell}")
                try:
                    values = np.asarray(row[1:], dtype=float)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"non-numeric raw counts for cell {cell}") from error
                if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)) or np.any(values > 2**53 - 1):
                    raise ValueError(f"raw counts must be finite nonnegative integers below 2**53: {cell}")
                count_values = [int(value) for value in values]
                total = sum(count_values)
                detected = int(np.count_nonzero(values))
                mito_fraction = sum(count_values[index] for index in mitochondrial) / total if total else 0.0
                reasons = []
                if total == 0:
                    reasons.append("zero_library")
                if total < qc_policy["min_counts"]:
                    reasons.append("low_counts")
                if detected < qc_policy["min_genes"]:
                    reasons.append("low_detected_genes")
                if mito_fraction > qc_policy["max_mito_fraction"]:
                    reasons.append("high_mitochondrial_fraction")
                qc = {"total_counts": total, "detected_genes": detected, "mitochondrial_fraction": mito_fraction,
                      "passed": not reasons, "reasons": reasons, "policy": qc_policy}
                meta = metadata[cell]
                if "batch_id" in meta and meta["batch_id"] != batch_id:
                    raise ValueError(f"metadata batch_id disagrees with ingestion batch for cell {cell}")
                # Batch membership is provenance, not biological identity.
                stable_meta = {name: value for name, value in meta.items() if name != "batch_id"}
                cell_payload = {"source": source, "evidence_type": evidence_type, "metadata": stable_meta,
                                "gene_panel_id": panel_id, "counts": {gene: count for gene, count in zip(genes, count_values) if count},
                                "qc_policy": qc_policy}
                cell_hash = content_hash(cell_payload)
                cell_key = content_hash([source, meta["sample_id"], cell])
                prior = connection.execute("SELECT payload_hash FROM atlas_cells WHERE cell_key = ?", (cell_key,)).fetchone()
                if prior is not None:
                    if prior[0] != cell_hash:
                        raise ValueError(f"existing cell identity has changed data, metadata or QC policy: {source}/{meta['sample_id']}/{cell}")
                    duplicates += 1
                else:
                    connection.execute("INSERT INTO atlas_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                       (cell_key, source, meta["sample_id"], cell, cell_hash, canonical_json(cell_payload),
                                        int(not reasons), canonical_json(qc), panel_id))
                    added += 1
                connection.execute("INSERT OR IGNORE INTO atlas_cell_batches VALUES (?, ?)", (cell_key, batch_key))
                passed += int(not reasons)
                failed += int(bool(reasons))
                organs[meta["organ"]] += 1
                cell_types[meta["cell_type"]] += 1
            if not seen:
                raise ValueError("counts matrix contains no cells")
            missing = set(metadata) - seen
            if missing:
                raise ValueError(f"metadata contains cells absent from counts: {sorted(missing)[:5]}")
            report = {"batch_id": batch_id, "source": source, "evidence_type": evidence_type,
                      "batch_hash": batch_hash, "gene_panel_id": panel_id, "genes": len(genes),
                      "cells_total": len(seen), "added_cells": added, "duplicate_cells": duplicates,
                      "qc_passed": passed, "qc_failed": failed, "organs": dict(organs), "cell_types": dict(cell_types),
                      "mitochondrial_genes_detected": len(mitochondrial), "qc_policy": qc_policy,
                      "provenance": provenance, "idempotent": False, "response_model_updated": False,
                      "limitations": ["State atlas only; no causal response labels are inferred.",
                                      "Cells are not longitudinally tracked by repeated destructive sequencing.",
                                      "Basic count QC does not perform doublet removal, ambient-RNA correction or cell-type annotation."]}
            connection.execute("INSERT INTO atlas_batches VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (batch_key, source, batch_id, batch_hash, canonical_json(payload), canonical_json(report),
                                datetime.now(timezone.utc).isoformat()))
        return report

    def pseudobulk_records(self) -> list[dict]:
        """Return current state snapshots; snapshots from successive calls overlap.

        Group by source, donor, sample, organ, cell type, condition, biological
        time, species, intervention/dose when supplied, gene panel and evidence
        type. Missing panel genes are never imputed as measured zero. Summation
        happens before CP10K/log1p conversion; the denominator is counts across
        the supplied panel, not an inferred whole-transcriptome library size.
        """
        groups = {}
        with self._connection() as connection:
            panels = {key: json.loads(value) for key, value in connection.execute("SELECT panel_id, genes_json FROM atlas_panels")}
            membership = {}
            for cell_key, batch_id in connection.execute("SELECT c.cell_key, b.batch_id FROM atlas_cell_batches c JOIN atlas_batches b USING (batch_key)"):
                membership.setdefault(cell_key, set()).add(batch_id)
            for cell_key, payload_hash, payload_json in connection.execute("SELECT cell_key, payload_hash, payload_json FROM atlas_cells WHERE passed_qc = 1 ORDER BY cell_key"):
                payload = json.loads(payload_json)
                meta = payload["metadata"]
                identity = {"source": payload["source"], "evidence_type": payload["evidence_type"],
                            "gene_panel_id": payload["gene_panel_id"],
                            **{name: meta.get(name) for name in ("sample_id", "donor_id", "organ", "cell_type", "condition", "biological_time_hours", "species",
                                                               "intervention", "dose", "dose_unit")}}
                key = content_hash(identity)
                group = groups.setdefault(key, {"identity": identity, "raw_counts": {}, "cells": [], "batch_ids": set()})
                for gene, count in payload["counts"].items():
                    group["raw_counts"][gene] = group["raw_counts"].get(gene, 0) + count
                group["cells"].append([cell_key, payload_hash])
                group["batch_ids"].update(membership.get(cell_key, set()))
        records = []
        for key, group in sorted(groups.items()):
            identity, counts = group["identity"], group["raw_counts"]
            total = sum(counts.values())
            genes = panels[identity["gene_panel_id"]]
            outcomes = {gene: math.log1p(counts.get(gene, 0) / total * 1e4) for gene in genes}
            state_hash = content_hash([identity, group["cells"]])
            source_record_id = "atlas_state:" + key
            record = {"observation_id": "atlas:" + state_hash, "source_record_id": source_record_id,
                      "experiment_id": content_hash([identity["source"], identity["sample_id"]]),
                      "group_id": content_hash([identity["source"], identity["donor_id"]]),
                      "source": identity["source"], "evidence_type": identity["evidence_type"],
                      "organ": identity["organ"], "cell_type": identity["cell_type"], "condition": identity["condition"],
                      "donor_id": identity["donor_id"], "sample_id": identity["sample_id"],
                      "intervention": identity["intervention"] if identity["intervention"] is not None else "observational_state",
                      "features": {"n_cells": float(len(group["cells"])), "library_size": float(total)},
                      "outcomes": outcomes, "outcome_units": {gene: "log1p_cp10k" for gene in genes},
                      "raw_counts": {gene: counts.get(gene, 0) for gene in genes}, "raw_count_unit": "counts",
                      "normalization": {"method": "log1p_cp10k", "scale_factor": 10000.0,
                                        "denominator": "summed raw counts across the supplied gene panel",
                                        "denominator_counts": total, "gene_panel_id": identity["gene_panel_id"],
                                        "whole_transcriptome_library_size_inferred": False},
                      "n_cells": len(group["cells"]), "n_samples": 1, "n_donors": 1,
                      "statistical_unit": "one donor/sample/context pseudobulk state; cells are not independent biological replicates",
                      "gene_panel_id": identity["gene_panel_id"], "batch_ids": sorted(group["batch_ids"]),
                      "state_snapshot": True, "snapshot_hash": state_hash,
                      "response_training_eligible": False,
                      "limitations": ["Overlapping atlas snapshots are not independent new experiments.",
                                      "Pseudobulk state observations are not perturbation-response labels."]}
            for name in ("biological_time_hours", "species", "dose", "dose_unit"):
                if identity[name] is not None:
                    record[name] = identity[name]
            records.append(record)
        return records

    def summary(self) -> dict:
        with self._connection() as connection:
            n_batches = connection.execute("SELECT COUNT(*) FROM atlas_batches").fetchone()[0]
            cells = connection.execute("SELECT payload_json, passed_qc FROM atlas_cells").fetchall()
        organs, cell_types, evidence = Counter(), Counter(), Counter()
        passed = 0
        for payload_json, passed_qc in cells:
            payload = json.loads(payload_json)
            organs[payload["metadata"]["organ"]] += 1
            cell_types[payload["metadata"]["cell_type"]] += 1
            evidence[payload["evidence_type"]] += 1
            passed += passed_qc
        records = self.pseudobulk_records()
        return {"schema_version": 1, "n_batches": n_batches, "n_cells": len(cells), "n_qc_passed": passed,
                "n_qc_failed": len(cells) - passed, "n_pseudobulk_groups": len(records),
                "organs": dict(organs), "cell_types": dict(cell_types), "evidence_counts": dict(evidence),
                "state_version": content_hash(sorted(record["snapshot_hash"] for record in records)),
                "response_model_updated": False,
                "limitations": ["Metadata and pseudobulk accumulators reside in memory; this is a research-scale SQLite store.",
                                "Sequencing batch order is not biological time or a trajectory of the same cell."]}
