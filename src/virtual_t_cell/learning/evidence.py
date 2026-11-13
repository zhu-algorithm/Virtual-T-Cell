"""Append-only evidence journal with transactional, provenance-aware deduplication."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import canonical_json, content_hash, evidence_identity, validate_record


class EvidenceStore:
    """A local journal; entries are never silently edited or used as model labels."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS evidence (
            observation_id TEXT PRIMARY KEY,
            source_identity TEXT NOT NULL UNIQUE,
            record_hash TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            source TEXT NOT NULL,
            payload TEXT NOT NULL,
            inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS batches (
            manifest_path TEXT PRIMARY KEY,
            manifest_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            report TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def append(self, records: list[dict]) -> dict:
        """Validate the whole batch and append atomically; exact repeats are no-ops."""
        for record in records:
            validate_record(record)
            if record["evidence_type"] == "model_prediction":
                raise ValueError("model predictions cannot be appended as new observed evidence")
        inserted = 0
        repeated = 0
        aliases = []
        with self.connection:
            for record in records:
                payload = canonical_json(record)
                digest = content_hash(record)
                existing = self.connection.execute(
                    "SELECT record_hash FROM evidence WHERE observation_id = ?",
                    (record["observation_id"],),
                ).fetchone()
                if existing:
                    if existing[0] != digest:
                        raise ValueError(f"immutable observation has conflicting content: {record['observation_id']}")
                    repeated += 1
                    continue
                identity = evidence_identity(record)
                original = self.connection.execute(
                    "SELECT observation_id, payload FROM evidence WHERE source_identity = ?", (identity,)
                ).fetchone()
                if original:
                    previous = json.loads(original[1])
                    previous.pop("observation_id")
                    comparison = {key: value for key, value in record.items() if key != "observation_id"}
                    if previous != comparison:
                        raise ValueError("same source evidence has conflicting data; provide a reconciled, versioned record")
                    repeated += 1
                    aliases.append({"ignored_id": record["observation_id"], "original_id": original[0]})
                    continue
                self.connection.execute(
                    "INSERT INTO evidence (observation_id, source_identity, record_hash, evidence_type, source, payload) VALUES (?, ?, ?, ?, ?, ?)",
                    (record["observation_id"], identity, digest, record["evidence_type"], record["source"], payload),
                )
                inserted += 1
        return {"inserted": inserted, "repeated": repeated, "aliases": aliases, "total": self.summary()["total"]}

    def records(self, *, evidence_type: str | None = None) -> list[dict]:
        if evidence_type is None:
            rows = self.connection.execute("SELECT payload FROM evidence ORDER BY observation_id")
        else:
            rows = self.connection.execute(
                "SELECT payload FROM evidence WHERE evidence_type = ? ORDER BY observation_id", (evidence_type,)
            )
        return [json.loads(row[0]) for row in rows]

    def summary(self) -> dict:
        counts = dict(self.connection.execute("SELECT evidence_type, COUNT(*) FROM evidence GROUP BY evidence_type"))
        sources = [row[0] for row in self.connection.execute("SELECT DISTINCT source FROM evidence ORDER BY source")]
        hashes = list(self.connection.execute("SELECT observation_id, record_hash FROM evidence ORDER BY observation_id"))
        return {"total": sum(counts.values()), "evidence_counts": counts, "sources": sources,
                "snapshot_hash": content_hash(hashes)}

    def batch_status(self, path: Path, digest: str) -> dict | None:
        row = self.connection.execute("SELECT manifest_hash, status, report FROM batches WHERE manifest_path = ?",
                                      (str(path.resolve()),)).fetchone()
        if row is None:
            return None
        if row[0] != digest:
            raise ValueError("an already recorded manifest was changed; use a new versioned filename")
        return {"status": row[1], "report": json.loads(row[2]) if row[2] else None}

    def record_batch(self, path: Path, digest: str, status: str, report: dict) -> None:
        if status not in {"complete", "error"}:
            raise ValueError("invalid batch status")
        self.batch_status(path, digest)
        with self.connection:
            self.connection.execute("""INSERT INTO batches (manifest_path, manifest_hash, status, report)
                VALUES (?, ?, ?, ?) ON CONFLICT(manifest_path) DO UPDATE SET
                status = excluded.status, report = excluded.report, updated_at = CURRENT_TIMESTAMP""",
                (str(path.resolve()), digest, status, canonical_json(report)))
