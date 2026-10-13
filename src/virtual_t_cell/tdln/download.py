"""Download the processed public GEO files required for TDLN calibration."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


FILES = {
    "GSE212797_TCR_117.csv.gz": "5b7082bbedbbf1bf44c3406d604a11aec50c0c378c04f050324736842474c339",
    "GSE212797_TCR_118.csv.gz": "5f60fbfb8007f391c6a49386d5283fad410cf1b7615bb6a10585a081d8ba4b3f",
    "GSE239514_TPM_expression_matrix.tsv.gz": "1400ae5270955fc87362dfc354ba4970c154aca6f1b7bb013d585e64181d04cb",
    "GSE316782_Metadata.csv.gz": "0eefc07b85782ea967afc3edce1ba1782800515869e4a62ce2129b5c74e2fce1",
    "GSE316782_Protein_counts.tsv.gz": "8e40c53992aa863ca88d5adf2e29a9819149ff3ff68082dd783c2d855a9f2dd6",
    "GSE316782_feature_reference.csv.gz": "ccdb724a4b57987466fc565a94f2dc33979bcde444aeeb68288aed24d4ca8616",
}


def _url(name: str) -> str:
    accession = name.split("_", 1)[0]
    family = accession[:-3] + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{family}/{accession}/suppl/{name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def download_tdln(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        target = out_dir / name
        if not target.exists() or _sha256(target) != expected:
            partial = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(_url(name), partial)
            if _sha256(partial) != expected:
                partial.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {name}")
            partial.replace(target)
        records.append({"file": name, "bytes": target.stat().st_size,
                        "sha256": expected, "url": _url(name)})
    manifest = {"source": "NCBI GEO processed supplementary files", "files": records}
    (out_dir / "tdln_download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
