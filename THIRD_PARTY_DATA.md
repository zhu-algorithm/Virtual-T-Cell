# Third-party TCR data

The software in this repository is MIT licensed. Third-party databases retain
their original licenses and attribution requirements.

The bundled `models/vdjdb_2026_06_tcr_evidence.npz` is a compact transformation
of the human records in VDJdb release 2026-06-03. VDJdb is distributed under
AGPL-3.0; cite the VDJdb publication and use the database in accordance with its
license. The compact file preserves source, release and evidence fields and does
not relicense the underlying records.

Other sources in `data_sources/tcr_sources.json` are pointers only until their
schema, provenance, duplication and redistribution terms pass validation.

## Multi-omics sources

The v0.6 model is a compact, gene-aligned transformation of public processed
data. Source records remain subject to their repository and publication terms:

- PXD021250, *A proteomic map of resting and activated CD4+ T cells*, PRIDE / ProteomeXchange;
- QTD000031, BLUEPRINT CD4+ T-cell gene-expression QTL summary statistics,
  redistributed by the eQTL Catalogue;
- GSE174666, EPIC-array DNA methylation from purified human CD4/CD8 naive and
  memory lymphocytes, NCBI GEO;
- HGNC complete gene set and Illumina MethylationEPIC v1.0 B5 manifest for
  identifier harmonisation and probe annotation.

No individual-level genotype is bundled. The eQTL layer contains public
summary statistics only. Cite the source study and repository when using the
derived model in publications.
