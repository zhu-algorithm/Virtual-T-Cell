# Learned response prototype: reference replay only

`gse314342_response_prototype.npz` is an actually fitted, compact Bayesian
random-feature response surrogate, saved without pickle. It is not a released
AGI or a liver/spleen single-cell foundation model.

Reproduce from the repository's existing processed reference archive:

```bash
python -m pip install -e .
python scripts/benchmark_learning_core.py
```

The benchmark writes its active model to
`run_output/reference_benchmark/active_model.npz`. The included copy preserves
the initial model because its proposed update failed the PDCD1 regression gate.
No held-out test value is used to select the model or its hyperparameters.

## Data and support

- Source: GSE314342, human blood-derived primary CD4 T cells.
- Parent archive: `models/gse314342_primary_cd4_virtual_t_cell.npz`.
- Parent SHA256: `5e1c5d8962546c873b14a859e3542839d7e497f0ba44e5f52168506ad2e3c99b`.
- Trained initial split: 921 condition summaries, 307 perturbation targets.
- Nine response genes; outcomes measured in `log2_fold_change`.
- Inputs: fixed repository pathway/network priors and target/readout identity.
- Organ, cell type, condition and species are explicit categorical contexts.
- Missing/zero-filled reference entries are excluded using positive measurement SE.
- No raw single-cell matrix or independent donor trajectories were used.

The NPZ contains record hashes, source identities,
training groups, units, feature ranges, support domains, fitted coefficients,
posterior covariance, estimated residual noise and numerical assumptions.
Loading this model does not imply support for liver/spleen or mouse interventions.

## Observed held-out results and limitations

Fixed target-group test (64 targets, 192 condition summaries, same parent study):

| Model | Pooled RMSE, log2 fold change |
|---|---:|
| This nonlinear-feature surrogate | 0.724163 |
| Linear Bayesian baseline on the same evidence | 0.724012 |
| Training mean baseline | 0.727320 |
| Zero-effect baseline | 0.752713 |

These values do not show a meaningful nonlinear advantage. The feature set gives
only seven distinct vectors across all 512 targets; 504 targets lack the small
fixed pathway/network annotation set. Three test rows also exceed one training
feature range. A donor-independent or cross-study result is not established.

The archive's original selection of genes/targets used global data before this
new split. Treat this as a reproducible engineering/reference replay, not an
unbiased biological discovery benchmark. Approximate interval coverage on this
test is a diagnostic, not a general uncertainty calibration guarantee.

Full results: [benchmark](../../docs/learning/reference_benchmark.json) and
[summary](../../docs/learning/reference_benchmark_summary.json). See
[THIRD_PARTY_DATA.md](../../THIRD_PARTY_DATA.md) and the original GSE314342 source
for source attribution and data terms.
