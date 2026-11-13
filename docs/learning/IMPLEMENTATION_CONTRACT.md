# Learning core implementation contract

Research prototype. No AGI, clinical, causal-discovery, or autonomous wet-lab claim.
The existing Python project is preserved. Only numpy, pandas and the standard library
are required. Optional H5AD input may use the existing `atlas` extra.

## Records

See `learning/schema.py`. A JSONL row contains observation_id, experiment_id,
group_id (independent holdout unit), organ, cell_type, condition, intervention,
evidence_type, source, features (numeric name/value mapping), outcomes (numeric
name/value mapping), outcome_units and optional measurement_sd, donor_id, batch_id,
source_record_id, species, biological_time_hours and dose/dose_unit. Stable original
source identity is checked in addition to observation/group/experiment identity.
Outcomes may be missing by omitting keys,
never by filling with zero. A candidate has the same context/features, empty outcomes
and units, plus candidate_id and positive cost. Candidate evidence_type is
model_prediction. Experimental summary data, observational atlases, simulations and
model predictions remain distinct.

## World model API (world_model.py)

`BayesianWorldModel.fit(records, *, seed=0, random_features=16, prior_precision=1.0,
allow_synthetic=False)` returns a fitted model. Fit a small nonlinear random-feature
Bayesian regression using only training data; independent output heads support
missing outcomes and known measurement SD. Context encoding must include organ,
cell_type, condition and species (missing species is explicitly unspecified).
Report approximate model uncertainty separately from
residual/measurement noise. This is a learned probabilistic response surrogate,
not a trained transformer or complete biological simulator.

`model.predict(records)` returns a dict with outcome_names (list), mean,
epistemic_variance, noise_variance (N x O numpy arrays), in_domain (N booleans),
domain_reasons (list of lists). Strict feature-name/unit contracts. Unknown organs,
cell types and conditions flag out-of-domain. Feature range outside training also
flags extrapolation; no claimed guarantee for unseen mechanisms.

`model.covariance(records, outcome)` returns N x N latent covariance for that output.
`model.save(path)` / `BayesianWorldModel.load(path)` use NPZ with allow_pickle=False.
`model.metadata` includes training_ids, training_groups, outcome_units,
evidence_types, sources, seed, version/content identity, and training coverage.

## Planner API (planner.py)

`plan_experiment(model, candidates, *, outcome, budget, direction='maximize',
min_information_value=1e-6, allow_extrapolation=False)` returns a JSON-serializable
dict with decision (`experiment` or `stop`), selected candidate, rankings and
limitations. Compare correlated one-step KG per cost on ONE explicit outcome.
Cost is budget units, not subtracted from outcome units. Reject unsupported domain
by default, invalid/nonpositive costs, duplicate candidate IDs and nonfinite data.
Stop on insufficient affordable expected value. No execution of real instruments.

## Continual API (continual.py)

`evaluate_model(model, records)` returns metrics overall, per output and per organ,
with evidence, coverage and extrapolation disclosure. Mean-baseline comparison and
properly group-disjoint reports. `propose_update(current, replay_records,
new_records, validation_records, *, max_relative_regression=0.05,
allow_synthetic=False)` returns `(candidate_model, gate_report)` and never modifies
current. Reject training/validation group or observation overlap, repeated evidence,
unit mismatch, predictions as labels, synthetic/experimental mixing. Require replay
coverage of old training IDs to avoid silent forgetting. Compare overall and
per-organ metrics, store a decision but only caller promotes an accepted model.

## Atlas/data API (atlas.py and adapters.py)

Persistent liver/spleen-ready atlas input: wide raw-count CSV (cell_id then genes)
plus metadata CSV with cell_id, sample_id, donor_id, organ, cell_type, condition.
Retain other cells as contextual atlas data; do not relabel blood as liver/spleen.
Record source, checksum, QC, repeated-cell IDs, batches and biological time if
available. Group pseudobulk by sample/donor/organ/cell_type/condition/time. State
atlas updates are observational and cannot automatically train response heads.
Optional backed H5AD input if dependency available, never implicit counts from log X.
Tests must demonstrate idempotent repeated batches and distinct organ groups.

Repository NPZ adapter must use explicit effect_unit and source. GSE314342 has
declared log2_fold_change; unknown-unit archives require explicit override or reject.
Use positive measurement SE as an observed-entry mask; zero-filled missing effects
must not become experimental labels. No simulated doses/donors/tissues. Include
file checksum and source provenance. Use only pre-intervention descriptors as
features. Reference-table replay is not a prospective biological experiment.

## State belief and batch runner

`summarize_atlas_states(records)` consumes only the current atlas snapshot. It
averages repeated samples within donor before donor-weighted descriptive means
and empirical standard errors; a single donor has unknown between-donor error.
Organ, source, species, cell type, condition, biological time, measured gene panel,
intervention and dose remain separate. This is not a learned latent encoder.

`run_learning_cycle(*, model_path, evidence_path, validation_path, output_dir,
replay_path=None, allow_synthetic=False, max_relative_regression=0.05,
promote=True)` consumes compatible pending response evidence from the journal,
retains full replay, caches attempted evidence/model/validation combinations, and
only promotes accepted candidates. It keeps previous versions and refreshes the
belief sidecar using a shared model lock and replacement-time content checks.
The caller starts polling; no remote background service is implicitly created.

`sync-inbox` imports completed JSON manifests idempotently, bound to a persistent
atlas database UUID. An optional learning config invokes the runner after a
successful ingestion pass. Observational atlas input never becomes an intervention
response label merely because the polling loop is enabled.
