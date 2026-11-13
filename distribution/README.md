# Independent liver and spleen source distributions

Build both source ZIPs with the current checkout:

```bash
python -m pip install -e .
python scripts/build_organ_packages.py
```

The outputs in `run_output/organ_packages/` are independent packages for liver
and spleen. Each contains the complete Python package, local launchers, pinned
CPU dependencies for **64-bit CPython 3.12**, organ-specific synthetic fixtures,
documentation, tests, and a SHA256 manifest. No private path, API credential,
other-organ package, GPU or external database is required. Third-party wheels
are downloaded on first installation, or supplied with `--offline-wheels`.

The launcher creates its own virtual environment without system packages and
uses a normal, non-editable package installation. Subsequent `--no-install`
runs do not invoke pip. Real CSV/H5AD input is checked against the selected
organ and stored separately from the demo's synthetic evidence.

## Verify what recipients will run

```bash
python scripts/verify_organ_packages.py --h5ad
```

This command extracts each ZIP outside the repository into paths containing
spaces and Chinese characters, removes inherited Python search paths, cold
installs a separate environment, and runs the installed code from a different
working directory. It checks organ isolation, synthetic labels, first model
promotion, repeated runs without duplicate data or model reset, CSV/H5AD input,
and unchanged packaged-source hashes. The output is saved under
`run_output/organ_package_verification/`.

GitHub CI repeats these checks on Ubuntu, Windows and macOS, together with real
cross-process lock tests. Check the run's actual status before claiming a
platform passed. CI artifacts contain verification logs and source ZIPs.

## Scientific scope

The two packages configure the same tested learning core for different organs.
Their default examples follow a deliberately known linear synthetic numerical
law; the demonstration model therefore uses `random_features=0`. These are
software integration fixtures, not biological validation or AGI demonstrations.

No blood-derived reference weights are relabelled as liver or spleen weights.
The existing public-reference benchmark remains available in the main repository
with its original blood CD4 provenance and limitations. Real organ models require
actual organ data, appropriate controls, features and independent validation.
