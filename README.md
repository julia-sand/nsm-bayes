# NSM-Bayes

This repository contains a Python package for robust simulation-based
Bayesian inference with neural score matching (NSM) approaches, including
experimental scripts for a misspecified (contaminated) observation model.

## Overview

The project runs two inference methods on four benchmark simulators
(`gnk`, `sir`, `sir_undercounting`, `turin`):

- **normflow** ("general"): trains an sbi `NLE_A` normalising flow as a
  likelihood surrogate, builds the generalised posterior
  `π(θ)·exp(-β·n·L_NSM(θ; x))` and samples it with a slice sampler.
- **conjugate** ("case 1"): trains a neural exponential family
  (`T_φ`, `b_φ`) by score matching; the generalised posterior is Gaussian
  in closed form.

Both calibrate the tempering parameter **β** by bootstrap coverage
matching, so the `1-α` credible region has `1-α` frequentist coverage.

- `src/nsm_bayes/` package code: simulators, samplers, the general
  (normalising-flow) and conjugate (score-matching) methods, and the shared
  experiment drivers
- `scripts/` Hydra entry points, one per method (normflow / conjugate) and
  experiment (gnk, sir, sir_undercounting, turin)
- reproducible Python environment configuration via `pyproject.toml`

## Repository structure

```text
.
├── configs/            # Hydra configs (one per experiment + _base)
├── examples/           # Jupyter notebooks (gnk_general, sir_conjugate, fromdata)
├── scripts/
│   ├── conjugate/      # score-matching method entry points
│   │   ├── run_gnk.py
│   │   ├── run_sir.py
│   │   ├── run_sir_undercounting.py
│   │   └── run_turin.py
│   └── normflow/       # normalising-flow method entry points
│       ├── run_gnk.py
│       ├── run_sir.py
│       ├── run_sir_undercounting.py
│       └── run_turin.py
├── src/
│   └── nsm_bayes/
│       ├── experiments/    # shared driver + per-experiment modules
│       │   ├── common.py   # ExperimentData, run_experiment, the two method steps
│       │   ├── gnk.py
│       │   ├── sir.py
│       │   ├── sir_undercounting.py
│       │   ├── turin.py
│       │   └── test_data/  # dev-only saved-data bundle builders (not in the wheel)
│       ├── inference/
│       │   ├── beta_bo_calibration.py   # Bayesian-optimisation beta calibration
│       │   ├── beta_calibration_loop.py # unified coverage-matching loop (calibrate_beta)
│       │   ├── beta_update_rules.py     # update rules: gpc (SA), bisection, grid
│       │   ├── bootstrap.py             # importance-reweighted bootstrap helpers
│       │   └── slice_sampler.py         # multivariate slice sampler (normflow method)
│       ├── models/
│       │   ├── conjugate/          # EnergyBasedModel, score-matching training,
│       │   │                        # closed-form ConjugatePosterior, T_phi/b_phi nets
│       │   └── general/            # sbi flow wrapper (GeneralModel),
│       │                            # GeneralPosterior, MCMCSettings
│       ├── simulators/
│       │   ├── add_noise.py        # misspecification helpers
│       │   ├── benchmark_simulators/  # g-and-k, SIR and Turin simulators
│       │   └── fromdata.py         # pkl loader + Gaussian MLE
│       └── utils/          # linear algebra, robust statistics, weights, seeding
├── tests/              # pytest regression + unit tests
├── .gitignore
├── AGENTS.md
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## Installation

Requires Python 3.10–3.12 (on 3.13 `nflows`/`antlr` fail to build; the
dependency stack is numpy-2 compatible, so no numpy upper bound is set).

```bash
python -m pip install -e .
```

For development (tests):

```bash
python -m pip install pytest
```

## Usage

Each experiment script is a standalone Hydra app, so config values can be
overridden on the command line:

```bash
python scripts/normflow/run_gnk.py num_repeat=2
python scripts/conjugate/run_sir.py
python scripts/conjugate/run_sir_undercounting.py robust_flag=true
python scripts/normflow/run_turin.py density_estimator=maf
```

Results are written to `data/<experiment_name>/<method>/` under the directory
you launch from, where `<method>` is `general` for the `scripts/normflow/` runs
and `conjugate` for the `scripts/conjugate/` runs; the Turin experiments
additionally use `eps=<epsilon>/de=<density_estimator>` subfolders.

The β calibration strategy is the `calibration_method` config key
(`gpc`, `bisect` or `grid`; the latter two need the `beta_base_*` value as a
1-D array).

A new experiment script is a short Hydra entry point that passes one
experiment module (for example `nsm_bayes.experiments.gnk`) and one method
step (`run_normflow_methods` or `run_conjugate_method`) to
`nsm_bayes.experiments.common.run_experiment`.

## Testing

Run from the repository root (tests read `configs/*.yaml` by relative path):

```bash
python -m pytest -q
```

## API documentation

All modules, classes and functions in `src/nsm_bayes/` carry
[NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
Use `help()` in Python, or check them with numpydoc (the validation config
lives in `pyproject.toml`):

```bash
python -m pip install numpydoc
python -m numpydoc lint $(find src/nsm_bayes -name '*.py')
```

## Dependencies

Core dependencies are declared in `pyproject.toml`, including:

- `botorch` (pulls in `gpytorch`, imported directly by `inference/beta_bo_calibration.py`)
- `hydra-core` / `omegaconf`
- `numpy` (`>=1.26.4`)
- `scikit-learn`
- `sbi` (`>=0.27,<0.28`)
- `torch`

## Licence

MIT — see [LICENSE](LICENSE).
