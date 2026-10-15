# NSM-Bayes

This repository contains a Python package for simulation-based Bayesian
inference with neural score matching (NSM) approaches, including
experimental scripts for standard and misspecified settings.

## Overview

The project focuses on Bayesian inference for simulator-based models, with
implementations for:

- `src/nsm_bayes/` package code: simulators, samplers, the general
  (normalising-flow) and conjugate (score-matching) methods, and the shared
  experiment drivers
- `scripts/` Hydra entry points, one per method (normflow / conjugate) and
  experiment (gnk, sir, sir_undercounting, turin)
- reproducible Python environment configuration via `pyproject.toml` and
  `requirements.txt`

## Repository structure

```text
.
├── configs/            # Hydra configs (one per experiment + _base)
├── examples/           # Jupyter notebooks
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
│       │   ├── common.py   # ExperimentData, run_experiment, method steps
│       │   ├── gnk.py
│       │   ├── sir.py
│       │   ├── sir_undercounting.py
│       │   └── turin.py
│       ├── models/         # conjugate and general (normflow) models
│       ├── simulators/     # simulators and misspecification helpers
│       ├── gpc.py          # beta calibration (general and GPC)
│       ├── method.py       # score-matching losses and robust statistics
│       ├── slice_sampler.py
│       └── utils/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Using `pip` with the project metadata:

```bash
python -m pip install -e .
```

Or install the pinned requirements directly:

```bash
python -m pip install -r requirements.txt
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

Results are written to `data/<experiment_name>/` under the directory you
launch from; the Turin experiments additionally use `eps=<epsilon>/de=<density_estimator>`
subfolders.

A new experiment script is a short Hydra entry point that passes one
experiment module (for example `nsm_bayes.experiments.gnk`) and one method
step (`run_normflow_methods` or `run_conjugate_method`) to
`nsm_bayes.experiments.common.run_experiment`.

## Dependencies

Core dependencies are declared in `pyproject.toml`, including:

- `hydra-core`
- `joblib`
- `numpy`
- `scikit-learn`
- `sbi`
- `scipy`
- `torch`
- `tqdm`
