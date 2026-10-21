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
│       ├── inference/
│       │   ├── bootstrap.py        # importance-reweighted bootstrap helpers
│       │   ├── conjugate/          # GPC beta calibration (conjugate method)
│       │   └── general/            # beta calibration and slice sampler (normflow method)
│       ├── models/
│       │   ├── conjugate/          # exponential-family surrogate, training, closed-form posterior
│       │   └── general/            # sbi flow wrapper, generalised posterior, MCMC settings
│       ├── simulators/
│       │   ├── add_noise.py        # misspecification helpers
│       │   └── benchmark_simulators/  # g-and-k, SIR and Turin simulators
│       ├── utils/          # linear algebra, robust statistics, weight functions
│       └── viz/            # plotting scripts
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

Results are written to `data/<experiment_name>/<method>/` under the directory
you launch from, where `<method>` is `general` for the `scripts/normflow/` runs
and `conjugate` for the `scripts/conjugate/` runs; the Turin experiments
additionally use `eps=<epsilon>/de=<density_estimator>` subfolders.

A new experiment script is a short Hydra entry point that passes one
experiment module (for example `nsm_bayes.experiments.gnk`) and one method
step (`run_normflow_methods` or `run_conjugate_method`) to
`nsm_bayes.experiments.common.run_experiment`.

## API documentation

All modules, classes and functions in `src/nsm_bayes/` carry
[NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
Use `help()` in Python, or check them with numpydoc:

```bash
python -m pip install numpydoc
python -m numpydoc lint $(find src/nsm_bayes -name '*.py')
```

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
