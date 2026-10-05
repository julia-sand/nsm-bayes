# NSM-Bayes

This repository contains a Python package for simulation-based Bayesian inference with neural score matching (NSM) approaches, including experimental scripts for standard and misspecified settings.

## Overview

The project focuses on Bayesian inference for simulator-based models, with implementations for:

- `src/nsm_bayes/` package code for simulation models, samplers, and experiment runners
- reproducible Python environment configuration via `pyproject.toml` and `requirements.txt`

## Repository structure

```text
.
├── src/
│   └── nsm_bayes/
│       ├── __init__.py
│       ├── config/
│       ├── gpc.py
│       ├── method.py
│       ├── nn_case1.py
│       ├── run_gnk.py
│       ├── run_sir.py
│       ├── run_sir_undercounting.py
│       ├── run_turin.py
│       ├── simulators.py
│       ├── slice_sampler.py
│       └── utils.py
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

After installation, each experiment has a console script (these are Hydra apps, so config values can be overridden on the command line):

```bash
nsm-bayes-gnk
nsm-bayes-sir
nsm-bayes-sir-undercounting
nsm-bayes-turin
```

Equivalently, run them as modules:

```bash
python -m nsm_bayes.run_gnk num_repeat=2
```

Results are written to `data/<experiment_name>/` under the directory you launch from. The Turin experiment additionally expects pre-simulated data at `rca_sbi/data_turin/turin_theta.pt` and `turin_x_sim.pt`, also relative to the launch directory.

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
