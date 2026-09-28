# NSM-Bayes

This repository contains a Python package for simulation-based Bayesian inference with neural summary model (NSM) approaches, including experimental scripts for standard and misspecified settings.

## Overview

The project focuses on Bayesian inference for simulator-based models, with implementations for:

- `src/nsm_bayes/` package code for simulation models, samplers, and experiment runners
- `baselines/` directory containing reference baseline methods and evaluation utilities
- reproducible Python environment configuration via `pyproject.toml` and `requirements.txt`

## Repository structure

```text
.
├── baselines/
│   ├── README.md
│   ├── Robust-SBI/
│   ├── neuralgbi/
│   ├── npl_mmd_project/
│   ├── results/
│   ├── robust_snle/
│   └── scoring_rule/
├── src/
│   └── nsm_bayes/
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

The package exposes experiment runners in `src/nsm_bayes/`. Examples include:

```bash
python src/nsm_bayes/run_gnk.py
python src/nsm_bayes/run_sir.py
python src/nsm_bayes/run_turin.py
```

Depending on the experiment, some scripts may require additional configuration or environment setup. The `baselines/README.md` file documents additional benchmark pipelines and reference implementations.

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

## Baselines

The `baselines/` directory contains comparison methods and experiments for robust simulation-based inference, including:

- Robust-SBI
- neuralgbi
- npl_mmd_project
- robust_snle
- scoring_rule

See `baselines/README.md` for details on running these benchmark methods.

## Notes

This repository is primarily intended for research and experiment execution rather than as a general-purpose library package. Several scripts and support modules are specific to simulation experiments and benchmarking workflows.

## License

No license file is included in the repository root. If you plan to distribute or reuse this code publicly, confirm the intended project license before publication.
