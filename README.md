# NSM-Bayes

This repository contains a Python package for robust simulation-based
Bayesian inference with neural score matching (NSM) approaches, including
experimental scripts for a misspecified (contaminated) observation model.

## Overview

The project runs a Bayesian inference method. The loss is 

There are two surrogate likelihood models used: 

- **normflow**: This is an sbi `NLE_A` normalising flow as the
  likelihood surrogate, builds the generalised posterior
  `π(θ)·exp(-β·n·L_NSM(θ; x))` and samples it with sbi's MCMC slice sampler.
- **conjugate**: trains a neural exponential family
  (`T_φ`, `b_φ`) by score matching. The generalised posterior then has a closed form.

The score-matching losses are weighted, and the inverse temperature β for generalised Bayesian inference is calibrated such that the `1-α` credible region has `1-α` frequentist coverage.

Walkthrough notebooks demonstrating the two methods are found in the `examples` directory, for the [SIR simulator with the conjugate model](examples/sir_conjugate.ipynb), and the [GnK simulator with normflow model](examples/gnk_general.ipynb). 

There are four benchmark example data simulators used (`gnk`, `sir`, `sir_undercounting`, `turin`), but any numpy-compatible data can be passed into the method: see [from data](examples/fromdata.ipynb). 

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
│       ├── models/
│       │   ├── conjugate/          # EnergyBasedModel, score-matching training,
│       │   │                        # closed-form ConjugatePosterior, T_phi/b_phi nets
│       │   └── general/            # sbi flow wrapper (GeneralModel),
│       │                            # GeneralPosterior
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

Requires Python 3.10–3.12.

```bash
python -m pip install -e .
```

## Reproducing the Results

Standard experiments use configuration from configs/ and pass them into scripts using Hydra. Config values can be overridden on the command line. Some examples of how to run the experiments: 

```bash
python scripts/normflow/run_gnk.py num_repeat=2
python scripts/conjugate/run_sir.py
python scripts/conjugate/run_sir_undercounting.py robust_flag=false
```

Results of each run are written to `data/<experiment_name>/<method>/` in the same directory they were run.

## Dependencies

Core dependencies are declared in `pyproject.toml`, including:

- `botorch` (pulls in `gpytorch`, imported directly by `inference/beta_bo_calibration.py`)
- `hydra-core` / `omegaconf`
- `numpy` (`>=1.26.4`)
- `scikit-learn`
- `sbi` (`>=0.27,<0.28`)
- `torch`

## License

MIT — see [LICENSE](LICENSE).
