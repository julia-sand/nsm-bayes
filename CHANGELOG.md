# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- `nsm_bayes.utils.set_seed(seed)`: seeds `random`, `numpy` and `torch` in one call; experiment scripts now use it with a per-experiment `seed_offset` config value (gnk: 1, sir / sir_undercounting / turin: 123).
- `sigma_inverse` now accepts a `jitter` argument (default `1e-6`) and returns the inverse of `Sigma + jitter * I`; its broken error message (which referenced undefined names) was fixed.
- `gpc.weighted_mean_cov` gained a `jitter` argument (default `1e-6`).
- Small division/log safety epsilons are now named module constants in `gpc.py`, `method.py`, `turin.py`, `sir.py` and `add_noise.py`.

### Changed

- **Behaviour change 1:** `ConjugateModel.calibrated_posterior` now inverts the data covariance through `sigma_inverse`, whose default adds `1e-6 * I`. The original repo used `inv(S + 1e-6 I)` there, so this restores the original behaviour.
- **Behaviour change 2:** the default step-size schedule of `gpc.calibrate_beta_gpc` changed from `5 / (t + 10)` to `10 / (t + 10)`, matching what every experiment script always overrode it with; the scripts no longer pass the override.
- `c` (IMQ weight scale) is now a default argument (`c: float = 1.0`) of `gpc.calibrate_beta` and `gpc.calibrate_beta_gpc`; the hard-coded `c_obs = 1.` / `c_boot = 1.` inside `calibrate_beta_gpc` were removed. Experiment scripts no longer define or pass `c = 1.`.
- The inline `torch.linalg.inv(S + 1e-6 * torch.eye(...))` calls in the scripts, `gpc.py` and the example notebooks (`sir_conjugate.ipynb`, `gnk_general.ipynb`) were replaced with `sigma_inverse(S)`.
- The dead `step_schedule=lambda t: 10.0/(t+10.)` and `learning_rate_fn=lambda t: 10.0/(t+10.)` keyword overrides in the scripts were removed; they were identical to the functions' defaults, so results are unchanged.

### Fixed

- Removed broken dead imports (`compute_posterior_case1`) that made `scripts/normflow/run_gnk.py` and `scripts/normflow/run_sir_undercounting.py` fail at import time.
- `tests/test_run_gnk.py` loads the checkpoint written by `save_checkpoint` (`conjugate_nets_*.pt`).
