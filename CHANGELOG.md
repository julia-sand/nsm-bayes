# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- `nsm_bayes.utils.set_seed(seed)`: seeds `random`, `numpy` and `torch` in one call; experiment scripts now use it with a per-experiment `seed_offset` config value (gnk: 1, sir / sir_undercounting / turin: 123).
- `sigma_inverse` now accepts a `jitter` argument (default `1e-6`) and returns the inverse of `Sigma + jitter * I`; its broken error message (which referenced undefined names) was fixed.
- `gpc.weighted_mean_cov` gained a `jitter` argument (default `1e-6`).
- Small division/log safety epsilons are now named module constants in `gpc.py`, `method.py`, `turin.py`, `sir.py` and `add_noise.py`.

### Changed

- **Breaking API change:** the `ConjugateModel` class in `nsm_bayes.models.conjugate` is renamed `EnergyBasedModel`. All in-repo consumers are updated (`experiments/common.py`, `tests/`, `examples/sir_conjugate.ipynb`, `examples/pickle_simulator.ipynb`); the module path (`nsm_bayes.models.conjugate.conj_model`), the export name change in `nsm_bayes.models.conjugate.__init__` and any external code importing `ConjugateModel` must follow. Results are unaffected; checkpoints (`conjugate_nets_<rep>.pt`) store state dicts only and load unchanged via `EnergyBasedModel.load_checkpoint`.
- Split the monolithic `nsm_bayes.method` module into focused modules and re-pointed all consumers: `robust_mean_cov`, `sample_mean_and_covariance` and `estimate_stats` live in `nsm_bayes.utils.statistics`; `safe_cholesky`, `sigma_inverse`, `sample_mean_and_covariance`, `set_seed` and the `jacobian*`/`hessian*` helpers live in `nsm_bayes.utils.utils`; the conjugate posterior (`compute_posterior_conjugate`, `get_div_term_single`, ...) lives in `nsm_bayes.models.conjugate.conj_posterior`; the multivariate slice sampler lives in `nsm_bayes.slice_sampler`; and general-posterior sampling lives in `nsm_bayes.posterior`. `nsm_bayes.method` now only holds the general score-matching losses and `ScoreMatchingLogPosterior`. Importing `sigma_inverse`, `robust_mean_cov` or `calculate_training_loss` from `nsm_bayes.method` no longer works.
- **Behaviour change 1:** `ConjugateModel.calibrated_posterior` now inverts the data covariance through `sigma_inverse`, whose default adds `1e-6 * I`. The original repo used `inv(S + 1e-6 I)` there, so this restores the original behaviour.
- **Behaviour change 2:** the default step-size schedule of `gpc.calibrate_beta_gpc` changed from `5 / (t + 10)` to `10 / (t + 10)`, matching what every experiment script always overrode it with; the scripts no longer pass the override.
- `c` (IMQ weight scale) is now a default argument (`c: float = 1.0`) of `gpc.calibrate_beta` and `gpc.calibrate_beta_gpc`; the hard-coded `c_obs = 1.` / `c_boot = 1.` inside `calibrate_beta_gpc` were removed. Experiment scripts no longer define or pass `c = 1.`.
- The inline `torch.linalg.inv(S + 1e-6 * torch.eye(...))` calls in the scripts, `gpc.py` and the example notebooks (`sir_conjugate.ipynb`, `gnk_general.ipynb`) were replaced with `sigma_inverse(S)`.
- The dead `step_schedule=lambda t: 10.0/(t+10.)` and `learning_rate_fn=lambda t: 10.0/(t+10.)` keyword overrides in the scripts were removed; they were identical to the functions' defaults, so results are unchanged.

### Fixed

- **Behaviour change:** `GeneralPosterior._per_x_sm_loss` called the raw estimator as `self.flow.estimator.log_prob(xi, theta).reshape(1)` with `xi` of shape `(d_x,)` and a commented-out fallback. Under sbi 0.27 the flow's `log_prob` interprets the leading dimension as `sample_dim` (`(sample_dim, batch_dim)`), so a 1-D `xi` was mis-read as `sample_dim=d_x` and the broadcast against the 100-condition batch produced a `(d_x, 100)` result (400 elements for g-and-k, `d_x=4`) that the `.reshape(1)` could not flatten, raising `RuntimeError: shape '[100, 1]' is invalid for input of size 400` in `GeneralPosterior.sample` and `calibrate_beta`. The call now goes through `GeneralModel.log_prob(xi, theta_vec=theta)`, which batches `xi` to `(1, d_x)` so the estimator sees one input row per condition; the `-1e10` clamp of `GeneralModel.log_prob` also now applies to the score-matching loss, matching the NLE baseline.
- Removed broken dead imports (`compute_posterior_case1`) that made `scripts/normflow/run_gnk.py` and `scripts/normflow/run_sir_undercounting.py` fail at import time.
- `tests/test_run_gnk.py` loads the checkpoint written by `save_checkpoint` (`conjugate_nets_*.pt`).
