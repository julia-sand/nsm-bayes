# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- `nsm_bayes.utils.set_seed(seed)`: seeds `numpy` and `torch` in one call; experiment scripts now use it with a per-experiment `seed_offset` config value (gnk: 1, sir / sir_undercounting / turin: 123).
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
- `from __future__ import annotations` was removed from every module: nothing in the repo needed it (no forward references, no `TYPE_CHECKING` imports, Python ≥ 3.10); the corresponding AGENTS.md style rule was dropped. This unmasked an invalid annotation, `timed(...) -> Generator[None]` in `experiments/common.py` (only legal because it was never evaluated), which is now `Generator[None, None, None]`. Results are unchanged.

### Removed

- `random.seed(seed)` from `nsm_bayes.utils.set_seed`: nothing in `src/`, `scripts/`, `tests/` or `examples/` consumes the stdlib `random` module, so seeding it changed nothing. Results are bit-identical for a fixed seed.

- Dead code deleted (no behaviour change for any experiment; results are bit-identical for a fixed seed): `calculate_training_loss1` (`nsm_bayes.models.conjugate.conj_train`), `add_student_t_noise`, the free functions `simulate_contaminated_dataset` and `apply_undercounting_trajectory` (`nsm_bayes.simulators.add_noise`, both duplicates of `SIRSimulator` methods that were never called), `GandKSimulator.generate_observed_data`, `ConjugatePosterior.precompute`, the Turin `simulate(output="data")` branch with its `Ns`/`epsilon`/noise-only-receiver machinery, and the SIR negative-binomial / 5-parameter branch (`obs_model`, `log phi`; the `SIRSimulator(obs_model=...)` keyword is gone — `experiments/sir.py`, `tests/test_run_sir_undercounting_conjugate.py` and `examples/sir_conjugate.ipynb` updated). Also removed: the never-read config keys `n_obs_ref` and `sr_num_samples_per_param*`, `configs/__init__.py`, and the stale `[tool.setuptools.package-data]` entry in `pyproject.toml` (it pointed at a `config/` directory that does not exist).
- `joblib` dependency: the slice sampler ran its chains through `joblib.Parallel(n_jobs=1)` and no caller ever overrode `n_jobs`, so `run_multivariate_slice_sampler_tuned` now loops over chains plainly and its `n_jobs` keyword is gone. Results are bit-identical for a fixed seed.
- The `GNKSimulator` alias of `GandKSimulator` in `nsm_bayes.simulators.benchmark_simulators.simulators`; import `GandKSimulator` instead (`tests/test_run_gnk_conjugate.py` updated).
- The unused `jitter` argument of `sigma_inverse` (it was never applied to the matrix, so behaviour is unchanged).
- Stale `SIRSimulator` docstrings: `generate_training_data` documented three return values but returns two (`theta`, `y`); `summary`'s `y` argument no longer has a `None` default.

### Fixed

- `EnergyBasedModel.save_checkpoint` call site in `run_conjugate_method` still passed `(cfg, save_dir, rep)` after the signature changed to `(save_file)`; it now passes the full path `save_dir / f"conjugate_nets_{rep}.pt"`. The `"config"` key is no longer written to (or asserted in) checkpoints, matching the documented contents (state dicts + standardisers only).

- Added `__init__.py` to `nsm_bayes.simulators`, `nsm_bayes.simulators.benchmark_simulators`, `nsm_bayes.viz` and `nsm_bayes.inference.conjugate`. They were namespace packages, so a non-editable `pip install .` (setuptools `find_packages`) silently omitted them from the wheel; editable installs were unaffected.

- **Behaviour change:** `GeneralPosterior._per_x_sm_loss` called the raw estimator as `self.flow.estimator.log_prob(xi, theta).reshape(1)` with `xi` of shape `(d_x,)` and a commented-out fallback. Under sbi 0.27 the flow's `log_prob` interprets the leading dimension as `sample_dim` (`(sample_dim, batch_dim)`), so a 1-D `xi` was mis-read as `sample_dim=d_x` and the broadcast against the 100-condition batch produced a `(d_x, 100)` result (400 elements for g-and-k, `d_x=4`) that the `.reshape(1)` could not flatten, raising `RuntimeError: shape '[100, 1]' is invalid for input of size 400` in `GeneralPosterior.sample` and `calibrate_beta`. The call now goes through `GeneralModel.log_prob(xi, theta_vec=theta)`, which batches `xi` to `(1, d_x)` so the estimator sees one input row per condition; the `-1e10` clamp of `GeneralModel.log_prob` also now applies to the score-matching loss, matching the NLE baseline.
- Removed broken dead imports (`compute_posterior_case1`) that made `scripts/normflow/run_gnk.py` and `scripts/normflow/run_sir_undercounting.py` fail at import time.
- `tests/test_run_gnk.py` loads the checkpoint written by `save_checkpoint` (`conjugate_nets_*.pt`).
