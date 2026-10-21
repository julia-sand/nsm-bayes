# AGENTS.md — nsm-bayes

Guidance for AI coding agents (and humans) working in this repository.
Written against commit `968c79c` (2026-10-21, "refactor calibration loop for
normflow method"). If the code and this file disagree, trust the code and
update this file.

## What this project is

Research code for **robust simulation-based Bayesian inference with neural
score matching (NSM-Bayes)**. Two inference methods are run on four
benchmark simulators, each under a misspecified (contaminated) observation
model:

| Method | Where | Idea |
|---|---|---|
| **normflow** ("general") | `models/general/`, `inference/general/` | Train an sbi `SNLE` flow (`maf`/`mdn`) as likelihood surrogate; build a generalised posterior `π(θ)·exp(-β·n·L_NSM(θ; x))`; sample it with a slice sampler. |
| **conjugate** ("case 1") | `models/conjugate/`, `inference/conjugate/` | Train a neural exponential family (`T_phi`, `b_phi`) by score matching; the generalised posterior is Gaussian in closed form. |

Both calibrate the learning rate / temperature **β** by bootstrap coverage
matching (general posterior calibration, Syring & Martin), so the
`1-α` credible region of the β-posterior has `1-α` frequentist coverage.

Experiments: `gnk` (g-and-k + additive outliers), `sir` (SIR summaries +
contamination), `sir_undercounting` (SIR with reporting rate `q`), `turin`
(radio channel model; `theta` in log space).

## Repository layout (actual, at this commit)

```text
configs/                     Hydra configs: _base.yaml + one per experiment
scripts/{normflow,conjugate}/run_<exp>.py   thin Hydra entry points (8 total)
src/nsm_bayes/
  experiments/common.py      ExperimentData, run_experiment, the two method steps, save/timed
  experiments/<exp>.py       build_problem(cfg) + generate_data(...) per experiment
  models/general/            GeneralModel (SNLE wrapper), GeneralPosterior, MCMCSettings
  models/conjugate/          ConjugateModel, TphiNet/BphiNet, train_q_phi, ConjugatePosterior
  inference/bootstrap.py     bootstrap counts, importance reweighting, coverage indicator
  inference/general/         calibrate_beta (reweighting-based), slice sampler
  inference/conjugate/       calibrate_beta_gpc (closed-form posterior per bootstrap)
  simulators/                benchmark simulators + contamination helpers (add_noise.py)
  utils/                     linalg (sigma_inverse, safe_cholesky, jacobians),
                             statistics (robust/sample moments), weights (imq/huber/none),
                             misc (set_seed)
tests/                       pytest regression + unit tests (conftest has fixtures/stubs)
examples/                    notebooks (may lag the package API)
plot_gnk.py                  ad hoc plotting script
```

Note: the README tree and CHANGELOG still mention `nsm_bayes.gpc`,
`nsm_bayes.method`, `nsm_bayes.posterior`, `nsm_bayes.slice_sampler` and
`nsm_bayes.utils.utils`. **None of these modules exist any more.** Use the
paths above.

## Setup and commands

```bash
# Python 3.10–3.12. numpy is pinned <2; on 3.13 nflows/antlr fail to build.
python -m venv .venv && source .venv/bin/activate
python -m pip install -e . pytest

# Run an experiment (results go to ./data/<experiment_name>/)
python scripts/normflow/run_gnk.py num_repeat=2
python scripts/conjugate/run_sir.py robust_flag=true
python scripts/normflow/run_turin.py density_estimator=maf

# Tests (always from the repo root: tests read configs/*.yaml by relative path)
python -m pytest -q
python -m pytest -q tests/test_run_gnk_conjugate.py   # one experiment
```

The full suite takes ~5 min on CPU. Run the single relevant test file while
iterating, and the full suite before you finish.

## Core design invariants — do not break these

1. **Data before training.** `run_experiment` seeds, generates *all* data
   for a repetition (training + observed), saves it, and only then calls the
   method step. This is what makes the normflow and conjugate variants
   see **bit-identical data** for the same seed, and the
   `test_*_data_identical_across_methods` tests enforce it. Never draw random
   numbers for data inside a method step, and never reorder RNG calls inside
   `generate_data` without expecting every saved dataset to change.
2. **Seeding.** One call to `nsm_bayes.utils.set_seed(cfg.seed_offset + rep)`
   per repetition (seeds `random`, `numpy`, `torch`). Don't add ad hoc
   `torch.manual_seed` calls in library code. `seed_offset` is 1 for gnk and
   123 for sir/sir_undercounting/turin, to reproduce historical results.
3. **Uniform experiment interface.** Every `experiments/<exp>.py` exposes
   exactly `build_problem(cfg) -> (sim, prior, theta_true, d_x)` and
   `generate_data(sim, prior, theta_true, d_x, cfg) -> ExperimentData`,
   even if some arguments are unused. Experiment-specific things to save go in
   `ExperimentData.extras`.
4. **Uniform method-step interface.** `(data, prior, cfg, save_dir, rep) -> None`.
   A new script is just: pick an experiment module + a method step, call
   `common.run_experiment`. Keep scripts thin; logic belongs in `src/`.
5. **No outputs in the repo.** Scripts write to `get_original_cwd()/data/<experiment_name>/`;
   tests write to `tmp_path`. `data/` and `outputs/` are git-ignored.

## Numerical conventions

- **Invert covariances with `utils.linalg.sigma_inverse`**, never
  `torch.linalg.inv(S + eps*I)` inline. Use `safe_cholesky` for factorisations.
  Prefer `torch.linalg.solve` / `cholesky_solve` over forming inverses when
  you only need a solve.
- **Name your epsilons.** Division/log guards are module-level constants
  (`EPS = 1e-12`), not magic literals scattered in expressions.
- **β lives in log space.** Calibration updates `log β` with step
  `k_t = 10/(t+10)`, clipped to `±max_log_step` (0.25) and floored at
  `log(beta_min)`. Keep this structure; `beta_min_*` and `beta_base_*` are
  per-experiment config keys.
- **dtype.** sbi requires a `float32` prior; `GeneralModel` raises otherwise.
  Build priors/`theta_true` as `torch.float32` in `build_problem`.
- **Units.** `ConjugateModel` standardises `x` and `θ` internally
  (`TorchStandardizer`, buffers saved in `state_dict`, `ddof=1`).
  `ConjugatePosterior` takes and returns **original units**; all
  standardise → solve → unstandardise logic stays inside it. Turin and SIR
  `theta` are in transformed (log / unconstrained) space — compare against
  `theta_true` in the same space.
- **Robust weights.** `estimate_stats(x, robust)` gives `(mu_hat, Sigma_inv)`
  (median + MCD when robust); `weight_function("imq"|"huber"|"none", ...)`
  returns `(w2, grad_w2)` with an *analytic* gradient. If you change a weight,
  check the gradient against autograd.
- **Vectorise with `torch.func`** (`vmap`, `jacrev`, `jacfwd`) as the existing
  score/Laplacian code does, rather than Python loops over observations.
- **Log-prob safety.** Flow log-probs go through `nle_log_prob_safe`
  (clamped at `-1e10`); the slice sampler expects `-inf` outside prior support.

## Configuration (Hydra)

- One YAML per experiment, inheriting `_base` (logging format).
  `sir_undercounting.yaml` inherits from `sir.yaml` — so load configs with
  Hydra `compose`, **not** `OmegaConf.load`, or inherited keys go missing.
- Every tunable lives in the YAML with an inline comment. New optional keys
  are read with a default (`cfg.get("beta_min_general", 0.01)`) so older
  configs keep working.
- Don't hard-code hyperparameters in scripts or method steps. Defaults that
  every experiment shares belong as function defaults (e.g. IMQ `c=1.0`,
  step schedule), not repeated overrides.
- `MCMCSettings.from_cfg(cfg)` is the single place MCMC keys are read
  (`num_posterior_samples`, `num_chains`, `warmup_steps`, `thin`).

## Output artifacts

Pickles named `<name>_<rep>.pkl` via `common.save`; timings via
`with common.timed(save_dir, name, rep)` → `cost_<name>_<rep>.pkl`.
Models: `likelihood_estimator_full_<rep>.pt` (normflow),
`conjugate_nets_<rep>.pt` (conjugate: state dicts + standardisers + config).

- normflow writes: `samples_nle_mis`, `samples_sm_general`, `beta_general`,
  `gpc_history_general`, `cost_nle_training`, `cost_nle_mcmc`, `cost_sm_general_mcmc`.
- conjugate writes: `posterior_mean_case1`, `posterior_covariance_case1`,
  `beta_case1`, `gpc_history_case1`, `cost_sm_case1_train`, `cost_sm_case1_posterior`.
- Both: `theta`, `x_sim`, `x_obs_mis`, (`x_obs`), plus `extras`.

Downstream notebooks/plots load these names — renaming one is a breaking
change and must go in the CHANGELOG.

## Testing practices

- Tests call `common.run_experiment` directly with a **tiny config**
  (`conftest._tiny_base`: `num_repeat=1`, small `num_samples`, `T=2`, `B=6`,
  …) and a `tmp_path` save dir — no Hydra app, nothing written to `data/`.
- Heavy components are stubbed by the autouse `_fast_runs` fixture:
  `normflow.SNLE` → `_StubNLE`, and `train_q_phi` capped to 2 epochs.
  **When you change how library code calls the flow** (method names, kwargs
  like `condition=`), update `_StubNLE` to match the real sbi surface.
- Tests assert structure and sanity, not exact numbers: shapes, finiteness,
  symmetric PD covariance, `β ≥ beta_min`, history keys, posterior mean within
  3 sd of truth (gnk), cross-method data identity, determinism under a seed.
- Write unit tests for numerical helpers against a reference
  (e.g. `sigma_inverse` vs `inv(S + jitter·I)`), parametrised over `d=1` and
  `d>1` — the scalar `d_x=1` path is special-cased in several helpers.
- Test the artifacts the method under test actually writes: normflow tests
  should check `*_general` files, conjugate tests `*_case1` files.

## Code style

- `from __future__ import annotations`, type hints on public functions,
  `__all__` in modules with a public API.
- NumPy-style docstrings that state **tensor shapes** (`(n, d_x)`,
  `(M, d_theta)`) for every array argument and return. Keep docstrings in sync
  with signatures (several are currently stale — fix them when you touch the
  function).
- Use `logging.getLogger(__name__)` for diagnostics. Some calibration code
  still uses `print`/`tqdm`; prefer logging in new code and don't add more
  prints to library modules.
- Don't leave commented-out code or `#is this intentional?` notes in
  committed library code; open an issue or note it in the PR instead.
- Keep imports pointing at the real module (`nsm_bayes.utils.linalg`, not a
  re-export path that may move again).

## Change discipline

- Record every user-visible change in `CHANGELOG.md` under **Unreleased**
  (Added / Changed / Fixed). Anything that can change numerical results is
  flagged as **Behaviour change** with the old vs new behaviour, as the
  existing entries do.
- Pure refactors must leave results bit-identical for a fixed seed. Verify by
  running an experiment with a small config before and after and comparing
  the pickles.
- When moving or renaming modules, update in the same change: all imports in
  `src/`, `scripts/`, `tests/`, `examples/*.ipynb`, `plot_gnk.py`, the README
  structure tree, and the CHANGELOG.
- Never commit `data/`, `outputs/`, `sbi-logs/`, checkpoints or
  `__pycache__/`.
