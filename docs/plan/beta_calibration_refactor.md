# Plan: `calibrate_beta` — pluggable calibration strategies

Status: **draft for review — not yet implemented**

## 1. Goal

Refactor `calibrate_beta` in `src/nsm_bayes/inference/beta_calibration_loop.py`
so the fixed stochastic-approximation (GPC) loop is replaced by a
`method: str = "gpc"` argument that dispatches to multiple calibration
strategies:

| method | what it does | `initial_beta` |
|---|---|---|
| `gpc` (default) | current inlined SA loop: `log_beta += clip(k_t (c̄ − target), ±max_log_step)`, floored at `log(beta_min)`, `k_t = 10/(t+10)` | scalar (required) |
| `rm` | Robbins–Monro: `a/(t+1) (c̄ − target)`, `a = 1/|slope|` with `|slope|` clamped to `[0.02, 0.5]`; same clip + floor | scalar (required; slope must be supplied) |
| `bisect` | bisection in `log beta`: current point replaces the bracket end on its side of the target, next point is the bracket midpoint; same floor | array (required — supplies the bracket) |
| `grid` | non-iterative: coverage on a grid, closest point wins (ties: coverage ≥ target, then larger beta) | array (required) |

The stepping logic already exists as reference implementations in
`examples/experimental/beta_update.py` (`step_gpc`, `step_rm`, `step_bisect`,
`run_grid_search`, `_grid_start`, `BetaState`). That logic is ported into the
main module; the experimental file is left untouched.

## 2. Design constraints (agreed)

1. **General (normflow) method accepts `gpc` and `rm` only.** A scalar
   `initial_beta` is required; an array raises `ValueError`. Rationale:
   the array `initial_beta` triggers a full grid search, which is only
   implemented for the conjugate (closed-form) posterior.
2. **Conjugate method:** scalar `initial_beta` → `gpc`/`rm`; array
   `initial_beta` → `grid` search (log-bisection refinement of the crossing,
   interpolated start, plus `slope`/`bracket` extraction) feeds the
   `rm`/`bisect` strategies.
3. **`gpc` is the default** and must be **bit-identical** to the current
   inlined loop when `initial_beta` is a scalar (pure-refactor invariant —
   verify per AGENTS.md by running one tiny experiment before/after and
   comparing the pickles).
4. **Config plumbing reuses the existing keys** `beta_base_general` /
   `beta_base_case1` as the `initial_beta` source (values unchanged); a new
   `calibration_method` key selects the strategy.

## 3. New public signature

```python
def calibrate_beta(
    posterior,
    x_obs: torch.Tensor | None = None,
    initial_beta: float | np.ndarray | torch.Tensor = 0.01,
    method: str = "gpc",
    alpha: float = 0.05,
    B: int = 200,
    n_iterations: int = 10,
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
    ess_threshold: float = 0.3,
    refine_steps: int = 3,
    rm_slope: float | None = None,   # only for method="rm" with scalar initial_beta
) -> tuple[float, dict]:
```

- `beta_init` is renamed to `initial_beta` and now also accepts a 1-D
  numpy array / tensor (lists/tuples are rejected with `TypeError`,
  matching the experimental code).
- `step_schedule` is dropped from the public API: the `10/(t+10)` schedule
  is the only one used and becomes a module default (AGENTS.md: shared
  defaults live as function defaults, not repeated overrides).
- `rm_slope` is an escape hatch for conjugate `rm` without a grid; the
  primary path derives the slope from the array `initial_beta`.

### Returns

`beta: float` — the calibrated value, plus the existing flat `history` dict
with the existing keys (names are stable for downstream notebooks):

- `{"betas": [...], "coverages": [...]}` — always; for `grid`, the grid in
  increasing order and its coverages (the loop's per-iteration history is
  empty since there is no loop).
- `grid: {"betas": [...], "coverages": [...]}` — extra evaluated points
  (grid + refinement mids) when `initial_beta` is an array; the final
  `bracket: (lo, hi)` in beta units when a crossing was found.

## 4. Validation rules

Run up front, before any `prepare_calibration` / coverage work:

| condition | error |
|---|---|
| `method` not in `{"gpc", "rm", "bisect", "grid"}` | `ValueError` (list valid methods) |
| `initial_beta` is a list/tuple | `TypeError` (use `np.array` / tensor) |
| array `initial_beta` with `ndim != 1`, empty, or non-positive values | `ValueError` |
| general (normflow) posterior with array `initial_beta` | `ValueError` |
| general (normflow) posterior with `method` in `{"bisect", "grid"}` | `ValueError` (only `gpc`, `rm`) |
| `method="rm"` with scalar `initial_beta` and no `rm_slope` and no `posterior.slope` | `ValueError` (needs a coverage slope) |
| `method="bisect"` without an array `initial_beta` whose coverage crosses the target | `ValueError` (needs a bracket) |

Posterior class detection: an `isinstance` check against
`GeneralPosterior` / `ConjugatePosterior` is brittle to the test fakes;
instead add a small capability attribute on each posterior class,
e.g. `supports_grid_search: bool` (False for `GeneralPosterior`, True for
`ConjugatePosterior`), and validate against that. This mirrors how
`prepare_calibration` / `estimate_coverage` already encode per-method
differences behind one flat signature.


## 5. Internal structure

`beta_calibration_loop.py` keeps the dispatch and the shared
one-time setup; the stepping logic lives next to it:

```text
src/nsm_bayes/inference/
  beta_calibration_loop.py    # calibrate_beta: validate -> prepare -> dispatch
  beta_update_rules.py        # BetaState, step_gpc, step_rm, step_bisect,
                              # run_grid_search, grid_start (ported from
                              # examples/experimental/beta_update.py)
```

Flow inside `calibrate_beta`:

1. **Validate** (table above).
2. **`posterior.prepare_calibration(...)`** exactly as today, using the
   scalar start. For an array `initial_beta`, the scalar start is the
   grid-search result (or the closest grid point when there is no
   crossing) -- the grid search itself is part of `prepare` for the
   conjugate method, so both posterior classes keep one flat
   `prepare_calibration(beta, alpha, B, x_obs, ess_threshold)` signature
   and the loop does not need to know.
   - General: the base MCMC pool is still drawn at the scalar start, as
     now.
   - Conjugate + array: `prepare_calibration` additionally caches the
     grid `slope` and `bracket` on the posterior (used by the `rm` /
     `bisect` steps) and returns them, or the calibrator retrieves them
     from the posterior after prepare.
3. **Dispatch:**
   - `grid` -> `run_grid_search`: no loop; coverage on the grid; return
     the closest point.
   - `gpc` / `rm` / `bisect` -> the shared iteration loop:

     ```text
     for t in range(n_iterations):
         beta     = exp(state.log_beta)
         c_hat    = posterior.estimate_coverage(beta, alpha, B, x_obs, ess_threshold)
         state.log_beta = step_fn(state, c_hat, t, max_log_step=max_log_step,
                                  beta_min=beta_min, target_coverage=1 - alpha,
                                  **rule_kwargs)
         history["betas"].append(exp(state.log_beta))
         history["coverages"].append(c_hat)
     ```

     `step_gpc` is the current inlined update verbatim, so `method="gpc"`
     is bit-identical to today's behaviour.

### Posterior-side changes

- `ConjugatePosterior.prepare_calibration` gains an optional `grid`
  argument (a list of positive floats; `None` keeps today's behaviour):
  when set, it runs the grid + refinement search (the experimental
  `_grid_start` logic, using the same bootstrap coverage evaluation the
  class already performs) and caches `self.calibration_start`,
  `self.calibration_slope`, `self.calibration_bracket` plus the grid
  history. `theta_hat` is still cached first, as now.
- `GeneralPosterior` gains `supports_grid_search = False` (and
  `ConjugatePosterior` gains `True`); no other changes --
  `estimate_coverage` stays as is, including its ESS-gated pool refresh.


## 6. Config changes

New key (each experiment YAML, next to the existing beta keys):

```yaml
calibration_method: gpc     # one of: gpc, rm, bisect, grid
```

`beta_base_general` / `beta_base_case1` are **reused as `initial_beta`
sources** and may be scalar or a 1-D array:

```yaml
beta_base_general: 1.0        # general: scalar only (gpc/rm)
beta_base_case1: 0.1          # conjugate: scalar or [0.01, 0.1, 1.0, 10.0]
```

`src/nsm_bayes/experiments/common.py`:

- `run_normflow_methods` (~line 227): pass
  `initial_beta=cfg.beta_base_general, method=cfg.get("calibration_method_general", cfg.get("calibration_method", "gpc"))`.
  (A single `calibration_method` would be ambiguous since the two methods
  can use different strategies -- see open question Q1.)
- `run_conjugate_method` (~line 327): pass
  `initial_beta=cfg.beta_base_case1, method=...` the same way.
- Docstrings updated to mention the new keys.

`config_keys` lists in `src/nsm_bayes/experiments/test_data/make_test_data_*.py`
get the new key(s) so tiny test bundles carry them.

## 7. Tests

`tests/test_units.py`:

- Keep the existing forced-coverage tests; they now run `method="gpc"`
  (default) and must stay bit-identical.
- Add known-answer tests for the pure stepping rules (port from
  `examples/experimental/test_beta_update.py`):
  - `gpc` update matches the closed form on a fixed coverage sequence.
  - `rm` raises without a slope; with a slope, the first step matches
    `a/(1) * (c - target)` clipped.
  - `bisect` halves the bracket each step; final `beta` is inside the
    initial bracket.
  - `grid` picks the closest coverage; ties prefer coverage >= target,
    then larger beta; rejects lists/tuples, empty, non-positive grids.
- Add validation tests: general posterior + array -> `ValueError`;
  general + `bisect`/`grid` -> `ValueError`; unknown method -> `ValueError`.

`conftest.py`: the tiny-config helper gets `calibration_method: gpc` (and
any method-specific key) so existing tests are unaffected.

Full suite: run `python -m pytest -q` before finishing; while iterating,
`python -m pytest -q tests/test_units.py`.

## 8. Documentation & CHANGELOG

- CHANGELOG (Unreleased):
  - **Breaking API change:** `calibrate_beta(..., beta_init=...,
    step_schedule=...)` -> `calibrate_beta(..., initial_beta=..., method=...)`;
    old kwargs raise `TypeError`.
  - **Added:** `rm` / `bisect` / `grid` calibration methods; array
    `initial_beta` grid search for the conjugate method;
    `calibration_method` config key.
  - **Behaviour change:** none for `method="gpc"` with a scalar
    `initial_beta` (bit-identical for a fixed seed -- verified).
- AGENTS.md: update the `inference/` layout lines (new
  `beta_update_rules.py`) and the calibration description.
- `src/nsm_bayes/inference/__init__.py` docstring: mention the
  strategies.
- Notebooks (`examples/*.ipynb`) keep working: default `method="gpc"` and
  scalar `initial_beta` are exactly the old call shape.

## 9. Out of scope / open questions

- **config key granularity:** `calibration_method` used only for the conjugate method. general method only the gpc, otherwise raise error.
- **allow only gpc with scalar for the general method** 
- yes. **Q3 -- history extras:** whether `gpc_history` pkl gains the
  `grid`/`bracket`/`slope` entries (plan: yes, additive; downstream code
  reads only `betas`/`coverages`, so this is safe).
- yes. `examples/experimental/beta_update.py` and its tests: left in place as
  reference during the port; a follow-up could delete them once the port is
  verified.
- yes. The experimental `step_gd` rule is **not** ported (it was rejected in
  the design discussion; the four methods are `gpc`, `rm`, `bisect`, `grid`).

