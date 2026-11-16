"""Pluggable beta update rules for the conjugate GPC calibration (experimental).

Companion to
:func:`nsm_bayes.inference.conjugate.calibrate_beta.calibrate_beta_gpc` that
does not modify the library: it drives the same bootstrap-coverage loop,
delegating every posterior evaluation to
:meth:`~nsm_bayes.models.conjugate.conj_posterior.ConjugatePosterior.calibrated_posterior`.
Only the ``log(beta)`` update after each bootstrap-coverage measurement changes
between rules: ``step_gpc`` reproduces the library's stochastic-approximation
step exactly, ``step_gd`` runs plain gradient descent on
``(coverage - target)^2``, ``step_rm`` is a Robbins-Monro step whose gain
is set by the slope of the coverage curve and ``step_bisect`` keeps bisecting
the grid bracket, one coverage evaluation per step.  :func:`run_grid_search`
is the non-iterative baseline: coverage on a fixed grid, closest point wins.
``initial_beta`` is a number
(fixed start) or an array of ``beta`` values grid-searched for the start.
"""
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from scipy.stats import chi2

from nsm_bayes.models.conjugate.conj_posterior import ConjugatePosterior
from nsm_bayes.utils.statistics import mahalanobis_sq

__all__ = [
    "BetaState", "step_gpc", "step_gd", "step_rm", "step_bisect",
    "run_update_rule", "run_grid_search",
]


def run_update_rule(
    step_fn: Callable[..., float],
    x_obs: torch.Tensor,
    posterior: ConjugatePosterior,
    initial_beta: Any,
    *,
    target_coverage: float = 0.95,
    num_iterations: int = 50,
    num_bootstraps: int = 200,
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
    refine_steps: int = 3,
    seed: int | None = None,
    **rule_kwargs: Any,
) -> dict[str, Any]:
    """Run the GPC bootstrap loop with the update rule ``step_fn``.

    Mirrors ``calibrate_beta_gpc`` exactly (same ``theta_hat``, same
    bootstrap stream, same ``chi2`` test, same defaults); with
    ``step_fn=step_gpc`` and a numeric ``initial_beta`` the result is
    bit-identical to a library call from the same RNG state.  ``step_fn``
    takes ``(state, coverage, t, *, max_log_step, beta_min, target_coverage,
    **rule_kwargs)`` and returns the updated ``log(beta)``.

    ``initial_beta`` is a number, used as the starting ``beta``, or a 1-D
    numpy array / tensor of ``beta`` values, grid-searched for the start
    (see :func:`_grid_start`); the grid also sets ``state.slope`` for
    :func:`step_rm` and ``state.bracket`` for :func:`step_bisect`.  Lists and
    tuples are rejected.

    ``seed=None`` leaves the global torch RNG alone (as the library does);
    an integer reseeds it before the loop, and again after a grid search, so
    rules run with the same seed draw the same bootstrap resamples.  With
    ``robust=True`` the MCD estimator also draws from numpy's global RNG,
    which this function does not seed.

    Returns
    -------
    dict
        ``{"beta": float, "history": {"betas": [...], "coverages": [...]}}``,
        plus ``"initial_beta"``, ``"slope"``, ``"bracket"`` (the final
        ``(lo, hi)`` in ``beta``) and ``"grid"`` when ``initial_beta`` is an
        array.
    """
    d_theta = posterior.prior_mean.shape[0]
    chi2_value = chi2.ppf(target_coverage, df=d_theta)
    if seed is not None:
        torch.manual_seed(seed)  # the library draws torch.randint on the global RNG

    with torch.no_grad():
        theta_hat = posterior.compute_theta_hat_true(x_obs)

    if isinstance(initial_beta, (list, tuple)):
        raise TypeError(
            "initial_beta must be a number or a numpy array / tensor, "
            "e.g. np.array([0.01, 0.1, 1.0]); lists and tuples are not accepted"
        )
    values = torch.as_tensor(initial_beta, dtype=torch.float64)
    extra: dict[str, Any] = {}
    slope = bracket = None
    if values.ndim == 0:  # a number: fixed start
        start = float(values)
    elif values.ndim == 1:  # an array: grid search
        with torch.no_grad():
            start, slope, bracket, extra["grid"] = _grid_start(
                x_obs,
                posterior,
                theta_hat,
                values.tolist(),
                chi2_value=chi2_value,
                num_bootstraps=num_bootstraps,
                beta_min=beta_min,
                target_coverage=target_coverage,
                refine_steps=refine_steps,
            )
        extra["initial_beta"] = start
        extra["slope"] = slope
        if seed is not None:
            torch.manual_seed(seed)  # loop's bootstrap stream does not depend on the grid
    else:
        raise ValueError("initial_beta array must be 1-D")

    state = BetaState(log_beta=math.log(start), slope=slope, bracket=bracket)
    history = {"betas": [], "coverages": []}
    for t in range(num_iterations):
        beta = math.exp(state.log_beta)
        coverage = _bootstrap_coverage(
            x_obs, posterior, theta_hat, beta, num_bootstraps, chi2_value
        )
        state.log_beta = step_fn(
            state,
            coverage,
            t,
            max_log_step=max_log_step,
            beta_min=beta_min,
            target_coverage=target_coverage,
            **rule_kwargs,
        )
        history["betas"].append(math.exp(state.log_beta))
        history["coverages"].append(coverage)

    final = history["betas"][-1] if history["betas"] else start  # no iterations: the start
    if state.bracket is not None:
        extra["bracket"] = (math.exp(state.bracket[0]), math.exp(state.bracket[1]))
    return {"beta": final, "history": history, **extra}


def run_grid_search(
    x_obs: torch.Tensor,
    posterior: ConjugatePosterior,
    betas: Any,
    *,
    target_coverage: float = 0.95,
    num_bootstraps: int = 200,
    seed: int | None = None,
    **unused: Any,
) -> dict[str, Any]:
    """Coverage on a fixed grid of ``beta``; return the point closest to the target.

    ``betas`` is a 1-D numpy array / tensor (lists and tuples are rejected).
    Each value costs one coverage evaluation, the same test as
    :func:`run_update_rule`.  Ties in ``|coverage - target|`` go to coverage
    at or above the target, then to the larger ``beta``.  Other keyword
    arguments of :func:`run_update_rule` (``num_iterations``, ``beta_min``,
    ...) are accepted and ignored, so both can be called the same way.

    Returns
    -------
    dict
        ``{"beta": float, "history": {"betas": [...], "coverages": [...]}}``
        with the grid in increasing order and its coverages.
    """
    del unused
    if isinstance(betas, (list, tuple)):
        raise TypeError("betas must be a numpy array / tensor, e.g. np.logspace(-4, 1, 10)")
    values = torch.as_tensor(betas, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("betas must be a non-empty 1-D array")
    grid = sorted(values.tolist())
    if grid[0] <= 0:
        raise ValueError("betas must be positive")

    d_theta = posterior.prior_mean.shape[0]
    chi2_value = chi2.ppf(target_coverage, df=d_theta)
    if seed is not None:
        torch.manual_seed(seed)  # the library draws torch.randint on the global RNG
    with torch.no_grad():
        theta_hat = posterior.compute_theta_hat_true(x_obs)
        covs = [
            _bootstrap_coverage(x_obs, posterior, theta_hat, b, num_bootstraps, chi2_value)
            for b in grid
        ]

    # rank by distance to target (rounded so k/B ties are exact), then prefer
    # coverage >= target, then the larger beta
    ranked = [
        (round(abs(c - target_coverage), 12), c < target_coverage, -b, b)
        for b, c in zip(grid, covs)
    ]
    best = min(ranked)[3]
    return {"beta": best, "history": {"betas": grid, "coverages": covs}}


@dataclass
class BetaState:
    """Mutable per-rule state for the ``log beta`` update.

    ``log_beta`` is floored at ``log(beta_min)``.  From the grid start (both
    ``None`` without one): ``slope`` is ``d coverage / d log(beta)``, used by
    :func:`step_rm`; ``bracket`` is ``(log lo, log hi)`` with coverage at or
    above the target at ``lo`` and below it at ``hi``, used by
    :func:`step_bisect`.
    """

    log_beta: float
    slope: float | None = None
    bracket: tuple[float, float] | None = None

def _bootstrap_coverage(
    x_obs: torch.Tensor,
    posterior: ConjugatePosterior,
    theta_hat: torch.Tensor,
    beta: float,
    num_bootstraps: int,
    chi2_value: float,
) -> float:
    """Empirical GPC coverage of ``theta_hat`` at ``beta``.

    ``num_bootstraps`` resamples of the rows of ``x_obs`` (``torch.randint``
    on the *global* generator, the same draw the library uses), each scored by
    whether ``theta_hat`` falls inside the credible ellipsoid of
    ``posterior.calibrated_posterior(x_boot, beta)``.
    """
    n_obs = x_obs.shape[0]
    device = x_obs.device
    hits = 0
    for _ in range(num_bootstraps):
        x_boot = x_obs[torch.randint(0, n_obs, (n_obs,), device=device)]
        mu_boot, Sigma_boot = posterior.calibrated_posterior(x_boot, beta)
        hits += mahalanobis_sq(theta_hat, mu_boot, Sigma_boot).item() <= chi2_value
    return hits / num_bootstraps

def _default_learning_rate(t: int) -> float:
    """``k_t = 10 / (t + 10)``, the library's default GPC gain."""
    return 10.0 / (t + 10)

def step_gpc(
    state: BetaState,
    coverage: float,
    t: int,
    *,
    max_log_step: float,
    beta_min: float,
    target_coverage: float,
    learning_rate_fn: Callable[[int], float] | None = None,
) -> float:
    """The library's GPC step (reference rule):
    ``log_beta += clip(k_t * (coverage - target), +/-max_log_step)``, then
    floored at ``log(beta_min)``.  ``k_t = learning_rate_fn(t)``, by default
    ``10 / (t + 10)``.  Bit-identical to ``calibrate_beta_gpc``.
    """
    k_t = _default_learning_rate(t) if learning_rate_fn is None else learning_rate_fn(t)
    log_step = max(-max_log_step, min(max_log_step, k_t * (coverage - target_coverage)))
    return max(state.log_beta + log_step, math.log(beta_min))


def step_gd(
    state: BetaState,
    coverage: float,
    t: int,
    *,
    max_log_step: float,
    beta_min: float,
    target_coverage: float,
    lr: float = 0.2,
) -> float:
    """Plain gradient descent on ``(coverage - target)^2`` with a constant
    learning rate (no clipping), then the ``log(beta_min)`` floor.
    ``t`` / ``max_log_step`` are unused; kept for a uniform rule signature.
    """
    del t, max_log_step
    grad = -2.0 * (coverage - target_coverage)
    return max(state.log_beta - lr * grad, math.log(beta_min))


def step_rm(
    state: BetaState,
    coverage: float,
    t: int,
    *,
    max_log_step: float,
    beta_min: float,
    target_coverage: float,
    slope: float | None = None,
) -> float:
    """Robbins-Monro step with gain tuned to the coverage curve:
    ``log_beta += clip(a / (t + 1) * (coverage - target), +/-max_log_step)``
    with ``a = 1 / |slope|`` (``|slope|`` clamped to ``[0.02, 0.5]``), then
    the ``log(beta_min)`` floor.  ``slope`` is ``d coverage / d log(beta)``;
    by default it is ``state.slope`` from an array ``initial_beta``.
    """
    s = state.slope if slope is None else slope
    if s is None:
        raise ValueError(
            "step_rm needs the coverage slope: pass an array initial_beta whose "
            "coverage crosses the target, or slope=..."
        )
    a = 1.0 / min(max(abs(s), 0.02), 0.5)
    log_step = max(-max_log_step, min(max_log_step, a / (t + 1) * (coverage - target_coverage)))
    return max(state.log_beta + log_step, math.log(beta_min))


def step_bisect(
    state: BetaState,
    coverage: float,
    t: int,
    *,
    max_log_step: float,
    beta_min: float,
    target_coverage: float,
) -> float:
    """Bisection in ``log(beta)``, continuing the grid start's bracket.

    The coverage at the current ``beta`` replaces the bracket end on its side
    of the target (``lo`` if at or above it, ``hi`` if below), and the next
    ``beta`` is the bracket's midpoint, floored at ``log(beta_min)``.  Needs an
    array ``initial_beta`` whose coverage crosses the target.
    ``t`` / ``max_log_step`` are unused; kept for a uniform rule signature.
    """
    del t, max_log_step
    if state.bracket is None:
        raise ValueError(
            "step_bisect needs a bracket: pass an array initial_beta whose "
            "coverage crosses the target"
        )
    lo, hi = state.bracket
    if coverage >= target_coverage:
        lo = state.log_beta
    else:
        hi = state.log_beta
    state.bracket = (lo, hi)
    return max(0.5 * (lo + hi), math.log(beta_min))


def _grid_start(
    x_obs: torch.Tensor,
    posterior: ConjugatePosterior,
    theta_hat: torch.Tensor,
    grid: list[float],
    *,
    chi2_value: float,
    num_bootstraps: int,
    beta_min: float,
    target_coverage: float,
    refine_steps: int,
) -> tuple[float, float | None, tuple[float, float] | None, dict[str, list[float]]]:
    """Grid-search start for ``beta``.

    Evaluates the coverage at each ``beta`` in ``grid``, bisects the first
    bracket where it drops below the target ``refine_steps`` times in
    ``log(beta)``, and interpolates the crossing; the slope of the final
    bracket is returned for :func:`step_rm`.  With no crossing it returns the
    largest grid value (all above target), the smallest (all below) or the
    closest, and slope and bracket ``None``.  Returns ``(start, slope,
    bracket, {"betas": [...], "coverages": [...]})``, ``bracket`` being the
    final ``(log lo, log hi)``.
    """
    betas = sorted(float(b) for b in grid)
    if not betas or betas[0] <= 0:
        raise ValueError("initial_beta grid must contain positive values")

    covs = [
        _bootstrap_coverage(x_obs, posterior, theta_hat, b, num_bootstraps, chi2_value)
        for b in betas
    ]
    seen = dict(zip(betas, covs))  # every evaluated beta -> coverage

    i = None  # first j with covs[j] >= target > covs[j + 1]
    for j in range(len(betas) - 1):
        if covs[j] >= target_coverage > covs[j + 1]:
            i = j
            break

    slope = bracket = None
    if i is not None:
        lo, c_lo, hi, c_hi = betas[i], covs[i], betas[i + 1], covs[i + 1]
        for _ in range(refine_steps):
            mid = math.sqrt(lo * hi)
            c_mid = _bootstrap_coverage(
                x_obs, posterior, theta_hat, mid, num_bootstraps, chi2_value
            )
            seen[mid] = c_mid
            if c_mid >= target_coverage:
                lo, c_lo = mid, c_mid
            else:
                hi, c_hi = mid, c_mid
        frac = (c_lo - target_coverage) / (c_lo - c_hi)
        start = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        slope = (c_hi - c_lo) / (math.log(hi) - math.log(lo))
        bracket = (math.log(lo), math.log(hi))
    elif all(c >= target_coverage for c in covs):
        start = betas[-1]
    elif all(c < target_coverage for c in covs):
        start = betas[0]
    else:
        gaps = [abs(c - target_coverage) for c in covs]
        start = betas[gaps.index(min(gaps))]

    ordered = sorted(seen)
    return max(start, beta_min), slope, bracket, {
        "betas": ordered,
        "coverages": [seen[b] for b in ordered],
    }