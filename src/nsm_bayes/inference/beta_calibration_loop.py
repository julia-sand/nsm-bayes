"""Pluggable beta-calibration strategies.

:func:`nsm_bayes.inference.beta_calibration_loop.calibrate_beta` dispatches
on its ``method`` argument to the update rules in
:mod:`nsm_bayes.inference.beta_update_rules`: the stochastic-approximation
loop (``gpc``), bisection in ``log(beta)`` (``bisect``) and the non-iterative
grid search (``grid``).  Array-valued ``initial_beta`` runs a grid search
first (log-bisection refinement of the coverage crossing), which feeds the
start and bracket of the iterative rules; it is only available on posteriors
that declare ``supports_grid_search`` (the conjugate method).
"""
import math
import torch
from typing import Any

from nsm_bayes.inference.beta_update_rules import (
    grid_start,
    run_grid_search,
    step_bisect,
    step_gpc,
)

__all__ = ["calibrate_beta"]

# Methods dispatchable by :func:`calibrate_beta`.
_METHODS = ("gpc", "bisect", "grid")


def _validate_initial_beta(initial_beta: Any) -> tuple[bool, list[float] | None, float | None]:
    """Parse ``initial_beta`` into ``(is_array, grid, scalar_start)``.

    A scalar number becomes the start directly; a 1-D numpy array / tensor
    is the grid for a start search; lists and tuples are rejected with
    ``TypeError`` (matching the historical experimental interface).  Grid
    values must be positive.

    Parameters
    ----------
    initial_beta : float or 1-D array
        Scalar starting ``beta``, or a grid of positive ``beta`` values.

    Returns
    -------
    is_array : bool
        Whether `initial_beta` was an array (grid search needed).
    grid : list of float or None
        The grid in increasing order (``None`` for a scalar start).
    scalar_start : float or None
        The scalar start (``None`` for the array case; it is only known
        after the grid search).
    """
    if isinstance(initial_beta, (list, tuple)):
        raise TypeError(
            "initial_beta must be a number or a numpy array / tensor, "
            "e.g. np.array([0.01, 0.1, 1.0]); lists and tuples are not accepted"
        )
    values = torch.as_tensor(initial_beta, dtype=torch.float64)
    if values.ndim == 0:
        start = float(values)
        if start <= 0:
            raise ValueError("initial_beta must be positive")
        return False, None, start
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("initial_beta array must be a non-empty 1-D array")
    grid = sorted(float(b) for b in values.tolist())
    if grid[0] <= 0:
        raise ValueError("initial_beta grid must contain positive values")
    return True, grid, None


def _validate_method(
    method: str,
    posterior: Any,
    is_array: bool,
) -> None:
    """Validate the ``method`` / ``initial_beta`` / posterior combination.

    Rules: the general (normflow) method supports ``gpc`` only, with a
    scalar start; posteriors with ``supports_grid_search = True`` (the
    conjugate method) accept all four methods, where ``bisect`` and
    ``grid`` additionally require an array start (they consume the grid
    and bracket it supplies).
    """
    if method not in _METHODS:
        raise ValueError(
            f"unknown calibration method {method!r}; expected one of {_METHODS}"
        )
    supports_grid = bool(getattr(posterior, "supports_grid_search", False))
    if not supports_grid:
        if is_array:
            raise ValueError(
                "array initial_beta (grid search) is only supported by posteriors "
                f"with supports_grid_search=True, not {type(posterior).__name__}"
            )
        if method != "gpc":
            raise ValueError(
                f"method={method!r} is not supported by {type(posterior).__name__}; "
                "only 'gpc' is"
            )
    if method == "grid" and not is_array:
        raise ValueError("method='grid' requires an array initial_beta (the grid)")
    if method == "bisect" and not is_array:
        raise ValueError(
            "method='bisect' requires an array initial_beta (it supplies the bracket)"
        )

def _iterative_loop(
    posterior: Any,
    x_obs: torch.Tensor | None,
    log_beta: float,
    bracket: tuple[float, float] | None = None,
    *,
    method: str,
    target: float,
    B: int,
    n_iterations: int,
    alpha: float,
    max_log_step: float,
    beta_min: float,
    ess_threshold: float,
) -> tuple[float, list[float], list[float], tuple[float, float] | None]:
    """Run the shared iteration loop under the selected update rule.

    Each iteration estimates the coverage at the current ``beta`` through
    ``posterior.estimate_coverage`` (which, for the general method, also
    performs the ESS-gated pool refresh) and hands the coverage to the
    rule, which returns the updated ``log(beta)``.

    Parameters
    ----------
    posterior, x_obs, B, alpha, ess_threshold
        Forwarded to ``posterior.estimate_coverage``.
    log_beta : float
        Starting ``log(beta)``.
    bracket : tuple of float or None
        Bracket ``(lo, hi)`` in ``log(beta)``, required by ``bisect``.
    method : str
        One of ``"gpc"``, ``"bisect"`` (dispatches to the matching step
        function).
    target : float
        Coverage target ``1 - alpha``.
    n_iterations : int
        Number of loop iterations.
    max_log_step : float
        Per-iteration cap on ``|log(beta)|`` change (``gpc`` only).
    beta_min : float
        Floor applied to ``log(beta)`` by the rules.

    Returns
    -------
    beta : float
        ``exp(log_beta)`` after the final iteration.
    betas, coverages : list of float
        ``beta`` after each update and the coverage that drove it.
    final_bracket : tuple of float or None
        The final bracket after the last update (``bisect`` only).
    """
    step_fn = {"bisect": step_bisect}.get(method, step_gpc)

    betas: list[float] = []
    coverages: list[float] = []
    for t in range(n_iterations):
        beta = math.exp(log_beta)
        c_hat = posterior.estimate_coverage(
            beta, alpha=alpha, B=B, x_obs=x_obs, ess_threshold=ess_threshold
        )
        if method == "bisect":
            log_beta, bracket = step_fn(
                log_beta, c_hat, t,
                bracket=bracket,
                max_log_step=max_log_step,
                beta_min=beta_min,
                target_coverage=target,
            )
        else:
            log_beta = step_fn(
                log_beta, c_hat, t,
                max_log_step=max_log_step,
                beta_min=beta_min,
                target_coverage=target,
            )
        betas.append(math.exp(log_beta))
        coverages.append(c_hat)
    return math.exp(log_beta), betas, coverages, bracket


def calibrate_beta(
    posterior,
    x_obs: torch.Tensor | None = None,
    initial_beta: float | torch.Tensor = 0.01,
    method: str = "gpc",
    alpha: float = 0.05,
    B: int = 200,
    n_iterations: int = 10,
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
    ess_threshold: float = 0.3,
    refine_steps: int = 3,
    **kwargs: Any,
) -> tuple[float, dict]:
    """Calibrate ``beta`` so the bootstrap coverage matches ``1 - alpha``.

    Dispatches on `method` to the update rules in
    :mod:`nsm_bayes.inference.beta_update_rules`.  The per-iteration work
    (draw bootstrap counts, importance-reweight / closed-form resample,
    estimate coverage, and for the general method the ESS-gated refresh of
    its MCMC pool) is delegated to the posterior's
    ``prepare_calibration`` / ``estimate_coverage`` methods.

    Parameters
    ----------
    posterior : GeneralPosterior or ConjugatePosterior
        Posterior whose ``prepare_calibration`` / ``estimate_coverage``
        methods estimate the bootstrap coverage (and, for the general
        method, refresh the MCMC pool when the importance weights
        degenerate).  Posteriors that can run an array-start grid search
        declare ``supports_grid_search = True`` (the conjugate method).
    x_obs : torch.Tensor or None
        Observed data of shape ``(n, d_x)``; used by the conjugate
        method's coverage estimate, ignored by the general method.
    initial_beta : float or 1-D numpy array / tensor, default 0.01
        Scalar starting ``beta``; for the general method it is also the
        value at which the base MCMC pool is drawn.  A 1-D array grid-
        searches the start (log-bisection refinement of the coverage
        crossing, interpolated start, plus slope and bracket extraction);
        it requires a posterior with ``supports_grid_search = True``.
        Lists and tuples are rejected.
    method : {"gpc", "bisect", "grid"}, default "gpc"
        Calibration strategy:

        * ``gpc``: the stochastic-approximation loop,
          ``log_beta += clip(k_t (coverage - target), +/ -max_log_step)``
          with ``k_t = 10 / (t + 10)``, floored at ``log(beta_min)``.
        * ``bisect``: bisection in ``log(beta)`` on the bracket supplied by
          an array `initial_beta` (requires it).
        * ``grid``: non-iterative; coverage on the array `initial_beta`,
          closest point wins (requires it).
    alpha : float, default 0.05
        Miscoverage level; the coverage target is ``1 - alpha``.
    B : int, default 200
        Number of bootstrap resamples per coverage evaluation.
    n_iterations : int, default 10
        Number of iterations of the iterative rules (unused by ``grid``).
    max_log_step : float, default 0.25
        Absolute cap on the per-iteration change in ``log beta``
        (``gpc`` only).
    beta_min : float, default 0.01
        Lower bound on ``beta``; ``log beta`` is clamped at ``log(beta_min)``.
    ess_threshold : float, default 0.3
        ESS-gated pool refresh trigger (general method only).
    refine_steps : int, default 3
        Number of log-bisection refinement steps after the grid finds a
        crossing (array `initial_beta` only).

    Returns
    -------
    beta : float
        The calibrated value of ``beta``.
    history : dict
        Always contains ``{"betas": list[float], "coverages": list[float]}``
        (for ``grid``: the grid in increasing order and its coverages;
        for the iterative rules: the value after each update and the
        empirical coverage that drove it).  Extra keys, present only when
        an array ``initial_beta`` was given: ``"grid"`` (all evaluated
        points, grid plus refinement mids), ``"initial_beta"`` (the scalar
        start actually used) and, when a crossing was found, ``"slope"``
        and ``"bracket"`` (the final bracket ``(lo, hi)`` in beta units).

    Raises
    ------
    TypeError
        If `initial_beta` is a list or tuple, or on the old ``beta_init``
        / ``step_schedule`` keyword arguments.
    ValueError
        If `method` is unknown, or the method / ``initial_beta`` combination
        is invalid for `posterior`.
    """

    target = 1 - alpha

    for old in ("beta_init", "step_schedule"):
        if old in kwargs:
            raise TypeError(
                f"calibrate_beta() got an unexpected keyword argument {old!r} "
                "(renamed to initial_beta; step schedules are set by the method)"
            )

    #validate the inputs for compatibility methods
    is_array, grid, scalar_start = _validate_initial_beta(initial_beta)
    _validate_method(method, posterior, is_array)

    if is_array:
        # One-time setup before any coverage call: theta_hat (and, for the
        # general method, the base pool and its cached losses), drawn at the
        # first grid value.
        posterior.prepare_calibration(
            grid[0], alpha=alpha, B=B, x_obs=x_obs, ess_threshold=ess_threshold
        )

        # Grid search first: the refined start also supplies the slope and
        # bracket that the iterative rules read from the state.
        start, slope, bracket, grid_history = grid_start(
            grid,
            lambda beta: posterior.estimate_coverage(
                beta, alpha=alpha, B=B, x_obs=x_obs, ess_threshold=ess_threshold
            ),
            target_coverage=target,
            beta_min=beta_min,
            refine_steps=refine_steps,
        )
    else:
        start, slope, bracket, grid_history = scalar_start, None, None, None

        # One-time setup: theta_hat (and, for the general method, the base
        # pool -- drawn here at the scalar start -- and its cached losses).
        posterior.prepare_calibration(
            start, alpha=alpha, B=B, x_obs=x_obs, ess_threshold=ess_threshold
        )

    history: dict[str, Any] = {"betas": [], "coverages": []}
    if is_array:
        history["grid"] = grid_history
        history["initial_beta"] = start

    if method == "grid":
        beta, grid_hist = run_grid_search(
            grid,
            lambda beta: posterior.estimate_coverage(
                beta, alpha=alpha, B=B, x_obs=x_obs, ess_threshold=ess_threshold
            ),
            target_coverage=target,
            beta_min=beta_min,
        )
        history["betas"] = grid_hist["betas"]
        history["coverages"] = grid_hist["coverages"]
    else:
        beta, betas, coverages, final_bracket = _iterative_loop(
            posterior, x_obs, math.log(start), bracket,
            method=method, target=target, B=B, n_iterations=n_iterations,
            alpha=alpha, max_log_step=max_log_step, beta_min=beta_min,
            ess_threshold=ess_threshold,
        )
        history["betas"] = betas
        history["coverages"] = coverages
        if final_bracket is not None:
            history["bracket"] = (
                math.exp(final_bracket[0]), math.exp(final_bracket[1])
            )

    if slope is not None:
        history["slope"] = slope

    return beta, history
