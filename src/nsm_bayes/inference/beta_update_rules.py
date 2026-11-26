"""Pluggable update rules for the beta calibration loop.

Companion to :func:`nsm_bayes.inference.beta_calibration_loop.calibrate_beta`:
each rule takes the current ``log(beta)`` and the coverage estimated at the
current ``beta``, and returns the updated ``log(beta)``.  ``step_bisect``
additionally takes and returns the bracket (``lo, hi``) in ``log(beta)`` as a
keyword argument.  :func:`run_grid_search` is the non-iterative baseline:
coverage on a fixed grid, closest point wins, and :func:`grid_start` is its
companion that refines the grid crossing (for an interpolated start and a
bisection bracket).
"""
import math
from collections.abc import Callable


__all__ = [
    "step_gpc", "step_bisect",
    "best_beta_from_grid", "grid_start",
]


def _default_learning_rate(t: int) -> float:
    """``k_t = 10 / (t + 10)``, the library's default GPC gain."""
    return 10.0 / (t + 10)


def step_gpc(
    log_beta: float,
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
    ``10 / (t + 10)``.

    Parameters
    ----------
    log_beta : float
        Current ``log(beta)``.
    coverage : float
        Coverage estimated at the current ``beta``.
    t : int
        Zero-based iteration index.
    max_log_step : float
        Per-iteration cap on ``|log(beta)|`` change.
    beta_min : float
        Floor applied to ``log(beta)``.
    target_coverage : float
        Coverage target ``1 - alpha``.
    learning_rate_fn : callable or None
        Learning rate function ``t -> k_t``; defaults to ``10 / (t + 10)``.

    Returns
    -------
    float
        Updated ``log(beta)``.
    """
    k_t = _default_learning_rate(t) if learning_rate_fn is None else learning_rate_fn(t)
    log_step = max(-max_log_step, min(max_log_step, k_t * (coverage - target_coverage)))
    return max(log_beta + log_step, math.log(beta_min))


def step_bisect(
    log_beta: float,
    coverage: float,
    t: int,
    *,
    bracket: tuple[float, float],
    max_log_step: float,
    beta_min: float,
    target_coverage: float,
) -> tuple[float, tuple[float, float]]:
    """Bisection in ``log(beta)``, continuing the grid start's bracket.

    The coverage at the current ``beta`` replaces the bracket end on its side
    of the target (``lo`` if at or above it, ``hi`` if below), and the next
    ``beta`` is the bracket's midpoint, floored at ``log(beta_min)``.  Needs a
    bracket, set by an array ``initial_beta`` (see :func:`grid_start`).
    ``t`` / ``max_log_step`` are unused; kept for a uniform rule signature.

    Parameters
    ----------
    log_beta : float
        Current ``log(beta)``.
    coverage : float
        Coverage estimated at the current ``beta``.
    t : int
        Zero-based iteration index (unused).
    bracket : tuple of float
        Current bracket ``(lo, hi)`` in ``log(beta)`` with coverage at or
        above the target at ``lo`` and below it at ``hi``.
    max_log_step : float
        Unused (kept for a uniform rule signature).
    beta_min : float
        Floor applied to ``log(beta)``.
    target_coverage : float
        Coverage target ``1 - alpha``.

    Returns
    -------
    updated_log_beta : float
        ``0.5 * (lo + hi)``, floored at ``log(beta_min)``.
    updated_bracket : tuple of float
        The updated bracket ``(lo, hi)``.
    """
    del t, max_log_step
    lo, hi = bracket
    if coverage >= target_coverage:
        lo = log_beta
    else:
        hi = log_beta
    updated_bracket = (lo, hi)
    return max(0.5 * (lo + hi), math.log(beta_min)), updated_bracket


def best_beta_from_grid(
    history,
    target_coverage: float,
    beta_min: float,
) -> tuple[float, dict[str, list[float]]]:
    """Coverage on a fixed grid of ``beta``; return the point closest to the target.

    ``grid`` is a sequence of positive ``beta`` values (already sorted and
    validated by the caller).  Each value costs one ``coverage_fn(beta)``
    evaluation; ``coverage_fn`` returns the 3-tuple
    ``(coverage, ess, pool_size)`` and only the coverage is used for the
    selection.  Ties in ``|coverage - target|`` go to coverage at or above
    the target, then to the larger ``beta``.

    Returns
    -------
    beta : float
        The selected grid value, floored at ``beta_min``.
    history : dict
        ``{"betas": [...], "coverages": [...], "ess": [...]}`` with the grid
        in increasing order and its coverages and mean effective sample
        sizes.
    """
    
    covs = history["coverages"]
    grid = history["betas"]
    
    gaps = [abs(c - target_coverage) for c in covs]
    best = min(range(len(grid)), key=lambda i: (gaps[i], -covs[i], -grid[i]))
    return max(grid[best], beta_min)

def coverage_on_grid(grid,coverage_fn):

    covs, esss = [], []
    for b in grid:
        cov, ess, _ = coverage_fn(b)
        print(rf"For beta = {b}, Coverage is {cov}")
        covs.append(float(cov))
        esss.append(float(ess))

    return covs,esss

def grid_start(
    grid: list[float],
    coverage_fn: Callable[[float], tuple[float, float, int]],
    *,
    target_coverage: float,
    beta_min: float,
) -> tuple[float, float | None, tuple[float, float] | None, dict[str, list[float]]]:
    """Grid-search start for ``beta``.

    Evaluates the coverage at each ``beta`` in ``grid`` (already sorted and
    validated by the caller), the slope of the final bracket is returned as history
    information.  It returns the largest grid value
    (all above target), the smallest (all below) or the closest, and slope
    and bracket ``None``.

    ``coverage_fn`` returns the 3-tuple ``(coverage, ess, pool_size)``;
    only the coverage drives the search.

    Returns
    -------
    start : float
        Starting ``beta`` (interpolated crossing, or a grid endpoint /
        closest grid point), floored at ``beta_min``.
    slope : float or None
        ``d coverage / d log(beta)`` on the final refined bracket, or
        ``None`` without a crossing.
    bracket : tuple[float, float] or None
        The final refined bracket ``(log lo, log hi)``, or ``None`` without
        a crossing.
    grid_history : dict
        ``{"betas": [...], "coverages": [...], "ess": [...]}`` for every
        evaluated point in increasing order.
    """
    if not grid or grid[0] <= 0:
        raise ValueError("initial_beta grid must contain positive values")

    covs, esss = coverage_on_grid(grid,coverage_fn)
    
    seen = dict(zip(grid, covs))  # every evaluated beta -> coverage
    seen_ess = dict(zip(grid, esss))  # every evaluated beta -> ess

    i = None  # first j with covs[j] >= target > covs[j + 1]
    for j in range(len(grid) - 1):
        if covs[j] >= target_coverage > covs[j + 1]:
            i = j
            break

    slope = bracket = None
    if i is not None:
        lo, c_lo, hi, c_hi = grid[i], covs[i], grid[i + 1], covs[i + 1]
        frac = (c_lo - target_coverage) / (c_lo - c_hi)
        start = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        slope = (c_hi - c_lo) / (math.log(hi) - math.log(lo))
        bracket = (math.log(lo), math.log(hi))
    elif all(c >= target_coverage for c in covs):
        start = grid[-1]
        
    elif all(c < target_coverage for c in covs):
        start = grid[0]
    else:
        gaps = [abs(c - target_coverage) for c in covs]
        start = grid[gaps.index(min(gaps))]

    ordered = sorted(seen)
    return max(start, beta_min), slope, bracket, {
        "betas": ordered,
        "coverages": [seen[b] for b in ordered],
        "ess": [seen_ess[b] for b in ordered],
    }