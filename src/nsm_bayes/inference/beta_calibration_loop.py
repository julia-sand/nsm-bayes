"""Shared stochastic-approximation loop for beta calibration.

The unified calibrator
(:func:`nsm_bayes.inference.calibrate_beta.calibrate_beta`) iterates this
update on ``log beta`` at each iteration::

    cov      <- coverage at the current beta
    k_t      <- step_schedule(t)
    log_step <- clip(k_t * (cov - target), -max_log_step, +max_log_step)
    log_beta <- max(log_beta + log_step, log(beta_min))
    beta     <- exp(log_beta)

and logs ``beta`` (after the update) and ``cov`` into a shared history dict.
This module factors that loop out so the general and conjugate methods
cannot drift apart.

Each method's per-iteration work (draw bootstrap counts, importance-reweight
/ closed-form resample, estimate coverage, and for the general method the
ESS-gated refresh of its MCMC pool) lives in the posterior's
``estimate_coverage`` method and is passed in as the ``coverage_fn``
closure.
"""

from __future__ import annotations

import math
from collections.abc import Callable

__all__ = ["run_beta_calibration_loop"]


def run_beta_calibration_loop(
    coverage_fn: Callable[[float], float],
    n_iterations: int,
    beta_init: float,
    beta_min: float,
    target_coverage: float,
    step_schedule: Callable[[int], float] = lambda t: 10.0 / (t + 10),
    max_log_step: float = 0.25,
) -> tuple[float, dict]:
    """Run the shared SA loop over ``log beta``.

    Parameters
    ----------
    coverage_fn : callable
        ``beta -> float``; estimates the empirical bootstrap coverage at the
        given value of ``beta``. The caller performs all bootstrap
        resampling / importance reweighting / closed-form resampling inside
        this closure and returns the fraction of resamples whose credible
        region contains the point estimate.
    n_iterations : int
        Number of stochastic-approximation iterations.
    beta_init : float
        Starting value of ``beta``.
    beta_min : float
        Lower bound on ``beta``; ``log beta`` is clamped at ``log(beta_min)``.
    target_coverage : float
        Coverage target in ``(0, 1)`` (e.g. ``0.95``), i.e. ``1 - alpha``.
    step_schedule : callable, default ``lambda t: 10.0 / (t + 10)``
        Step-size schedule ``k_t`` for the update of ``log beta``; called
        with the zero-based iteration index.
    max_log_step : float, default 0.25
        Absolute cap on the per-iteration change in ``log beta``.

    Returns
    -------
    beta : float
        The calibrated value of ``beta`` after the final iteration.
    history : dict
        ``{"betas": list[float], "coverages": list[float]}``: the value of
        ``beta`` after each update and the empirical coverage that drove it.
    """
    target = target_coverage
    log_beta = math.log(beta_init)
    log_beta_min = math.log(beta_min)

    history: dict = {
        "betas": [],
        "coverages": [],
    }

    for t in range(n_iterations):
        beta = math.exp(log_beta)

        # --- Step 1: estimate coverage at the current beta ---
        c_hat = coverage_fn(beta)

        # --- Step 2: stochastic-approximation update for log beta ---
        k_t = float(step_schedule(t))
        log_step = k_t * (c_hat - target)
        log_step = max(-max_log_step, min(max_log_step, log_step))
        log_beta += log_step

        # clamp beta to avoid collapse to near-zero
        log_beta = max(log_beta, log_beta_min)

        # --- Step 3: record this iteration ---
        history["betas"].append(math.exp(log_beta))
        history["coverages"].append(c_hat)

    return beta, history
