"""Shared stochastic-approximation loop for beta calibration.

:func:`calibrate_beta` iterates this update on ``log beta`` at each
iteration::

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
``prepare_calibration`` / ``estimate_coverage`` methods.
"""

import math
import torch
from collections.abc import Callable

__all__ = ["calibrate_beta"]


def calibrate_beta(
    posterior,
    x_obs: torch.Tensor | None = None,
    beta_init: float=0.01,
    alpha: float = 0.05,
    B: int = 200,
    n_iterations: int = 10,
    step_schedule: Callable[[int], float] = lambda t: 10.0 / (t + 10),
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
    ess_threshold: float = 0.3,
) -> tuple[float, dict]:
    """Run the shared SA loop over ``log beta``.

    Parameters
    ----------
    posterior : GeneralPosterior or ConjugatePosterior
        Posterior whose ``prepare_calibration`` / ``estimate_coverage``
        methods estimate the bootstrap coverage (and, for the general
        method, refresh the MCMC pool when the importance weights
        degenerate).
    x_obs : torch.Tensor or None
        Observed data of shape ``(n, d_x)``; used by the conjugate
        method's coverage estimate, ignored by the general method.
    beta_init : float, default 0.01
        Starting value of ``beta``; for the general method it is also the
        value at which the base MCMC pool is drawn.
    alpha : float, default 0.05
        Miscoverage level; the coverage target is ``1 - alpha``.
    B : int, default 200
        Number of bootstrap resamples per iteration.
    n_iterations : int, default 10
        Number of stochastic-approximation iterations.
    step_schedule : callable, default ``lambda t: 10.0 / (t + 10)``
        Step-size schedule ``k_t`` for the update of ``log beta``; called
        with the zero-based iteration index.
    max_log_step : float, default 0.25
        Absolute cap on the per-iteration change in ``log beta``.
    beta_min : float, default 0.01
        Lower bound on ``beta``; ``log beta`` is clamped at ``log(beta_min)``.
    ess_threshold : float, default 0.3
        ESS-gated pool refresh trigger (general method only).

    Returns
    -------
    beta : float
        The calibrated value of ``beta`` after the final iteration.
    history : dict
        ``{"betas": list[float], "coverages": list[float]}``: the value of
        ``beta`` after each update and the empirical coverage that drove it.
    """
    target = 1 - alpha
    log_beta = math.log(beta_init)
    log_beta_min = math.log(beta_min)

    # One-time setup: theta_hat (and, for the general method, the base pool
    # -- drawn here at beta_init -- and its cached losses).
    posterior.prepare_calibration(beta_init, alpha=alpha,
                                    B=B,
                                    x_obs=x_obs,
                                    ess_threshold=ess_threshold)

    history: dict = {
        "betas": [],
        "coverages": [],
    }

    for t in range(n_iterations):
        beta = math.exp(log_beta)

        # --- Step 1: estimate coverage at the current beta ---
        # depending on the posterior, will also check for effective sample size
        c_hat = posterior.estimate_coverage(beta, alpha=alpha,
                                            B=B, x_obs=x_obs,
                                            ess_threshold=ess_threshold)

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
