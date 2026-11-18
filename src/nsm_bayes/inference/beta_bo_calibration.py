"""Bayesian-optimisation beta calibration via a GP on a coverage grid.

Companion to :mod:`nsm_bayes.inference.beta_update_rules`.  Instead of
iterating a stochastic-approximation step, :func:`optimize_beta_bo` fits a
Gaussian process to pre-computed ``(beta, coverage)`` pairs and returns the
``beta`` where the GP mean crosses the target coverage.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


__all__ = ["optimize_beta_bo"]


def optimize_beta_bo(
    betas: list[float] | np.ndarray,
    coverages: list[float] | np.ndarray,
    *,
    target_coverage: float,
    beta_min: float = 0.01,
) -> float:
    """Fit a GP on ``(log beta, coverage)`` and return the beta where the GP
    mean crosses *target_coverage*.

    Parameters
    ----------
    betas : array-like of float
        Positive beta values at which coverage was evaluated.
    coverages : array-like of float
        Coverage values corresponding to each beta (same length as *betas*).
    target_coverage : float
        The desired coverage level (e.g. ``1 - alpha = 0.95``).
    beta_min : float
        Floor applied to the returned beta.

    Returns
    -------
    float
        The beta whose GP-mean coverage is closest to *target_coverage*,
        floored at *beta_min*.
    """
    betas = np.asarray(betas, dtype=np.float64)
    coverages = np.asarray(coverages, dtype=np.float64)

    if betas.ndim != 1 or coverages.ndim != 1:
        raise ValueError("betas and coverages must be 1-D arrays")
    if betas.shape[0] != coverages.shape[0]:
        raise ValueError("betas and coverages must have the same length")
    if betas.shape[0] < 2:
        raise ValueError("need at least two (beta, coverage) pairs")
    if np.any(betas <= 0):
        raise ValueError("all beta values must be positive")

    x = np.log(betas).reshape(-1, 1)
    y = coverages

    kernel = Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)
    gp.fit(x, y)

    # Dense evaluation over the log-beta range to find the crossing.
    x_min, x_max = float(x.min()), float(x.max())
    # Pad the range slightly so the crossing isn't clipped at the boundary.
    pad = 0.1 * (x_max - x_min) if x_max > x_min else 0.1
    eval_x = np.linspace(x_min - pad, x_max + pad, 2001).reshape(-1, 1)
    mean = gp.predict(eval_x, return_std=False)

    # Find where the GP mean crosses the target.
    diff = mean - target_coverage
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) == 0:
        # No crossing: return the beta whose GP-mean coverage is closest.
        best_idx = int(np.argmin(np.abs(diff)))
        best_log_beta = eval_x[best_idx, 0]
    else:
        # Use the first crossing; linear interpolation in (log beta, coverage).
        i = int(sign_changes[0])
        x0, x1 = eval_x[i, 0], eval_x[i + 1, 0]
        y0, y1 = mean[i], mean[i + 1]
        frac = (target_coverage - y0) / (y1 - y0)
        best_log_beta = x0 + frac * (x1 - x0)

    beta = math.exp(best_log_beta)
    return max(beta, beta_min)
