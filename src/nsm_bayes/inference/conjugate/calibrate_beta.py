"""Beta calibration (general posterior calibration) for NSM-Bayes."""

import math
from collections.abc import Callable

import torch
from scipy.stats import chi2
from tqdm import tqdm

from nsm_bayes.models.conjugate.conj_posterior import ConjugatePosterior


def _mahalanobis_sq_point(
    theta: torch.Tensor,
    mu: torch.Tensor,
    Sigma_inv: torch.Tensor,
) -> torch.Tensor:
    """Squared Mahalanobis distance between a point and a Gaussian.

    Parameters
    ----------
    theta : torch.Tensor
        Point of shape ``(d,)``.
    mu : torch.Tensor
        Mean of shape ``(d,)``.
    Sigma_inv : torch.Tensor
        Inverse covariance of shape ``(d, d)``.

    Returns
    -------
    torch.Tensor
        Scalar tensor with the squared distance.
    """
    diff = theta - mu
    return (diff @ Sigma_inv * diff).sum()


@torch.no_grad()
def calibrate_beta_gpc(
    x_obs: torch.Tensor,
    posterior: ConjugatePosterior,
    # --- GPC Hyperparameters ---
    initial_beta: float = 0.1,
    target_coverage: float = 0.95,
    num_iterations: int = 50,
    num_bootstraps: int = 200,
    learning_rate_fn: Callable[[int], float] = lambda t: 10.0 / (t + 10),
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
) -> tuple[float, dict]:
    """Calibrate ``beta`` using the General Posterior Calibration (GPC).

    Parameters
    ----------
    x_obs : torch.Tensor
        Observed data of shape ``(n, d_x)`` in the original (unnormalised)
        feature scale.
    posterior : ConjugatePosterior
        A configured ``ConjugatePosterior`` holding the trained
        ``ConjugateModel`` and the prior. Used to evaluate the posterior
        for any bootstrap resample.
    initial_beta : float, default 0.1
        Starting value for ``beta``.
    target_coverage : float, default 0.95
        Desired frequentist coverage of the beta-credible region.
    num_iterations : int, default 50
        Number of stochastic-approximation iterations.
    num_bootstraps : int, default 200
        Number of bootstrap resamples used per iteration to estimate the
        empirical coverage.
    learning_rate_fn : callable, default ``lambda t: 10 / (t + 10)``
        Step-size schedule ``k_t`` for the log-beta update.
    max_log_step : float, default 0.25
        Absolute cap on ``k_t * (coverage error)`` per iteration.
    beta_min : float, default 0.01
        Lower bound on ``beta`` to avoid collapse.

    Returns
    -------
    float
        The calibrated value of ``beta``.
    dict
        History with keys ``"betas"`` and ``"coverages"``.

    Notes
    -----
    All data is handled in the original scale: ``ConjugatePosterior``
    normalises internally and returns the posterior in original units, so
    the bootstrap resample and ``theta_hat_true`` live on the same scale.
    """
    print("--- Starting General Posterior Calibration for beta ---")

    # --- Device and Dimension Setup ---
    device = x_obs.device
    d_theta = posterior.prior_mean.shape[0]
    n_obs = x_obs.shape[0]

    # --- Step 0: Establish the "Ground Truth" Parameter for the Bootstrap ---
    theta_hat_true = posterior.compute_theta_hat_true(x_obs)
    print("Estimated true theta: ", theta_hat_true.detach())
    
    # --- Start the Iterative Calibration Process ---
    beta = initial_beta
    log_beta = math.log(initial_beta)
    log_beta_min = math.log(beta_min)

    history = {'betas': [], 'coverages': []}
    
    # The critical value from the Chi-Squared distribution for the credible region
    chi2_critical_value = chi2.ppf(target_coverage, df=d_theta)

    for t in range(num_iterations):
        coverage_counter = 0

        # --- Step 1: Estimate Empirical Coverage via Bootstrap ---
        iterable = tqdm(
            range(num_bootstraps),
            desc=f"Iter {t + 1}/{num_iterations} | beta={beta:.4f}",
        )
        for _ in iterable:
            # Resample rows with replacement
            indices = torch.randint(0, n_obs, (n_obs,), device=device)
            x_boot = x_obs[indices]

            # Posterior evaluated via ConjugatePosterior (original scale)
            mu_boot, Sigma_inv_boot = posterior.calibrated_posterior(x_boot, beta)

            # Check if theta_hat_true falls inside the credible region
            mahalanobis_sq = _mahalanobis_sq_point(theta_hat_true, mu_boot, Sigma_inv_boot)
            if mahalanobis_sq.item() <= chi2_critical_value:
                coverage_counter += 1

        # --- Step 2: Evaluate the Coverage ---
        empirical_coverage = coverage_counter / num_bootstraps

        # --- Step 3: Update Beta using Stochastic Approximation ---
        k_t = learning_rate_fn(t)
        log_step = k_t * (empirical_coverage - target_coverage)
        log_step = max(-max_log_step, min(max_log_step, log_step))
        log_beta += log_step

        # Clamp beta to avoid collapse
        log_beta = max(log_beta, log_beta_min)
        beta = math.exp(log_beta)

        # --- Logging and History ---
        print(
            f"Iter {t + 1}/{num_iterations} | Current Beta: {beta:.4f} | "
            f"Empirical Coverage: {empirical_coverage:.3f} | "
            f"Target: {target_coverage}"
        )
        history["betas"].append(beta)
        history["coverages"].append(empirical_coverage)

    print(f"Final calibrated beta: {beta:.4f}")
    return beta, history
