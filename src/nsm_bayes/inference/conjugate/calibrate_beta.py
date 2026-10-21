"""Beta calibration for the conjugate NSM-Bayes method.

Implements general posterior calibration (GPC; Syring and Martin, 2019) for
the closed-form Gaussian posterior of
:class:`~nsm_bayes.models.conjugate.conj_posterior.ConjugatePosterior`.
Because the posterior is available analytically, every bootstrap resample is
evaluated exactly; no importance reweighting is needed (compare
:func:`nsm_bayes.inference.general.calibrate.calibrate_beta`).
"""

import math
from collections.abc import Callable

import torch
from scipy.stats import chi2
from tqdm import tqdm

from nsm_bayes.models.conjugate.conj_posterior import ConjugatePosterior
from nsm_bayes.utils.statistics import mahalanobis_sq

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
    """Calibrate ``beta`` for the conjugate posterior by GPC.

    ``beta`` is tuned so that the ``target_coverage`` credible ellipsoid of
    the calibrated posterior covers the point estimate ``theta_hat`` with
    the same frequency across bootstrap resamples of `x_obs`.

    Parameters
    ----------
    x_obs : torch.Tensor
        Observed data of shape ``(n, d_x)`` in the original (unstandardised)
        scale.
    posterior : ConjugatePosterior
        Posterior wrapping the trained
        :class:`~nsm_bayes.models.conjugate.conj_model.ConjugateModel` and
        the prior; evaluated on every bootstrap resample.
    initial_beta : float, default 0.1
        Starting value of ``beta``.
    target_coverage : float, default 0.95
        Desired frequentist coverage of the credible region.
    num_iterations : int, default 50
        Number of stochastic-approximation iterations.
    num_bootstraps : int, default 200
        Number of bootstrap resamples per iteration used to estimate the
        empirical coverage.
    learning_rate_fn : callable, default ``lambda t: 10 / (t + 10)``
        Step-size schedule ``k_t`` for the update of ``log(beta)``; called
        with the zero-based iteration index.
    max_log_step : float, default 0.25
        Absolute cap on the per-iteration change in ``log(beta)``.
    beta_min : float, default 0.01
        Lower bound on ``beta``, to stop it collapsing towards zero.

    Returns
    -------
    beta : float
        The calibrated value of ``beta`` after the final iteration.
    history : dict
        ``{"betas": list[float], "coverages": list[float]}``: the value of
        ``beta`` after each update and the empirical coverage that drove it.

    See Also
    --------
    nsm_bayes.inference.general.calibrate.calibrate_beta :
        The equivalent routine for the general (normalising-flow) method.

    Notes
    -----
    Each iteration proceeds as follows:

    1. Draw ``num_bootstraps`` resamples of the rows of `x_obs` with
       replacement and compute the posterior ``N(mu_b, Sigma_b)`` for each
       at the current ``beta``.
    2. Count how often ``theta_hat`` falls inside the ellipsoid
       ``(theta - mu_b)^T Sigma_b^{-1} (theta - mu_b) <= chi2_{d}(target)``,
       where ``d = d_theta``. This gives the empirical coverage ``c_t``.
    3. Update ``log(beta) += clip(k_t * (c_t - target_coverage),
       -max_log_step, max_log_step)`` and clamp at ``log(beta_min)``.

    Over-coverage therefore increases ``beta`` (sharper posterior) and
    under-coverage decreases it.  ``theta_hat`` is
    :meth:`ConjugatePosterior.compute_theta_hat_true` on the full data and
    is held fixed throughout.

    All quantities are in the original scale: ``ConjugatePosterior``
    standardises internally and returns its moments in original units.
    Progress is printed to stdout.

    References
    ----------
    .. [1] N. Syring and R. Martin, "Calibrating general posterior credible
       regions", *Biometrika*, 106(2), 479-486, 2019.
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
            mu_boot, Sigma_boot = posterior.calibrated_posterior(x_boot, beta)

            # Check if theta_hat_true falls inside the credible region
            mahalanobis_sq_val = mahalanobis_sq(theta_hat_true, mu_boot, Sigma_boot)
            if mahalanobis_sq_val.item() <= chi2_critical_value:
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
