"""Beta calibration for the general (normalising-flow) NSM-Bayes method.

The generalised posterior of
:class:`~nsm_bayes.models.general.posterior.GeneralPosterior` has no closed
form, so bootstrap posteriors are approximated by importance-reweighting one
set of MCMC samples (see :mod:`nsm_bayes.inference.bootstrap`), and the
samples are refreshed only when the weights degenerate.
"""

import math
from collections.abc import Callable

import torch

from nsm_bayes.inference.bootstrap import (
    bootstrap_coverage_indicator,
    multinomial_bootstrap_counts,
    reweight_for_bootstrap_beta,
)
from nsm_bayes.models.general.posterior import GeneralPosterior


def calibrate_beta(posterior: GeneralPosterior,
        theta_samples_base: torch.Tensor,   # (M, d) samples drawn at beta_base on original data
        beta_base: float,
        alpha: float = 0.05,
        B: int = 200,
        T: int = 10,
        beta_init: float | None = None,
        step_schedule: Callable[[int], float] = lambda t: 10.0 / (t + 10),
        max_log_step: float = 0.25,
        beta_min: float = 0.01,
        ess_threshold: float = 0.3,
        ) -> tuple[float, dict]:
    """Calibrate ``beta`` for the general posterior by bootstrap coverage matching.

    Parameters
    ----------
    posterior : GeneralPosterior
        Generalised posterior for the observed data. Supplies the
        per-observation losses, the point estimate ``theta_hat`` and fresh
        MCMC samples when the importance weights degenerate.
    theta_samples_base : torch.Tensor
        Samples from `posterior` at `beta_base` on the original data,
        shape ``(M, d)``.
    beta_base : float
        Value of ``beta`` at which `theta_samples_base` were drawn.
    alpha : float, default 0.05
        Miscoverage level; ``1 - alpha`` is the target coverage.
    B : int, default 200
        Number of bootstrap resamples per iteration.
    T : int, default 10
        Number of stochastic-approximation iterations.
    beta_init : float, optional
        Starting value of ``beta``. Defaults to `beta_base`.
    step_schedule : callable, default ``lambda t: 10.0 / (t + 10)``
        Step-size schedule ``k_t`` for the update of ``log(beta)``; called
        with the zero-based iteration index.
    max_log_step : float, default 0.25
        Absolute cap on the per-iteration change in ``log(beta)``.
    beta_min : float, default 0.01
        Lower bound on ``beta``, to stop it collapsing towards zero.
    ess_threshold : float, default 0.3
        If the mean effective sample size of the bootstrap weights falls
        below ``ess_threshold * M``, new samples are drawn at the current
        ``beta`` and become the new base.

    Returns
    -------
    beta_star : float
        The calibrated value of ``beta`` after the final iteration.
    history : dict
        ``{"betas": list[float], "coverages": list[float]}``: the value of
        ``beta`` after each update and the empirical coverage that drove it.

    See Also
    --------
    nsm_bayes.inference.conjugate.calibrate_beta.calibrate_beta_gpc :
        The equivalent routine for the conjugate method.

    Notes
    -----
    ``theta_hat`` is the NSM minimiser on the full data, found by
    :meth:`GeneralPosterior.compute_theta_hat_true` (200 Adam steps with
    learning rate ``1e-2``, started from the mean of `theta_samples_base`).
    It is held fixed throughout. Each iteration then:

    1. draws ``B`` multinomial bootstrap count vectors;
    2. importance-reweights the base samples from ``beta_base`` on the
       original data to the current ``beta`` on each resample
       (:func:`~nsm_bayes.inference.bootstrap.reweight_for_bootstrap_beta`);
    3. estimates coverage as the fraction of resamples whose weighted
       ``1 - alpha`` ellipsoid contains ``theta_hat``
       (:func:`~nsm_bayes.inference.bootstrap.bootstrap_coverage_indicator`);
    4. updates ``log(beta) += clip(k_t * (coverage - (1 - alpha)),
       -max_log_step, max_log_step)``, clamped at ``log(beta_min)``;
    5. if the mean ESS ``1 / sum_m W_bm^2`` is below
       ``ess_threshold * M``, redraws samples with ``posterior.sample`` at
       the updated ``beta`` and recomputes the loss cache.

    Progress is printed to stdout.
    """

    device = posterior.x_obs.device
    dtype  = posterior.x_obs.dtype

    M, d = theta_samples_base.shape
    n = posterior.x_obs.shape[0]

    # Cache per-x losses for all base samples, plus base average loss
    # ell:   (M, n)   per-x losses
    # L_orig:(M,)     mean loss on *original* data for each theta sample
    ell, L_orig = posterior.per_x_losses(theta_samples_base, )

    # theta_hat_true = argmin_theta L_NSM(theta; x_obs), i.e. NSM M-estimator
    theta_init = theta_samples_base.mean(dim=0)  # starting point
    theta_hat = posterior.compute_theta_hat_true(theta_init=theta_init,
                                                  num_steps=200,             # tune if needed
                                                  lr=1e-2,                  # tune if needed
                                                )

    # --- Initialize in Log-Space ---
    current_beta = beta_init if beta_init is not None else beta_base
    log_beta = math.log(current_beta)
    log_beta_min = math.log(beta_min)

    current_beta_base = beta_base 

    history = {
        "betas": [],
        "coverages": [],
    }

    for t in range(T):
        # --- Step 1: Get current beta for this iteration
        current_beta = math.exp(log_beta) #is this intentional?

        # --- Step 2: draw bootstrap count vectors (B, n)
        counts = multinomial_bootstrap_counts(n, B, device=device, dtype=dtype)

        # --- Step 3: importance weights for each bootstrap at the current beta
        # W has shape (B, M), one weight vector per bootstrap
        W = reweight_for_bootstrap_beta(ell, L_orig, counts, beta_new=current_beta, beta_base=current_beta_base)

        # --- Step 4: estimate coverage
        # For each bootstrap b, build weighted posterior (mu_b, Sigma_b) and check if theta_hat lies in its weighted (1-alpha) ellipsoid.
        cover = 0
        
        for b in range(B):
            cover += bootstrap_coverage_indicator(
                                                theta_samples_base,  # (M, d)
                                                W[b],                # (M,)
                                                theta_hat,           # (d,)
                                                alpha                # alpha, so target is 1-alpha
                                            )
        c_hat = cover / B  # empirical coverage at this beta

        # --- Step 5: stochastic approximation update for beta
        target = 1.0 - alpha
        k_t = float(step_schedule(t))
        log_step = k_t * (c_hat - target)
        log_step = max(-max_log_step, min(max_log_step, log_step))
        log_beta += log_step

        # clamp beta to avoid collapse to near-zero
        log_beta = max(log_beta, log_beta_min)

        # Update current_beta for logging and potential refresh
        current_beta = math.exp(log_beta)

        # --- Step 6: track effective sample size (ESS) to detect weight degeneracy
        # ESS per bootstrap b: 1 / sum_m W[b,m]^2
        ess = (W**2).sum(dim=1)
        ess = 1.0 / ess
        ess_mean = ess.mean().item()

        # If weights have collapsed and we can resample, rejuvenate MCMC at new beta
        if ess_mean < ess_threshold * M:
            theta_samples_base = posterior.sample(current_beta)  # (M, d)
            theta_samples_base = theta_samples_base.to(device=device, dtype=dtype)

            # recompute cached losses with the refreshed samples
            ell, L_orig = posterior.per_x_losses(theta_samples_base, )

            current_beta_base = current_beta

        # --- Step 7: log this iteration
        print(
            f"Iter {t+1}/{T} | beta={current_beta:.4f} | coverage={c_hat:.3f} "
            f"| target={(1.0 - alpha):.3f} | ESS_mean={ess_mean:.2f}"
        )
        history["betas"].append(float(current_beta))
        history["coverages"].append(float(c_hat))

    beta_star = math.exp(log_beta)
    print(f"[calibrate_beta] Finished. beta* = {beta_star:.4f}")
    return beta_star, history
