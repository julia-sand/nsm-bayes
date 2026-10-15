"""Beta calibration (general posterior calibration) for NSM-Bayes."""

import math
from collections.abc import Callable

import torch
from scipy.stats import chi2
from torch.func import vmap
from tqdm import tqdm

from nsm_bayes.inference.general.train import compute_theta_hat_true_general
from nsm_bayes.models.conjugate.conj_posterior import compute_posterior_conjugate
from nsm_bayes.utils.linalg import (
    get_cov_term_single,
    get_div_term_single,
    get_non_div_term_single,
    jacobian_and_hessian,
    sigma_inverse,
)
from nsm_bayes.utils.statistics import estimate_stats
from nsm_bayes.utils.weightfuns import imq_weight, weight_function

# Small constant that only guards divisions and logs from 0/NaN.
EPS = 1e-12


def cache_perx_sm_losses(
    theta_samples: torch.Tensor,           # (M, d_theta)
    x_obs: torch.Tensor,                   # (n, d_x)
    q_phi_log_prob,                        # fn (x, theta) -> scalar
    w2: torch.Tensor,                      # (n,)
    grad_w2: torch.Tensor,                 # (n, d_x)
    chunk: int = 128
):
    """
    Returns:
      ell:   (M, n) tensor with per-x SM loss contributions for each theta sample.
      L_orig:(M,)  average SM loss on the original data.
    """
    device = x_obs.device
    dtype  = x_obs.dtype
    M, d_theta = theta_samples.shape
    n, d_x     = x_obs.shape

    ell = torch.empty((M, n), device=device, dtype=dtype)

    # vmap single-x loss for a fixed theta
    def per_x_loss_for_theta(xi, w2_i, grad_w2_i, theta):
        # log q(x | theta)
        log_q_fn = lambda x_vec: q_phi_log_prob(x_vec.unsqueeze(0), theta.unsqueeze(0)).sum()
        
        score, hess = jacobian_and_hessian(xi,log_q_fn)
        
        term1 = w2_i * (score @ score)
        term2 = 2.0 * grad_w2_i @ score
        if hess.ndim == 0:
            term3 = 2.0 * w2_i * hess
        else:
            term3 = 2.0 * w2_i * torch.trace(hess)
        return term1 + term2 + term3

    # vmap over x for a fixed theta
    def losses_for_one_theta(theta):
        return vmap(
            per_x_loss_for_theta,
            in_dims=(0, 0, 0, None)
        )(x_obs, w2, grad_w2, theta)

    # chunk over theta to control memory
    for s in range(0, M, chunk):
        tchunk = theta_samples[s:s+chunk]
        vals = [losses_for_one_theta(th) for th in tchunk]  # list of (n,)
        ell[s:s+chunk] = torch.stack(vals, dim=0)

    L_orig = ell.mean(dim=1)  # (M,)
    return ell, L_orig

def multinomial_bootstrap_counts(n: int, B: int, device=None, dtype=None):
    # Draw B multinomial(n, p_i=1/n) vectors of counts (B, n)
    p = torch.full((n,), 1.0/n, device=device, dtype=dtype or torch.float32)
    counts = torch.distributions.Multinomial(total_count=n, probs=p).sample((B,))
    return counts.to(dtype=torch.float32)

def reweight_for_bootstrap_beta(
    ell: torch.Tensor,        # (M, n)
    L_orig: torch.Tensor,     # (M,)
    counts: torch.Tensor,     # (B, n)
    beta_new: float,
    beta_base: float,
) -> torch.Tensor:
    """
    Returns weights W of shape (B, M) for each bootstrap b and sample m.
    """
    M, n = ell.shape
    B = counts.shape[0]

    # >>> ensure same device & dtype as ell
    counts = counts.to(device=ell.device, dtype=ell.dtype)

    # (B, M) = (B, n) @ (n, M) / n
    L_b = (counts @ ell.T) / n

    expo = - (beta_new * n) * L_b + (beta_base * n) * L_orig.unsqueeze(0)
    expo = expo - expo.max(dim=1, keepdim=True).values
    W = torch.exp(expo)
    W = W / (W.sum(dim=1, keepdim=True) + EPS)
    return W

def weighted_mean_cov(
    theta_samples: torch.Tensor,
    w: torch.Tensor,
):
    """
    Weighted mean and covariance of samples.

    Parameters
    ----------
    theta_samples : torch.Tensor
        Samples of shape ``(M, d)``.
    w : torch.Tensor
        Non-negative weights of shape ``(M,)``.
    
    Returns
    -------
    mu : torch.Tensor
        Weighted mean of shape ``(d,)``.
    Sigma : torch.Tensor
        Weighted covariance of shape ``(d, d)``.
    """
    w = w.view(-1, 1)
    mu = (w * theta_samples).sum(dim=0)
    xc = theta_samples - mu
    # weighted covariance (posterior convention: no Bessel correction)
    Sigma = (w * xc).T @ xc / (w.sum() + EPS)
    
    return mu, Sigma

def weighted_mahalanobis_sq(theta_samples: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor):
    xc = theta_samples - mu
    return torch.sum(xc @ Sigma_inv * xc, dim=1)  # (M,)

def weighted_quantile(values: torch.Tensor, w: torch.Tensor, q: float):
    """
    Weighted quantile in [0,1] for 1D tensor 'values' with nonnegative weights 'w'.
    """
    v, idx = torch.sort(values)
    w_sorted = w[idx]
    cdf = torch.cumsum(w_sorted, dim=0)
    cdf = cdf / (cdf[-1] + EPS)
    return v[torch.searchsorted(cdf, torch.tensor(q, device=v.device))]

def bootstrap_coverage_indicator(
        theta_samples: torch.Tensor,   # (M, d)
        weights_row: torch.Tensor,     # (M,)
        theta_hat: torch.Tensor,       # (d,)
        alpha: float
        ) -> int:   
    w = weights_row / (weights_row.sum() + EPS)
    mu, Sigma = weighted_mean_cov(theta_samples, w)
    Sigma_inv = sigma_inverse(Sigma)
    d2 = weighted_mahalanobis_sq(theta_samples, mu, Sigma_inv)
    thresh = weighted_quantile(d2, w, 1.0 - alpha)
    d2_hat = (theta_hat - mu).unsqueeze(0) @ Sigma_inv @ (theta_hat - mu).unsqueeze(1)
    return int(d2_hat.item() <= thresh.item())

def calibrate_beta(
    theta_samples_base: torch.Tensor,   # (M, d) samples drawn at beta_base on original data
    beta_base: float,
    x_obs: torch.Tensor,                # (n, d_x)
    q_phi_log_prob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mu_hat: torch.Tensor,
    Sigma_inv: torch.Tensor,
    weight_type: str,
    c: float = 1.0,
    alpha: float = 0.05,
    B: int = 200,
    T: int = 10,
    beta_init: float | None = None,
    step_schedule: Callable[[int], float] = lambda t: 10.0 / (t + 10),
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
    ess_threshold: float = 0.3,
    refresh_sampler: Callable[[float], torch.Tensor] | None = None,
    ) -> tuple[float, dict]:
    """
    Calibrate beta via bootstrap coverage matching for NSM-Bayes.

    Parameters
    ----------
    theta_samples_base : torch.Tensor
        Samples drawn at `beta_base` on the original data, shape ``(M, d)``.
    beta_base : float
        Beta value the base samples were drawn at.
    x_obs : torch.Tensor
        Observed data, shape ``(n, d_x)``.
    q_phi_log_prob : callable
        ``(x, theta) -> scalar`` log-density of the estimator.
    mu_hat : torch.Tensor
        Robust location of the data, shape ``(d_x,)``.
    Sigma_inv : torch.Tensor
        Inverse robust covariance, shape ``(d_x, d_x)``.
    weight_type : str
        Name of the weight function, e.g. ``"imq"`` or ``"none"``.
    c : float, default 1.0
        Scale parameter of the IMQ weight.
    alpha : float, default 0.05
        Significance level; ``1 - alpha`` is the target coverage.
    B : int, default 200
        Number of bootstraps per iteration.
    T : int, default 10
        Number of stochastic approximation iterations.
    beta_init : float or None, default None
        Starting beta; falls back to `beta_base` when None.
    step_schedule : callable, default lambda t: 10.0 / (t + 10)
        Step size ``k_t`` used for the log-beta update at iteration ``t``.
    max_log_step : float, default 0.25
        Cap on ``k_t * (coverage error)`` per log-beta update.
    beta_min : float, default 0.01
        Lower bound for beta to prevent collapse.
    ess_threshold : float, default 0.3
        ESS-to-M ratio below which samples may be refreshed.
    refresh_sampler : callable or None, default None
        ``sampler(beta) -> (M, d)`` used to rejuvenate collapsed samples.

    Returns
    -------
    beta_star : float
        Final calibrated beta.
    history : dict
        Dictionary with 'betas' and 'coverages' over iterations.
    Notes
    -----
    - We importance-reweight the original MCMC samples from beta_base to get
      approximate posteriors at new beta values, for each bootstrap resample.
    - If weights degenerate (low ESS) and refresh_sampler is provided, we rejuvenate:
      resample theta at the current beta and rebuild loss cache.
    - theta_hat (the pseudo-true parameter) stays fixed as the mean at beta_base.
    """

    device = x_obs.device
    dtype  = x_obs.dtype

    M, d = theta_samples_base.shape
    n = x_obs.shape[0]

    # Precompute IMQ weights (or "none") once
    w2, grad_w2 = weight_function(weight_type, x_obs, mu_hat, Sigma_inv, c)
    
    # Cache per-x losses for all base samples, plus base average loss
    # ell:   (M, n)   per-x losses
    # L_orig:(M,)     mean loss on *original* data for each theta sample
    ell, L_orig = cache_perx_sm_losses(theta_samples_base, x_obs, q_phi_log_prob, w2, grad_w2)

    # theta_hat_true = argmin_theta L_NSM(theta; x_obs), i.e. NSM M-estimator
    theta_init = theta_samples_base.mean(dim=0)  # starting point
    theta_hat = compute_theta_hat_true_general(
        x_obs=x_obs,
        q_phi_log_prob=q_phi_log_prob,
        w2=w2,
        grad_w2=grad_w2,
        theta_init=theta_init,
        num_steps=200,       # tune if needed
        lr=1e-2,             # tune if needed
        verbose=False,
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
        current_beta = math.exp(log_beta)

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
        ess = 1.0 / (ess + EPS)
        ess_mean = ess.mean().item()

        # If weights have collapsed and we can resample, rejuvenate MCMC at new beta
        if ess_mean < ess_threshold * M and refresh_sampler is not None:
            theta_samples_base = refresh_sampler(current_beta)  # (M, d)
            theta_samples_base = theta_samples_base.to(device=device, dtype=dtype)

            # recompute cached losses with the refreshed samples
            ell, L_orig = cache_perx_sm_losses(theta_samples_base, x_obs, q_phi_log_prob, w2, grad_w2)

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



######-------Function needed for general posterior calibration (setting the learning rate) for NSM-Bayes-conj------#########

def compute_theta_hat_true_conjugate(
    x_obs: torch.Tensor,
    model,
    weight_fn,
    mu_hat_obs: torch.Tensor,
    Sigma_inv_obs: torch.Tensor,
    c: float,
) -> torch.Tensor:
    """
    Compute the NSM-only minimiser theta_hat_true for NSM-Bayes-conj:
        theta_hat_true = argmin_theta L_NSM(theta; x_obs, phi_hat)

    This corresponds to the M-estimator used as the "true" parameter
    in the Syring & Martin calibration paper.

    Parameters
    ----------
    x_obs : torch.Tensor
        (n, d_x) observed data (already normalized if you use standardizer_x).
    model : object
        Model exposing ``model.T_phi`` (network mapping x to the natural
        parameters) and ``model.b_phi`` (network mapping x to the
        log-partition term).
    weight_fn : callable
        Function ``w^2(x)`` used elsewhere.
    mu_hat_obs : torch.Tensor
        Robust location of the data.
    Sigma_inv_obs : torch.Tensor
        Inverse robust covariance of the data.
    c : float
        Scale parameter of the IMQ weight.

    Returns
    -------
    torch.Tensor
        ``theta_hat_true`` of shape ``(d_theta,)`` in the same (normalized)
        theta space that T_phi produces.
    """

    # IMQ weight w^2(x) and its analytic gradient (n,) and (n, d_x)
    w2, grad_w2 = weight_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c)

    
    # A = sum_i w2_i * J_T(x_i)^T J_T(x_i)   (d_theta, d_theta)
    # The networks are not batched: vmap them with in_dim=None. get_cov_term_
    # single(xi, fn) returns (J_T^T J_T, J_T); only the covariance term is kept.
    batched_cov_terms = vmap(get_cov_term_single, in_dims=(0, None))(
        x_obs, model.T_phi
    )  # (n, d_theta, d_theta)
    A = (w2.view(-1, 1, 1) * batched_cov_terms).sum(dim=0)  # (d_theta, d_theta)

    # B = sum_i [ w2_i * div_theta log p(x_i | theta)
    #             + grad_x w2(x_i)^T score_x(x_i) ]   (d_theta,)
    # get_non_div_term_single(xi, fn1, fn2, Jac1, Jac2) = Jac1 @ Jac2; both
    # Jacobians are recomputed per point and w2 is applied outside, since this
    # helper takes no weights.
    batched_non_div_terms = vmap(
        get_non_div_term_single, in_dims=(0, None, None)
    )(x_obs, model.T_phi, model.b_phi)  # (n, d_theta)
    term_non_div = (w2.view(-1, 1) * batched_non_div_terms).sum(dim=0)  # (d_theta,)

    # term_div = sum_i div_x[ w(x_i)^2 * score_x(x_i) ]  (vector in R^{d_theta})
    # get_div_term_single(xi, model, w2, grad_w2) reads model.T_phi itself and
    # weights by w2 inside, so it is NOT re-applied when reducing below.
    batched_divergences = vmap(get_div_term_single, in_dims=(0, None, 0, 0))(
        x_obs, model.T_phi, w2, grad_w2
    )  # (n, d_theta)
    term_div = batched_divergences.sum(dim=0)  # (d_theta,)

    B = term_non_div + term_div  # (d_theta,)

    # ---------- 3. Solve A * theta_hat_true = -B ----------
    d_theta = A.shape[0]
    scale = torch.trace(A) / A.shape[0]
    lam = 1e-2 * scale + EPS
    A_reg = 0.5 * (A + A.T) + lam * torch.eye(d_theta, device=A.device, dtype=A.dtype)

    try:
        L = torch.linalg.cholesky(A_reg)
        theta_hat_true = -torch.cholesky_solve(B.unsqueeze(1), L).squeeze(1)
    except RuntimeError:
        theta_hat_true = -torch.linalg.lstsq(A_reg, B).solution

    return theta_hat_true

@torch.no_grad()
def calibrate_beta_gpc(
    x_obs: torch.Tensor,
    model,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    # --- GPC Hyperparameters ---
    initial_beta: float = 0.1,
    target_coverage: float = 0.95,
    num_iterations: int = 50,
    num_bootstraps: int = 200,
    c: float = 1.0,
    learning_rate_fn = lambda t: 10.0 / (t + 10),
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
    robust: bool = True,
) -> tuple[float, dict]:
    """
    Calibrates the 'beta' hyperparameter using the General Posterior
    Calibration (GPC) algorithm for NSM-Bayes-conj.

    Parameters
    ----------
    x_obs : torch.Tensor
        The original observed data, shape (n, d_x).
    model : ConjugateModel
        Trained NSM-Bayes-conj model (networks and standardizers).
    prior_mean : torch.Tensor
        The original (un-normalized) prior mean.
    prior_cov : torch.Tensor
        The original (un-normalized) prior covariance.
    initial_beta : float, default 0.1
        Starting guess for beta.
    robust : bool, default True
        If True, use robust (median/MCD) statistics for the observed data;
        otherwise use plain sample moments.
    target_coverage : float, default 0.95
        The desired frequentist coverage (e.g., 0.95).
    num_iterations : int, default 50
        Number of iterations for the stochastic approximation.
    num_bootstraps : int, default 200
        Number of bootstrap samples to estimate coverage at each iteration.
    c : float, default 1.0
        Scale parameter of the IMQ weight.
    learning_rate_fn : callable, default lambda t: 10.0 / (t + 10)
        Step size schedule for beta updates.
    max_log_step : float, default 0.25
        Cap on ``k_t * (coverage error)`` per log-beta update.
    beta_min : float, default 0.01
        Lower bound for beta to prevent collapse.

    Returns
    -------
    float
        The final calibrated value for beta.
    dict
        A history of beta values and empirical coverages during training.
    """
    print("--- Starting General Posterior Calibration for beta ---")
    
    # --- Device and Dimension Setup ---
    device = x_obs.device
    d_theta = prior_mean.shape[0]
    n_obs = x_obs.shape[0]

    # --- Pre-calculate normalized prior parameters ---
    prior_mean_normalized = model.standardizer_theta(prior_mean)
    scales = model.standardizer_theta.std
    prior_cov_normalized = prior_cov / torch.outer(scales, scales)

    # --- Step 0: Establish the "Ground Truth" Parameter for the Bootstrap ---
    x_obs_normalized = model.standardizer_x(x_obs)
    mu_hat_obs, Sigma_inv_obs = estimate_stats(x_obs_normalized, robust)

    # NSM-only minimiser (in normalized theta space)
    theta_hat_norm_true = compute_theta_hat_true_conjugate(
        x_obs_normalized,
        model,
        imq_weight,
        mu_hat_obs,
        Sigma_inv_obs,
        c,
    )
    
    # Transform back to original theta scale
    theta_hat_true = theta_hat_norm_true * scales + model.standardizer_theta.mean
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
            desc=f"Iter {t+1}/{num_iterations} | beta={beta:.4f}"
        )
        for _ in iterable:
            # a. Create a bootstrap sample (by resampling indices with replacement)
            indices = torch.randint(0, n_obs, (n_obs,), device=x_obs.device)
            x_boot = x_obs[indices]

            # b. Compute posterior for the bootstrap sample using the current beta
            x_boot_normalized = model.standardizer_x(x_boot)
            mu_hat_boot, Sigma_inv_boot = estimate_stats(x_boot_normalized, robust)

            mu_boot_norm, Sigma_boot_norm = compute_posterior_conjugate(
                x_boot_normalized, model, beta,
                prior_mean_normalized, prior_cov_normalized,
                imq_weight, mu_hat_boot, Sigma_inv_boot, c
            )

            # c. Construct credible region (by un-normalizing) and check coverage
            mu_boot_orig = mu_boot_norm * scales + model.standardizer_theta.mean
            Sigma_boot_orig = Sigma_boot_norm * torch.outer(scales, scales)
            
            # Ensure covariance is invertible
            Sigma_inv_boot_orig = sigma_inverse(Sigma_boot_orig)
            diff = theta_hat_true - mu_boot_orig
            
            # d. Check if theta_hat_true is inside the credible region
            mahalanobis_sq = torch.dot(diff, Sigma_inv_boot_orig @ diff)
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
        print(f"Iter {t+1}/{num_iterations} | Current Beta: {beta:.4f} | Empirical Coverage: {empirical_coverage:.3f} | Target: {target_coverage}")
        history['betas'].append(beta)
        history['coverages'].append(empirical_coverage)

    print(f"Final calibrated beta: {beta:.4f}")
    return beta, history
