import torch
from torch.func import vmap, jacrev, hessian
from method import *
from utils import *
from slice_sampler import *
from scipy.stats import chi2
from tqdm import tqdm
import math

######-------Functions needed for general posterior calibration (setting the learning rate) for NSM-Bayes------#########
def weight_function_factory_batched(name: str, x_obs: torch.Tensor, mu_hat: torch.Tensor,
                            Sigma_inv: torch.Tensor, c: float):
    """
    Returns two tensors (w2_i, grad_w2_i) for all x_i in x_obs.
    w2_i has shape (n,), grad_w2_i has shape (n, d_x).
    """
    n, d_x = x_obs.shape

    if name.lower() == "none":
        w2 = torch.ones(n, device=x_obs.device, dtype=x_obs.dtype)
        grad_w2 = torch.zeros(n, d_x, device=x_obs.device, dtype=x_obs.dtype)
        return w2, grad_w2

    if name.lower() == "imq":
        diff = (x_obs - mu_hat).detach()
        # norm_sq: (n,)
        norm_sq = torch.sum((diff @ Sigma_inv) * diff, dim=1)
        denom = 1.0 + norm_sq / c + 1e-12 
        w2 = 1.0 / (denom ** 2) 
        grad_w2 = - (4.0 / c) * (1.0 / (denom ** 3)).unsqueeze(-1) * (diff @ Sigma_inv.T) 
        return w2, grad_w2

    raise ValueError(f"Unknown weight_type={name}")

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
        score = jacrev(log_q_fn)(xi)      # (d_x,)
        hess  = hessian(log_q_fn)(xi)     # (d_x,d_x) or scalar if d_x=1

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
    W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
    return W


def weighted_mean_cov(theta_samples: torch.Tensor, w: torch.Tensor):
    """
    theta_samples: (M, d); w: (M,) normalised weights.
    returns mu: (d,), Sigma: (d,d)
    """
    M, d = theta_samples.shape
    w = w.view(-1, 1)
    mu = (w * theta_samples).sum(dim=0)
    xc = theta_samples - mu
    # weighted covariance with Bessel-like correction off (posterior covariance)
    Sigma = (w * xc).T @ xc / (w.sum() + 1e-12)
    # regularise a bit
    eye = torch.eye(d, device=theta_samples.device, dtype=theta_samples.dtype)
    Sigma = Sigma + 1e-6 * eye
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
    cdf = cdf / (cdf[-1] + 1e-12)
    return v[torch.searchsorted(cdf, torch.tensor(q, device=v.device))]

def bootstrap_coverage_indicator(
    theta_samples: torch.Tensor,   # (M, d)
    weights_row: torch.Tensor,     # (M,)
    theta_hat: torch.Tensor,       # (d,)
    alpha: float
) -> int:
    w = weights_row / (weights_row.sum() + 1e-12)
    mu, Sigma = weighted_mean_cov(theta_samples, w)
    Sigma_inv = torch.linalg.inv(Sigma)
    d2 = weighted_mahalanobis_sq(theta_samples, mu, Sigma_inv)
    thresh = weighted_quantile(d2, w, 1.0 - alpha)
    d2_hat = (theta_hat - mu).unsqueeze(0) @ Sigma_inv @ (theta_hat - mu).unsqueeze(1)
    return int(d2_hat.item() <= thresh.item())


def compute_theta_hat_true_general(
    x_obs: torch.Tensor,                       # (n, d_x)
    q_phi_log_prob,                            # fn (x_batch, theta_batch) -> log prob
    w2: torch.Tensor,                          # (n,)
    grad_w2: torch.Tensor,                     # (n, d_x)
    theta_init: torch.Tensor,                  # (d_theta,), starting point
    num_steps: int = 200,
    lr: float = 1e-2,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute the NSM-only minimiser theta_hat_true for NSM-Bayes:

        theta_hat_true = argmin_theta L_NSM(theta; x_obs)

    using gradient-based optimisation.

    Args
    ----
    x_obs      : (n, d_x) observed data.
    q_phi_log_prob : function (x, theta) -> log q_phi(x | theta), batch-safe.
    w2, grad_w2   : outputs of weight_function_factory_batched (n,), (n, d_x).
    theta_init    : initial guess for theta (e.g. posterior mean at beta_base).
    num_steps     : number of optimisation steps.
    lr            : learning rate for Adam.
    verbose       : if True, prints loss occasionally.

    Returns
    -------
    theta_hat_true : (d_theta,) tensor (same space/scale as theta_init).
    """

    device = x_obs.device
    dtype  = x_obs.dtype
    n, d_x = x_obs.shape
    d_theta = theta_init.shape[0]

    theta = theta_init.clone().detach().to(device=device, dtype=dtype)
    theta.requires_grad_(True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    def nsm_loss_current_theta():
        # per-x NSM loss, same structure as in cache_perx_sm_losses
        def per_x_loss_for_theta(xi, w2_i, grad_w2_i):
            # log q(x | theta) for single xi, current theta
            log_q_fn = lambda x_vec: q_phi_log_prob(
                x_vec.unsqueeze(0),
                theta.unsqueeze(0)
            ).sum()

            score = jacrev(log_q_fn)(xi)      # (d_x,)
            hess  = hessian(log_q_fn)(xi)     # (d_x,d_x) or scalar

            term1 = w2_i * (score @ score)
            term2 = 2.0 * grad_w2_i @ score
            if hess.ndim == 0:
                term3 = 2.0 * w2_i * hess
            else:
                term3 = 2.0 * w2_i * torch.trace(hess)
            return term1 + term2 + term3

        # vectorise over x
        losses = vmap(
            per_x_loss_for_theta,
            in_dims=(0, 0, 0)
        )(x_obs, w2, grad_w2)  # (n,)

        return losses.mean()

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = nsm_loss_current_theta()
        loss.backward()
        optimizer.step()

        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(f"[theta_hat optimisation] step {step:4d} | L_NSM = {loss.item():.4f}")
    print("Estimated theta true: ", theta)
    return theta.detach()

def calibrate_beta(
    theta_samples_base: torch.Tensor,   # (M, d) samples drawn at beta_base on original data
    beta_base: float,
    x_obs: torch.Tensor,                # (n, d_x)
    q_phi_log_prob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mu_hat: torch.Tensor,
    Sigma_inv: torch.Tensor,
    weight_type: str,
    c: float,
    alpha: float = 0.05,                # alpha -> 1-alpha is target coverage (e.g. 0.95)
    B: int = 200,                       # bootstraps per iteration
    T: int = 10,                        # number of stochastic approx iterations
    beta_init: Optional[float] = None,
    # 
    step_schedule: Callable[[int], float] = lambda t: 10.0 / (t + 10),
    max_log_step: float = 0.25,     # cap on k_t * (coverage error)
    beta_min: float = 0.01,
    ess_threshold: float = 0.3,         # if ESS too small, we may refresh samples
    refresh_sampler: Optional[Callable[[float], torch.Tensor]] = None,  # sampler(beta)->(M,d)
) -> tuple[float, dict]:
    """
    Calibrate beta via bootstrap coverage matching for NSM-Bayes

    Returns
    -------
    beta_star : float
        Final calibrated beta.
    history : dict
        Dictionary with 'betas' and 'coverages' over iterations, useful for plotting.
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
    w2, grad_w2 = weight_function_factory_batched(weight_type, x_obs, mu_hat, Sigma_inv, c)

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
        if log_beta < log_beta_min:
            log_beta = log_beta_min

        # Update current_beta for logging and potential refresh
        current_beta = math.exp(log_beta)

        # --- Step 6: track effective sample size (ESS) to detect weight degeneracy
        # ESS per bootstrap b: 1 / sum_m W[b,m]^2
        ess = (W**2).sum(dim=1)
        ess = 1.0 / (ess + 1e-12)
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

def compute_theta_hat_true_case1(
    x_obs: torch.Tensor,
    T_phi: torch.nn.Module,
    b_phi: torch.nn.Module,
    w_imq_squared_fn,
    mu_hat_obs: torch.Tensor,
    Sigma_inv_obs: torch.Tensor,
    c: float,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the NSM-only minimiser theta_hat_true for NSM-Bayes-conj:
        theta_hat_true = argmin_theta L_NSM(theta; x_obs, phi_hat)

    This corresponds to the M-estimator used as the "true" parameter
    in the Syring & Martin calibration paper.

    Args:
        x_obs: (n, d_x) observed data (already normalized if you use standardizer_x).
        T_phi, b_phi: trained networks.
        w_imq_squared_fn: function w^2(x) used elsewhere.
        mu_hat_obs, Sigma_inv_obs, c: the same robust-location / weight hyperparams
            you already pass to compute_posterior_case1.
        jitter: small diagonal regularizer for the linear solve.

    Returns:
        theta_hat_true: (d_theta,) tensor in the same (normalized) theta space
        that T_phi produces.
    """

    device = x_obs.device
    dtype = x_obs.dtype

    # ---------- 1. A = sum_i w(x_i)^2 * J_T(x_i)^T J_T(x_i) ----------
    def get_cov_term_single(xi):
        # T_phi: x -> R^{d_theta}
        T_phi_wrapped = lambda x_vec: T_phi(x_vec.unsqueeze(0)).squeeze(0)
        jac_T = jacrev(T_phi_wrapped)(xi)  # (d_theta, d_x)
        return jac_T @ jac_T.T             # (d_theta, d_theta)

    batched_cov_terms = vmap(get_cov_term_single)(x_obs)  # (n, d_theta, d_theta)

    weights = w_imq_squared_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c).view(-1, 1, 1)  # (n,1,1)
    A = (weights * batched_cov_terms).sum(dim=0)  # (d_theta, d_theta)

    # ---------- 2. B = term_non_div + term_div (vector in R^{d_theta}) ----------
    # term_non_div = sum_i w(x_i)^2 * J_T(x_i)^T * grad_x b(x_i)
    def get_non_div_term_single(xi):
        T_phi_wrapped = lambda x_vec: T_phi(x_vec.unsqueeze(0)).squeeze(0)
        b_phi_wrapped = lambda x_vec: b_phi(x_vec.unsqueeze(0)).squeeze()

        jac_T = jacrev(T_phi_wrapped)(xi)  # (d_theta, d_x)
        jac_b = jacrev(b_phi_wrapped)(xi)  # (d_x,)
        return jac_T @ jac_b               # (d_theta,)

    batched_non_div_terms = vmap(get_non_div_term_single)(x_obs)  # (n, d_theta)
    weights_h = w_imq_squared_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c).view(-1, 1)  # (n,1)
    term_non_div = (weights_h * batched_non_div_terms).sum(dim=0)  # (d_theta,)

    # term_div = sum_i div_x[ w(x_i)^2 * J_T(x_i)^T ]  (vector in R^{d_theta})
    def get_div_term_single(xi):
        def divergence_field_fn(x_vec):
            T_phi_wrapped = lambda z: T_phi(z.unsqueeze(0)).squeeze(0)
            w2_local = w_imq_squared_fn(x_vec.unsqueeze(0), mu_hat_obs, Sigma_inv_obs, c).squeeze()
            jac_T_local = jacrev(T_phi_wrapped)(x_vec)  # (d_theta, d_x)
            # field: (d_x, d_theta); column k is a vector field in R^{d_x}
            return w2_local * jac_T_local.T

        # jac_of_field: (d_x, d_theta, d_x)
        jac_of_field = jacrev(divergence_field_fn)(xi)
        # divergence of each column: trace over (0,2) dims, get (d_theta,)
        divergence = jac_of_field.diagonal(offset=0, dim1=0, dim2=2).sum(dim=1)
        return divergence  # (d_theta,)

    batched_divergences = vmap(get_div_term_single)(x_obs)  # (n, d_theta)
    term_div = batched_divergences.sum(dim=0)               # (d_theta,)

    B = term_non_div + term_div  # (d_theta,)

    eig = torch.linalg.eigvalsh(A).detach().cpu()
    print("A: ", A)
    print("A eig min/max:", eig.min().item(), eig.max().item())
    print("cond(A):", (eig.max()/eig.min().clamp_min(1e-30)).item())
    print("||B||:", torch.linalg.norm(B).item())

    # ---------- 3. Solve A * theta_hat_true = -B ----------
    d_theta = A.shape[0]
    scale = torch.trace(A) / A.shape[0]
    lam = 1e-2 * scale + 1e-12
    A_reg = 0.5 * (A + A.T) + lam * torch.eye(d_theta, device=A.device, dtype=A.dtype)

    try:
        L = torch.linalg.cholesky(A_reg)
        theta_hat_true = -torch.cholesky_solve(B.unsqueeze(1), L).squeeze(1)
    except RuntimeError:
        theta_hat_true = -torch.linalg.lstsq(A_reg, B).solution

    return theta_hat_true

def calibrate_beta_gpc(
    x_obs: torch.Tensor,
    T_phi_net: torch.nn.Module,
    b_phi_net: torch.nn.Module,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    standardizer_x,
    standardizer_theta,
    # --- GPC Hyperparameters ---
    initial_beta: float = 0.1,
    target_coverage: float = 0.95,
    num_iterations: int = 50,
    num_bootstraps: int = 200,
    # 
    learning_rate_fn = lambda t: 5.0 / (t + 10),
    max_log_step: float = 0.25,     # cap on k_t * (coverage error)
    beta_min: float = 0.01,
) -> tuple[float, dict]:
    """
    Calibrates the 'beta' hyperparameter using the General Posterior Calibration (GPC) algorithm for NSM-Bayes-conj.

    Args:
        x_obs (torch.Tensor): The original observed data, shape (n, d_x).
        T_phi_net, b_phi_net: The trained neural network models.
        prior_mean, prior_cov: The original (un-normalized) prior parameters.
        standardizer_x, standardizer_theta: The fitted standardizing nets.
        initial_beta (float): Starting guess for beta.
        target_coverage (float): The desired frequentist coverage (e.g., 0.95).
        num_iterations (int): Number of iterations for the stochastic approximation.
        num_bootstraps (int): Number of bootstrap samples to estimate coverage at each iteration.
        learning_rate_fn (function): Step size schedule for beta updates.
        beta_min (float): Lower bound for beta to prevent collapse.

    Returns:
        float: The final calibrated value for beta.
        dict: A history of beta values and empirical coverages during training.
    """
    print("--- Starting General Posterior Calibration for beta ---")
    
    # --- Device and Dimension Setup ---
    device = x_obs.device
    d_theta = prior_mean.shape[0]
    n_obs = x_obs.shape[0]

    # --- Pre-calculate normalized prior parameters ---
    prior_mean_normalized = standardizer_theta(prior_mean)
    scales = standardizer_theta.std
    prior_cov_normalized = prior_cov / torch.outer(scales, scales)

    # --- Step 0: Establish the "Ground Truth" Parameter for the Bootstrap ---
    x_obs_normalized = standardizer_x(x_obs)
    mu_hat_obs, Sigma_hat_obs = robust_mean_cov(x_obs_normalized)
    Sigma_inv_obs = torch.linalg.inv(
        Sigma_hat_obs + 1e-6 * torch.eye(x_obs.shape[1], device=x_obs.device, dtype=x_obs.dtype)
    )
    c_obs = 1.

    # NSM-only minimiser (in normalized theta space)
    theta_hat_norm_true = compute_theta_hat_true_case1(
        x_obs_normalized,
        T_phi_net,
        b_phi_net,
        w_imq_squared,
        mu_hat_obs,
        Sigma_inv_obs,
        c_obs,
    )
    
    # Transform back to original theta scale
    theta_hat_true = theta_hat_norm_true * scales + standardizer_theta.mean
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
            x_boot_normalized = standardizer_x(x_boot)
            mu_hat_boot, Sigma_hat_boot = robust_mean_cov(x_boot_normalized)
            Sigma_inv_boot = torch.linalg.inv(
                Sigma_hat_boot + 1e-6 * torch.eye(x_boot.shape[1], device=x_obs.device, dtype=x_obs.dtype)
            )
            c_boot = 1.

            mu_boot_norm, Sigma_boot_norm = compute_posterior_case1(
                x_boot_normalized, T_phi_net, b_phi_net, beta,
                prior_mean_normalized, prior_cov_normalized,
                w_imq_squared, mu_hat_boot, Sigma_inv_boot, c_boot
            )

            # c. Construct credible region (by un-normalizing) and check coverage
            mu_boot_orig = mu_boot_norm * scales + standardizer_theta.mean
            Sigma_boot_orig = Sigma_boot_norm * torch.outer(scales, scales)
            
            # Ensure covariance is invertible
            Sigma_inv_boot_orig = torch.linalg.inv(
                Sigma_boot_orig + 1e-6 * torch.eye(d_theta, device=x_obs.device, dtype=x_obs.dtype)
            )
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
        if log_beta < log_beta_min:
            log_beta = log_beta_min
        
        beta = math.exp(log_beta)

        # --- Logging and History ---
        print(f"Iter {t+1}/{num_iterations} | Current Beta: {beta:.4f} | Empirical Coverage: {empirical_coverage:.3f} | Target: {target_coverage}")
        history['betas'].append(beta)
        history['coverages'].append(empirical_coverage)

    print(f"Final calibrated beta: {beta:.4f}")
    return beta, history
