
import numpy as np
import torch
from sklearn.covariance import MinCovDet
from torch.func import grad as fgrad
from torch.func import hessian, jacrev, jvp, vmap

from nsm_bayes.utils.utils import jacobian_and_hessian, jacobian_and_laplacian
from nsm_bayes.utils.weightfuns import weight_function

# Small constant that keeps the relative-jitter scale in `safe_cholesky`
# from collapsing to zero when the data scale is tiny.
_EPS = 1e-12


def safe_cholesky(Sigma: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
    """Cholesky factor of Sigma.

    Cheaper to factorise it first, then check whether it is non-invertible.
    Add jitter only if needed.
    """

    L, info = torch.linalg.cholesky_ex(Sigma)
    if info == 0:
        return L
    d = Sigma.shape[-1]
    eye = torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
    scale = Sigma.diagonal().mean().abs().clamp_min(_EPS)
    jitter = 1e-8 * scale
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(Sigma + jitter * eye)
        if info == 0:
            return L
        jitter *= 10
    raise RuntimeError(f"Covariance not positive definite even with jitter {jitter:.2e}")

def sigma_inverse(Sigma_hat: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Return the inverse of ``Sigma_hat + jitter * I``.

    Parameters
    ----------
    Sigma_hat : torch.Tensor
        Scalar variance (shape ``(1,)`` or ``(1, 1)``) or covariance matrix.
    jitter : float, default 1e-6
        Value added to the diagonal before inverting; keeps the solve
        well-conditioned for near-singular covariances.

    Returns
    -------
    torch.Tensor
        Inverse of ``Sigma_hat + jitter * I`` (scalar for a scalar input,
        matrix otherwise).

    Raises
    ------
    ValueError
        If the variance is not finite or is non-positive in the scalar case.
    """
    if Sigma_hat.numel() == 1:   # scalar variance (d_x = 1)
        var = Sigma_hat.reshape(1, 1) + jitter
        if not torch.isfinite(var).all():
            raise ValueError(
                f"Variance estimate is not finite ({var.item()}); "
                "check the input data."
            )
        if var.item() <= 0:
            raise ValueError(
                f"Variance estimate {Sigma_hat.item():.3e} plus jitter "
                f"{jitter:.3e} is non-positive. "
                "Data may be constant or the outlier proportion too high."
            )
        return 1.0 / var

    # otherwise sigma is a matrix, find the inverse
    eye = torch.eye(
        Sigma_hat.shape[-1], device=Sigma_hat.device, dtype=Sigma_hat.dtype
    )
    L = safe_cholesky(Sigma_hat + jitter * eye)
    return torch.cholesky_inverse(L)

# ===========================================================================================
# PART 0: Common functions needed for the proposed neural score-matching Bayes method
# ===========================================================================================

def sample_mean_and_covariance(x_obs: torch.Tensor):
    """
    Compute the sample mean and (unbiased) sample covariance of x_obs.

    Args:
        x_obs (Tensor): shape (n, d)

    Returns:
        mean (Tensor): shape (d,)
        cov  (Tensor): shape (d, d)
    """
    if x_obs.ndim != 2:
        raise ValueError("x_obs must have shape (n, d)")

    n = x_obs.shape[0]

    # Sample mean
    mean = x_obs.mean(dim=0)

    # Centered data
    x_centered = x_obs - mean

    # Sample covariance (unbiased, divide by n-1)
    cov = (x_centered.T @ x_centered) / (n - 1)

    return mean, cov

def median_heuristic(x):
    dists = torch.cdist(x, x)
    med = torch.median(dists[dists > 0])
    return med**2

#######----Function to compute robust mean and covariance in the weights w_IMQ----######
def robust_mean_cov(x_obs):
    """
    Computes component-wise median and robust Minimum Covariance Determinant estimator.
    
    Inputs:
    - x_obs: (n, d_x) torch tensor
    Returns:
    - mu_hat: (d_x,) torch tensor (robust mean)
    - Sigma_hat: (d_x, d_x) torch tensor (robust covariance)
    """
    x_np = x_obs.detach().cpu().numpy()
    
    # Component-wise median as a robust location estimator
    mu_hat = torch.median(x_obs, dim=0).values
    # MCD for a robust covariance estimator
    mcd = MinCovDet().fit(x_np)
    Sigma_hat = torch.from_numpy(mcd.covariance_).to(x_obs.device, dtype=x_obs.dtype)
    if np.isnan(mcd.covariance_).any():
        Sigma_hat = torch.cov(x_obs.T)

    return mu_hat, Sigma_hat


# ===========================================================================================
# PART 1: Functions needed for NSM-Bayes (also referred to as the GENERAL case)
# ===========================================================================================

#######----Score Matching Loss Function in the general case used for MCMC Sampling from Generalized Bayes posterior----######

def score_matching_loss_general_loop(
    x_obs: torch.Tensor,            # (n_obs, d_x)
    theta: torch.Tensor,            # (d_theta,)
    q_phi_log_prob,                 # estimator object
    mu_hat: torch.Tensor,           # (d_x,)
    Sigma_inv: torch.Tensor,        # (d_x, d_x)
    c: float,
    weight_type: str = "imq",
):
    """
    Robust SM loss for one theta, averaged over x_obs, calling the NLE estimator
    with shapes (1,1,d_x) for x and (1,d_theta) for theta.
    """
    assert x_obs.ndim == 2, f"x_obs must be (n_obs, d_x); got {tuple(x_obs.shape)}"
    n_obs, d_x = x_obs.shape

    dev, dt = x_obs.device, x_obs.dtype

    # --- hard cast everything to (dev, dt) ---
    theta     = theta.to(device=dev, dtype=dt)
    mu_hat    = mu_hat.to(device=dev, dtype=dt)
    Sigma_inv = Sigma_inv.to(device=dev, dtype=dt)
    if torch.is_tensor(c):
        c = c.to(device=dev, dtype=dt)
    else:
        c = torch.as_tensor(c, device=dev, dtype=dt)

    weight_fn = lambda x: weight_function(weight_type, x, mu_hat, Sigma_inv, c)
    per_x_losses = []
    theta_detached = theta.detach()

    for i in range(n_obs):
        xi = x_obs[i]                                # (d_x,)
        xi_req = xi.detach().clone().requires_grad_(True)

        # log q_phi(x_i | theta) as scalar (with grad wrt xi_req)
        logp = nle_log_prob_safe(q_phi_log_prob, xi_req, theta_detached)

        # score wrt x: ∇_x log q
        score = torch.autograd.grad(logp, xi_req, create_graph=True)[0]  # (d_x,)

        # Hessian trace wrt x (exact; Hutchinson optional for speed)
        hess_trace = 0.0
        for j in range(d_x):
            second = torch.autograd.grad(score[j], xi_req, retain_graph=True)[0][j]
            hess_trace = hess_trace + second

        # robust weights at xi (no grad through weights)
        w2, grad_w2 = weight_fn(xi)

        term1 = w2 * (score @ score)
        term2 = 2.0 * grad_w2 @ score
        term3 = 2.0 * w2 * hess_trace
        per_x_losses.append(term1 + term2 + term3)

    return torch.stack(per_x_losses).mean()



def nle_log_prob_safe(estimator, x_row: torch.Tensor, theta_vec: torch.Tensor) -> torch.Tensor:
    """
    x_row: (d_x,), theta_vec: (d_theta,)
    Returns scalar log q_phi(x|theta) with shapes compatible with SBI's NFlowsFlow.
    """
    x_b  = x_row.reshape(1, 1, -1).contiguous()   # (sample=1, batch=1, d_x)
    th_b = theta_vec.reshape(1, -1).contiguous()  # (batch=1, d_theta)
    try:
        p = next(estimator.parameters())
        x_b  = x_b.to(p.device, p.dtype)
        th_b = th_b.to(p.device, p.dtype)
    except Exception:
        pass
    out = estimator.log_prob(x_b, th_b)  # (1,1)
    return out.reshape(-1).sum()         # scalar


def score_matching_loss_general_vmap(
    x_obs: torch.Tensor,            # (n, d_x)
    theta: torch.Tensor,            # (d_theta,)
    q_phi_log_prob,                 # estimator
    mu_hat: torch.Tensor,           # (d_x,)
    Sigma_inv: torch.Tensor,        # (d_x, d_x)
    c: float = 1.0,
    weight_type: str = "imq",
    hutchinson_probes: int = 0,     # 0 => exact trace; >=1 => Hutchinson
    probes: torch.Tensor | None = None,  # optional pre-generated probes (P, d_x)
) -> torch.Tensor:
    """
    Vectorized robust SM loss. No randomness inside vmap (probes pre-generated).
    """
    assert x_obs.ndim == 2, f"x_obs must be (n,d_x), got {tuple(x_obs.shape)}"
    n, d_x = x_obs.shape

    dev, dt = x_obs.device, x_obs.dtype

    # --- hard cast everything to (dev, dt) ---
    theta     = theta.to(device=dev, dtype=dt)
    mu_hat    = mu_hat.to(device=dev, dtype=dt)
    Sigma_inv = Sigma_inv.to(device=dev, dtype=dt)
    if torch.is_tensor(c):
        c = c.to(device=dev, dtype=dt)
    else:
        c = torch.as_tensor(c, device=dev, dtype=dt)


    weight_fn = lambda x: weight_function(weight_type, x, mu_hat, Sigma_inv, c)

    # Prepare probes if using Hutchinson
    use_hutch = (hutchinson_probes is not None) and (hutchinson_probes > 0) and (d_x > 1)
    if use_hutch:
        if probes is None:
            # SAME probes for all xi (fast, deterministic inside vmap)
            probes = torch.randn(hutchinson_probes, d_x, device=x_obs.device, dtype=x_obs.dtype)
        else:
            assert probes.shape == (hutchinson_probes, d_x), \
                f"probes must have shape (P, d_x) = ({hutchinson_probes}, {d_x}), got {tuple(probes.shape)}"

    # scalar log q(x|theta) as a function of x only
    def log_q_xonly(x_vec: torch.Tensor) -> torch.Tensor:
        return nle_log_prob_safe(q_phi_log_prob, x_vec, theta)

    def per_x_loss(xi: torch.Tensor) -> torch.Tensor:
        # score wrt x
        score = jacrev(log_q_xonly)(xi)  # (d_x,)

        # trace of Hessian wrt x: exact or Hutchinson
        if use_hutch:
            g = fgrad(log_q_xonly)  # ∇_x log q
            # vmap over probes: v -> (H v · v)
            def one_probe(v):
                hvp, _ = jvp(g, (xi,), (v,))    # hvp = H v
                return (hvp * v).sum()
            trH = vmap(one_probe, in_dims=0)(probes).mean()
        else:
            H = hessian(log_q_xonly)(xi)        # (d_x,d_x) or scalar
            trH = H if H.ndim == 0 else torch.trace(H)

        # weights (no grad through weights)
        w2, grad_w2 = weight_fn(xi)

        return w2 * (score @ score) + 2.0 * grad_w2 @ score + 2.0 * w2 * trH

    losses = vmap(per_x_loss, in_dims=0)(x_obs)  # (n,)
    return losses.mean()

        
class ScoreMatchingLogPosterior:
    def __init__(self, x_obs, prior, beta, q_phi_log_prob,
                 mu_hat, Sigma_inv, c=1.0, weight_type="imq",
                 use_vmap=True, hutchinson_probes=0, probes=None):
        self.x_obs = x_obs
        self.prior = prior
        self.beta = beta
        self.q_phi_log_prob = q_phi_log_prob  
        self.mu_hat = mu_hat
        self.Sigma_inv = Sigma_inv
        self.c = c
        self.weight_type = weight_type
        self.n_obs = x_obs.shape[0]
        self.use_vmap = use_vmap
        self.hutchinson_probes = hutchinson_probes
        self.probes = probes  # optional pre-generated (P, d_x)

    def __call__(self, theta_np):
        theta = torch.from_numpy(theta_np).float()

        with torch.no_grad():
            lp = self.prior.log_prob(theta).sum()
            if torch.isneginf(lp):
                return -np.inf

        if self.use_vmap:
            L_sm = score_matching_loss_general_vmap(
                x_obs=self.x_obs,
                theta=theta,
                q_phi_log_prob=self.q_phi_log_prob,
                mu_hat=self.mu_hat,
                Sigma_inv=self.Sigma_inv,
                c=self.c,
                weight_type=self.weight_type,
                hutchinson_probes=self.hutchinson_probes,
                probes=self.probes,
            )
        else:
            L_sm = score_matching_loss_general_loop(
                x_obs=self.x_obs,
                theta=theta,
                q_phi_log_prob = self.q_phi_log_prob,
                mu_hat=self.mu_hat,
                Sigma_inv=self.Sigma_inv,
                c=self.c,
                weight_type=self.weight_type,
            )

        return float(lp - (self.beta * self.n_obs * L_sm))



# ===========================================================================================
# PART 2: Functions needed for NSM-Bayes-conj method
# ===========================================================================================


def calculate_training_loss(x_batch, theta_batch, T_phi, b_phi, Sigma_inv):
    """
    Weighted score-matching loss (per-batch mean) for the exponential-family case:
       score(x;θ) = J_T(x)^T θ + ∇ b(x)
       L(x,θ)     = (score^T Σ^{-1} score) + 2 tr( Σ^{-1} ( Σ_i θ_i ∇^2 T_i(x) + ∇^2 b(x) ) )

    Shapes:
      x_batch:      (B, d_x)
      theta_batch:  (B, d_theta)
      T_phi(x):     (d_theta,)         -> J_T(x): (d_theta, d_x)
      b_phi(x):     scalar             -> ∇ b(x): (d_x,),  ∇^2 b(x): (d_x, d_x)
      Σ^{-1}:       (d_x, d_x)
    """
    B, d_x = x_batch.shape
    assert theta_batch.shape[0] == B, "Batch sizes of x and theta must match."
    d_theta = theta_batch.shape[1]

    # Ensure Σ^{-1} matches dtype/device
    Sigma_inv = Sigma_inv.to(device=x_batch.device, dtype=x_batch.dtype)
    assert Sigma_inv.shape == (d_x, d_x), "Sigma_inv must be (d_x, d_x)."

    # Temporarily set eval() to freeze any dropout/bn, but keep grads
    was_training_T, was_training_b = T_phi.training, b_phi.training
    T_phi.eval(); b_phi.eval()

    def loss_for_single_pair(xi, thetai):
      
        # Jacobians via torch.func
        # J_T: (d_theta, d_x)
        #J_T   = jacrev(T_phi_wrapped)(xi)
        # ∇b: (d_x,)
        #grad_b = jacrev(b_phi_wrapped)(xi)

        # Hessians via torch.func (note call style hessian(f)(x))
        # H_T: (d_theta, d_x, d_x)  (each slice is ∇^2 T_i(x))
        #H_T = hessian(T_phi_wrapped)(xi)
        # H_b: (d_x, d_x)
        #H_b = hessian(b_phi_wrapped)(xi)

        J_T,H_T = jacobian_and_hessian(xi,T_phi)
        grad_b, H_b = jacobian_and_hessian(xi,b_phi)

        # score(x;θ): (d_x,)
        score = J_T.T @ thetai + grad_b

        # ∇^2 log q(x|θ) = Σ_i θ_i ∇^2 T_i(x) + ∇^2 b(x)  -> (d_x, d_x)
        H_logq = torch.einsum('i,ijk->jk', thetai, H_T) + H_b

        # score^T Σ^{-1} score
        score_term = torch.dot(score, Sigma_inv @ score)

        # 2 * tr( Σ^{-1} H_logq ) = 2 * sum(Σ^{-1} * H_logq)
        div_term = 2.0 * torch.sum(Sigma_inv * H_logq)

        return (score_term + div_term) #/ d_x

    # Vectorize across batch
    losses = vmap(loss_for_single_pair, in_dims=(0, 0))(x_batch, theta_batch)

    # Restore training flags
    if was_training_T: T_phi.train()
    if was_training_b: b_phi.train()

    return losses.mean()


#######----Function to compute posterior precision matrix for NSM-Bayes-conj----######
def compute_posterior_precision_conjugate(
    x_obs, model, beta, prior_cov_inv, weight_fn, mu_hat_obs, Sigma_inv_obs, c
):
    """
    Computes the posterior precision matrix (inverse covariance) for the model.

    """

    # Define the calculation for a single data point
    def get_cov_term_single(xi):
        # Create a wrapper for T_phi that accepts a vector
        T_phi_wrapped = lambda x_vec: model.T_phi(x_vec.unsqueeze(0)).squeeze(0)
        jac_T = jacrev(T_phi_wrapped)(xi) # shape (d_theta, d_x)
        return jac_T @ jac_T.T

    # Vectorize the calculation over all observations
    batched_cov_terms = vmap(get_cov_term_single)(x_obs) # shape: (n, d_theta, d_theta)

    # Apply weights and sum
    weights = weight_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c).view(-1, 1, 1)

    sum_term = (weights * batched_cov_terms).sum(dim=0)

    # Assemble the final precision matrix
    Sigma_n_inv = prior_cov_inv + 2 * beta * sum_term

    return Sigma_n_inv

def compute_posterior_conjugate(
    x_obs, model, beta, prior_mean, prior_cov, weight_fn, mu_hat_obs, Sigma_inv_obs, c
):
    """
    Computes the posterior mean and covariance for the neural exponential family model.

    Returns:
    - mu_n: (d_theta,) tensor for the posterior mean.
    - Sigma_n: (d_theta, d_theta) tensor for the posterior covariance.
    """
    
    # Invert prior covariance to get prior precision
    prior_cov_inv = torch.linalg.inv(prior_cov)
    
    # 1. Compute the posterior precision matrix
    Sigma_n_inv = compute_posterior_precision_conjugate(
        x_obs, model, beta, prior_cov_inv, weight_fn, mu_hat_obs, Sigma_inv_obs, c
    )

    # --- 2. Calculate terms needed for the posterior mean ---


    batched_non_div_terms = vmap(get_non_div_term_single, in_dims=(0, None))(x_obs,model)
    weights_h,grad_w2 = weight_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c)
    term_non_div = (weights_h * batched_non_div_terms).sum(dim=0)

    batched_divergences = vmap(
        get_div_term_single, in_dims=(0, None, None, None, None)
                 )(x_obs, model, w2=weights_h,grad_w2=grad_w2)          # (n, d_theta)
    
    #batched_divergences = vmap(get_div_term_single)(x_obs)
    term_div = batched_divergences.sum(dim=0)
    
    # --- 3. Solve for the Posterior Mean ---
    
    # Assemble the right-hand side of the linear system: Sigma_n_inv @ mu_n = rhs
    prior_term = prior_cov_inv @ prior_mean
    rhs = prior_term - 2 * beta * (term_div + term_non_div)
    
    # Solve for mu_n using the stable solver.
    mu_n = torch.linalg.solve(Sigma_n_inv, rhs)

    # --- 4. Compute Posterior Covariance from Precision ---
    
    # Use a numerically stable Cholesky-based inverse
    L = torch.linalg.cholesky(Sigma_n_inv)
    Sigma_n = torch.cholesky_inverse(L)
    
    return mu_n, Sigma_n



# --- 2a. Vectorized Non-Divergence Term (h_n) ---
def get_non_div_term_single(xi,model):
    T_phi_wrapped = lambda x_vec: model.T_phi(x_vec.unsqueeze(0)).squeeze(0)
    b_phi_wrapped = lambda x_vec: model.b_phi(x_vec.unsqueeze(0)).squeeze()
    
    jac_T = jacrev(T_phi_wrapped)(xi) # shape: (d_theta, d_x)
    jac_b = jacrev(b_phi_wrapped)(xi) # shape: (d_x,)
    
    return jac_T @ jac_b

# --- 2b. Vectorized Divergence Term (D_n) 
def get_div_term_single(xi,model,w2,grad_w2):

    # 1. Compute Laplacian of T_phi: Tr(Hess(T_phi))
    # T_phi_wrapped: (d_x) -> (d_theta)
    # Hessian: (d_theta, d_x, d_x)
    jac_T,lap_T  = jacobian_and_laplacian(xi,model.T_phi)
    #jacfwd(jacobian_with_spare, has_aux=True)(xi, model.T_phi)   # (d_θ, d_x, d_x), (d_θ, d_x)
    #laplacian_T = torch.diagonal(hess_T, dim1=1, dim2=2).sum(dim=1) # shape: (d_theta,)

    grad_w2 = grad_w2.squeeze(0)
    
    # grad_w2: (d_x,)
    # jac_T: (d_theta, d_x)
    # laplacian_T: (d_theta,)
    divergence = grad_w2 @ jac_T.T + w2 * lap_T
    
    return divergence
    