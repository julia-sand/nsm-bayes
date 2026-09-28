import torch
from torch.func import vmap, jacrev, hessian, grad as fgrad, jvp
from torch.autograd.functional import jacobian
import numpy as np
from sklearn.covariance import MinCovDet
from utils import *
from typing import Callable

# ===========================================================================================
# PART 0: Common functions needed for the proposed neural score-matching Bayes method
# ===========================================================================================

def w_imq_squared(x, mu_hat, Sigma_inv, c):
    """
    Computes w_IMQ^2(x) for a batch of x.
    Inputs:
    - x: (n, d_x) torch tensor, where n is the number of observations.
    - mu_hat: (d_x,) robust mean
    - Sigma_inv: (d_x, d_x) robust covariance inverse
    - c: scalar hyperparameter
    Returns:
    - w2: (n,) tensor containing w_IMQ(x_i)^2 for each observation.
    """ 
    diff = x - mu_hat
    norm_sq = torch.sum((diff @ Sigma_inv) * diff, dim=1)
    
    denom = 1 + norm_sq / c + 1e-12
    w2 = 1.0 / (denom)**2

    return w2

# def weight_function_factory(name, mu_hat, Sigma_inv, c=1.0):
#     """
#     Returns a function (x) -> (w2, grad_w2)
#     for the chosen weighting scheme.
#     """
#     def imq_weight(x):
#         diff = (x - mu_hat).detach()
#         norm_sq = diff @ Sigma_inv @ diff
#         denom = 1.0 + norm_sq / c + 1e-12
#         w2 = 1.0 / (denom ** 2)
#         grad_w2 = - (4.0 / c) * (1.0 / (denom ** 3)) * (Sigma_inv @ diff) #- (2.0 / c) * (w2 ** 2) * (Sigma_inv @ diff)
#         return w2, grad_w2

#     def unweighted(x):
#         w2 = torch.tensor(1.0, device=x.device, dtype=x.dtype)
#         grad_w2 = torch.zeros_like(x)
#         return w2, grad_w2

#     weights = {
#         "imq": imq_weight,
#         "none": unweighted,
#     }
#     return weights[name.lower()]

def weight_function_factory(name, mu_hat, Sigma_inv, c=1.0, zeta: float = 1.0):
    """
    Returns a function (x) -> (w2, grad_w2)
    for the chosen weighting scheme.

    IMQ weight:
        w_zeta(x) = (1 + ||x-mu||_{Sigma_inv}^2 / c)^(-1/zeta)
        w2        = w_zeta(x)^2
        grad_w2   = ∇_x w2

    Default: zeta=1.0.
    """
    # make sure scalar-like values behave nicely
    if torch.is_tensor(c):
        c_t = c
    else:
        c_t = torch.as_tensor(c, dtype=mu_hat.dtype, device=mu_hat.device)

    if torch.is_tensor(zeta):
        zeta_t = zeta
    else:
        zeta_t = torch.as_tensor(zeta, dtype=mu_hat.dtype, device=mu_hat.device)

    def imq_weight(x):

        diff = (x - mu_hat).detach()  # (d_x,)
        s = diff @ Sigma_inv @ diff
        denom = 1.0 + s / c_t + 1e-12

        w2 = denom.pow(-2.0 / zeta_t)
        grad_w2 = -(4.0 / (zeta_t * c_t)) * denom.pow(-(2.0 / zeta_t + 1.0)) * (Sigma_inv @ diff)

        return w2, grad_w2

    def huber_weight(x):
        diff = (x - mu_hat).detach()
        norm = torch.sqrt((diff @ Sigma_inv @ diff) + 1e-12)
        if norm <= c_t:
            w = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            grad_w2 = torch.zeros_like(diff)
        else:
            w = c_t / norm
            grad_w2 = torch.zeros_like(diff)
        return w**2, grad_w2

    def unweighted(x):
        w2 = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        grad_w2 = torch.zeros_like(x)
        return w2, grad_w2

    weights = {
        "imq": imq_weight,
        "huber": huber_weight,
        "none": unweighted,
    }
    return weights[name.lower()]


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

    weight_fn = weight_function_factory(weight_type, mu_hat, Sigma_inv, c)
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


    weight_fn = weight_function_factory(weight_type, mu_hat, Sigma_inv, c)

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
        # Wrappers: input (d_x,) → outputs with correct shapes
        def T_phi_wrapped(x_vec):   # (d_x,) -> (d_theta,)
            return T_phi(x_vec.unsqueeze(0)).squeeze(0)

        def b_phi_wrapped(x_vec):   # (d_x,) -> scalar
            return b_phi(x_vec.unsqueeze(0)).squeeze()

        # Jacobians via torch.func
        # J_T: (d_theta, d_x)
        J_T   = jacrev(T_phi_wrapped)(xi)
        # ∇b: (d_x,)
        grad_b = jacrev(b_phi_wrapped)(xi)

        # score(x;θ): (d_x,)
        score = J_T.T @ thetai + grad_b

        # Hessians via torch.func (note call style hessian(f)(x))
        # H_T: (d_theta, d_x, d_x)  (each slice is ∇^2 T_i(x))
        H_T = hessian(T_phi_wrapped)(xi)
        # H_b: (d_x, d_x)
        H_b = hessian(b_phi_wrapped)(xi)

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


# --- Function to compute unnormalized log-probability q_phi(x|theta) for NSM-Bayes-conj ---
def compute_unnormalized_log_prob_case1(x,
                                  theta,
                                  T_phi: torch.nn.Module,
                                  b_phi: torch.nn.Module) -> torch.Tensor:
    """
    Computes the unnormalized log-probability, log(q_phi(x|theta)) + log(Z(theta)).
    """
    # Ensure inputs are PyTorch tensors.
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not isinstance(theta, torch.Tensor):
        theta = torch.as_tensor(theta)

    # Get the expected dtype from the model's parameters and cast inputs.
    model_dtype = next(T_phi.parameters()).dtype
    x = x.to(dtype=model_dtype)
    theta = theta.to(dtype=model_dtype)

    if x.dim() == 1:
        # If x is 1D (e.g., shape [1000]), reshape it to (1000, 1).
        x = x.unsqueeze(1)

    # Handle theta's shape to match the batch size of x.
    if theta.dim() == 1:
        theta = theta.expand(x.shape[0], -1) # Expand to (batch_size, d_theta)

    # Compute the two parts of the log-likelihood
    T_x = T_phi(x)
    b_x = b_phi(x)

    # Calculate the unnormalized log-probability for the batch
    log_likelihood_term = torch.einsum('bi,bi->b', T_x, theta)
    base_measure_term = b_x.squeeze(-1)

    result = log_likelihood_term + base_measure_term
    
    return result


#######----Function to compute posterior precision matrix for NSM-Bayes-conj----######
def compute_posterior_precision_case1(
    x_obs, T_phi, beta, prior_cov_inv, w_imq_squared_fn, mu_hat_obs, Sigma_inv_obs, c
):
    """
    Computes the posterior precision matrix (inverse covariance) for the model.

    """
    
    # Define the calculation for a single data point
    def get_cov_term_single(xi):
        # Create a wrapper for T_phi that accepts a vector
        T_phi_wrapped = lambda x_vec: T_phi(x_vec.unsqueeze(0)).squeeze(0)
        jac_T = jacrev(T_phi_wrapped)(xi) # shape (d_theta, d_x)
        return jac_T @ jac_T.T

    # Vectorize the calculation over all observations
    batched_cov_terms = vmap(get_cov_term_single)(x_obs) # shape: (n, d_theta, d_theta)
    
    # Apply weights and sum
    weights = w_imq_squared_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c).view(-1, 1, 1)

    sum_term = (weights * batched_cov_terms).sum(dim=0)
    
    # Assemble the final precision matrix
    Sigma_n_inv = prior_cov_inv + 2 * beta * sum_term
    
    return Sigma_n_inv

########----Function to compute posterior mean and covariance for NSM-Bayes-conj----######
def compute_posterior_case1(
    x_obs, T_phi, b_phi, beta, prior_mean, prior_cov, w_imq_squared_fn, mu_hat_obs, Sigma_inv_obs, c
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
    Sigma_n_inv = compute_posterior_precision_case1(
        x_obs, T_phi, beta, prior_cov_inv, w_imq_squared_fn, mu_hat_obs, Sigma_inv_obs, c
    )

    # --- 2. Calculate terms needed for the posterior mean ---

    # --- 2a. Vectorized Non-Divergence Term (h_n) ---
    def get_non_div_term_single(xi):
        T_phi_wrapped = lambda x_vec: T_phi(x_vec.unsqueeze(0)).squeeze(0)
        b_phi_wrapped = lambda x_vec: b_phi(x_vec.unsqueeze(0)).squeeze()
        
        jac_T = jacrev(T_phi_wrapped)(xi) # shape: (d_theta, d_x)
        jac_b = jacrev(b_phi_wrapped)(xi) # shape: (d_x,)
        
        return jac_T @ jac_b

    batched_non_div_terms = vmap(get_non_div_term_single)(x_obs)
    weights_h = w_imq_squared_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c).view(-1, 1)
    term_non_div = (weights_h * batched_non_div_terms).sum(dim=0)

    # --- 2b. Vectorized Divergence Term (D_n) ---
    def get_div_term_single(xi):
        def divergence_field_fn(x_vec):
            T_phi_wrapped = lambda z: T_phi(z.unsqueeze(0)).squeeze(0)
            w2_local = w_imq_squared_fn(x_vec.unsqueeze(0), mu_hat_obs, Sigma_inv_obs, c).squeeze()
            jac_T_local = jacrev(T_phi_wrapped)(x_vec) # shape: (d_theta, d_x)
            
            # The field is a matrix of shape (d_x, d_theta) whose columns are vector fields
            return w2_local * jac_T_local.T

        # Jacobian of the field will have shape (d_x, d_theta, d_x)
        jac_of_field = jacrev(divergence_field_fn)(xi)
        # The divergence of the k-th column is the trace of its Jacobian matrix.
        divergence = jac_of_field.diagonal(offset=0, dim1=0, dim2=2).sum(dim=1)
        return divergence

    batched_divergences = vmap(get_div_term_single)(x_obs)
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