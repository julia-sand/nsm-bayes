"""Neural score-matching losses and posterior for the general method."""
import numpy as np
import torch
from torch.func import grad as fgrad
from torch.func import jvp, vmap

from nsm_bayes.utils.linalg import jacobian, jacobian_and_laplacian
from nsm_bayes.utils.weightfuns import weight_function

# ===========================================================================================
# PART 0: Common functions needed for the proposed neural score-matching Bayes method
# ===========================================================================================



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
        # score wrt x and Hessian trace (Laplacian) wrt x, in one pass
        score, hess_trace = jacobian_and_laplacian(xi, lambda x: nle_log_prob_safe(q_phi_log_prob, x, theta_detached))

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
        if use_hutch:
            # score wrt x
            score = jacobian(xi,log_q_xonly)  # (d_x,)
            # Hutchinson estimate of the Hessian trace
            g = fgrad(log_q_xonly)  # ∇_x log q
            def one_probe(v):
                hvp, _ = jvp(g, (xi,), (v,))    # hvp = H v
                return (hvp * v).sum()
            trH = vmap(one_probe, in_dims=0)(probes).mean()
        else:
            # score wrt x and exact Hessian trace (Laplacian) in one pass
            score, trH = jacobian_and_laplacian(xi, log_q_xonly)

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


