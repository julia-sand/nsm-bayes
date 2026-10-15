"""Shared utilities for MCMC sampling and robust statistics."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.func import jacfwd, jacrev

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
    L = safe_cholesky(Sigma_hat) #adds jitter here
    return torch.cholesky_inverse(L)



def _single(fn):
    """Wrap fn: (1, d_x) -> (1, d_out) as (d_x,) -> (d_out,)."""
    return lambda x: fn(x.unsqueeze(0)).squeeze(0)


def jacobian_with_spare(xi, fn):
    """Jacobian of fn at xi, returned twice.

    fn maps (1, d_x) -> (1, d_out), e.g. model.T_phi or model.b_phi.
    jacfwd differentiates the first output (-> Hessian) and passes the
    second through as aux (-> Jacobian), so J is computed only once.
    """
    #fn_single = lambda x: fn(x.unsqueeze(0)).squeeze(0)   # (d_x,) -> (d_out,)
    J = jacrev(_single(fn))(xi)                              # (d_out, d_x)
    return J, J

def jacobian(xi, fn):
    """(d_out, d_x)"""
    return jacrev(_single(fn))(xi)


def jacobian_and_hessian(xi, fn):
    """Jacobian (d_out, d_x) and Hessian (d_out, d_x, d_x) from one forward-over-reverse pass."""
    H, J = jacfwd(jacobian_with_spare, has_aux=True)(xi, fn)
    return J, H

def jacobian_and_laplacian(xi, fn):
    """Jacobian (d_out, d_x) and Laplacian (d_out,)."""
    J, H = jacobian_and_hessian(xi, fn)
    return J, H.diagonal(dim1=-2, dim2=-1).sum(-1)

# Define the calculation for a single data point
def get_cov_term_single(xi, fn, Jac=None):
    #returns a copy of the jacobian for reuse
    if Jac is None:
        Jac = jacobian(xi, fn)

    return Jac @ Jac.T, Jac

# --- 2a. Vectorized Non-Divergence Term (h_n) ---
def get_non_div_term_single(xi, fn1, fn2, Jac1=None, Jac2=None):

    if Jac1 is None:
        Jac1 = jacobian(xi,fn1) # shape: (d_theta, d_x)
    if Jac2 is None:
        Jac2 = jacobian(xi,fn2) # shape: (d_x,)
    
    return Jac1 @ Jac2


# --- 2b. Vectorized Divergence Term (D_n) 
def get_div_term_single(xi,fn,w2,grad_w2):

    # 1. Compute Laplacian of T_phi: Tr(Hess(T_phi))
    # T_phi_wrapped: (d_x) -> (d_theta)
    # Hessian: (d_theta, d_x, d_x)
    jac_T,lap_T  = jacobian_and_laplacian(xi,fn)
    #jacfwd(jacobian_with_spare, has_aux=True)(xi, model.T_phi)   # (d_θ, d_x, d_x), (d_θ, d_x)
    #laplacian_T = torch.diagonal(hess_T, dim1=1, dim2=2).sum(dim=1) # shape: (d_theta,)

    grad_w2 = grad_w2.squeeze(0)
    
    # grad_w2: (d_x,)
    # jac_T: (d_theta, d_x)
    # laplacian_T: (d_theta,)
    divergence = grad_w2 @ jac_T.T + w2 * lap_T
    
    return divergence