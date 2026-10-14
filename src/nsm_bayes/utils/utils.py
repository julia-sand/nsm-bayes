"""Shared utilities for MCMC sampling and robust statistics."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.func import jacfwd, jacrev


def sample_mean_and_covariance(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean and covariance of a set of samples.

    Parameters
    ----------
    samples : torch.Tensor
        A tensor of shape (n_samples, d_theta).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The sample mean (d_theta,) and covariance (d_theta, d_theta).
    """
    mean = torch.mean(samples, dim=0)
    diff = samples - mean
    cov = (diff.T @ diff) / (samples.shape[0] - 1)
    return mean, cov



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

def jacobian(fn, xi):
    """(d_out, d_x)"""
    return jacrev(_single(fn))(xi)


def jacobian_and_hessian(fn, xi):
    """Jacobian (d_out, d_x) and Hessian (d_out, d_x, d_x) from one forward-over-reverse pass."""
    H, J = jacfwd(jacobian_with_spare, has_aux=True)(xi,fn)
    return J, H

def jacobian_and_laplacian(fn, xi):
    """Jacobian (d_out, d_x) and Laplacian (d_out,)."""
    J, H = jacobian_and_hessian(fn, xi)
    return J, H.diagonal(dim1=-2, dim2=-1).sum(-1)

def set_seed(seed: int) -> None:
    """Seed the random, numpy and torch global RNGs with `seed`.

    Parameters
    ----------
    seed : int
        Seed value passed to each of the three RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
