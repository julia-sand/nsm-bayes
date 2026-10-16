"""Robust and sample statistics of observed data for NSM-Bayes."""
import numpy as np
import torch
from sklearn.covariance import MinCovDet

from nsm_bayes.utils.linalg import sigma_inverse

# functions for getting statistics of observed data


#######----Function to compute robust mean and covariance ----######
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

def estimate_stats(x_obs: torch.Tensor, robust: bool):
    """Return ``(mu_hat, Sigma_inv)`` for the observed data.

    Mirrors the statistics used by the general-method posterior: robust
    (median / MCD) estimates when `robust` is true, plain sample moments
    otherwise, with the covariance inverted by :func:`sigma_inverse`.

    Returns mean and inverse covariance
    """
    if robust:
        mu_hat, Sigma_hat = robust_mean_cov(x_obs)
    else:
        mu_hat, Sigma_hat = sample_mean_and_covariance(x_obs)
    
    return mu_hat, sigma_inverse(Sigma_hat)
