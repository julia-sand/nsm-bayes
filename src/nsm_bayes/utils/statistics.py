"""Robust and sample statistics of observed data for NSM-Bayes."""
import numpy as np
import torch
from sklearn.covariance import MinCovDet

from nsm_bayes.utils.linalg import sigma_inverse

# functions for getting statistics of observed data


def mahalanobis_sq(
    theta: torch.Tensor,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
) -> torch.Tensor:
    """Squared Mahalanobis distances ``(theta - mu)^T Sigma^{-1} (theta - mu)``.

    Parameters
    ----------
    theta : torch.Tensor
        Points of shape ``(M, d)``, or a single point of shape ``(d,)``.
    mu : torch.Tensor
        Mean of shape ``(d,)``.
    Sigma : torch.Tensor
        Covariance of shape ``(d, d)``; inverted with
        :func:`~nsm_bayes.utils.linalg.sigma_inverse`.

    Returns
    -------
    torch.Tensor
        Distances of shape ``(M,)``, or a scalar for a single point.
    """
    Sigma_inv = sigma_inverse(Sigma)
    diff = theta - mu
    return (diff @ Sigma_inv * diff).sum(dim=-1)


#######----Function to compute robust mean and covariance ----######
def robust_mean_cov(x_obs):
    """Robust location and scatter of the observed data.

    Parameters
    ----------
    x_obs : torch.Tensor
        Data of shape ``(n, d_x)``.

    Returns
    -------
    mu_hat : torch.Tensor
        Component-wise median, shape ``(d_x,)``.
    Sigma_hat : torch.Tensor
        Minimum covariance determinant (MCD) estimate from
        :class:`sklearn.covariance.MinCovDet`, shape ``(d_x, d_x)``, on the
        dtype and device of `x_obs`. Falls back to the sample covariance if
        the MCD estimate contains NaNs.

    Notes
    -----
    The location is the median, not the MCD location. The MCD fit runs in
    NumPy on the CPU.
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
    """Sample mean and unbiased sample covariance.

    Parameters
    ----------
    x_obs : torch.Tensor
        Data of shape ``(n, d)``.

    Returns
    -------
    mean : torch.Tensor
        Sample mean, shape ``(d,)``.
    cov : torch.Tensor
        Sample covariance with divisor ``n - 1``, shape ``(d, d)``.

    Raises
    ------
    ValueError
        If `x_obs` is not two-dimensional.
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
    """Location and inverse scatter of the data, for the weight functions.

    Parameters
    ----------
    x_obs : torch.Tensor
        Data of shape ``(n, d_x)``.
    robust : bool
        If true, use :func:`robust_mean_cov` (median and MCD); otherwise
        :func:`sample_mean_and_covariance`.

    Returns
    -------
    mu_hat : torch.Tensor
        Location, shape ``(d_x,)``.
    Sigma_inv : torch.Tensor
        Inverse scatter from :func:`~nsm_bayes.utils.linalg.sigma_inverse`,
        shape ``(d_x, d_x)``.
    """
    if robust:
        mu_hat, Sigma_hat = robust_mean_cov(x_obs)
    else:
        mu_hat, Sigma_hat = sample_mean_and_covariance(x_obs)
    
    return mu_hat, sigma_inverse(Sigma_hat)
