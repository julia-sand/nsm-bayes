import numpy as np
import torch
from torch.distributions import Binomial, MultivariateNormal, StudentT

####----Function to run MCMC sampling using the NLE likelihood estimator----####
def run_mcmc(
    x_obs: torch.Tensor,
    inference,
    num_pos_samples: int = 250,
    mcmc_method: str = "slice_np",
    num_chains: int = 4,
    num_workers: int = 4,
    thin: int = 10,
    warmup_steps: int = 500,
    init_strategy: str = "proposal"
) -> torch.Tensor:
    """
    Builds a posterior from a trained sbi inference object and runs MCMC.

    Args:
        x_obs (torch.Tensor): The observed data to condition the posterior on.
        inference: The trained sbi inference object (e.g., SNLE).
        num_pos_samples (int): The number of posterior samples to generate per chain.
        mcmc_method (str): The MCMC algorithm to use (e.g., 'slice_np', 'nuts').
        num_chains (int): The number of independent MCMC chains to run.
        thin (int): The thinning factor to reduce sample autocorrelation.
        warmup_steps (int): The number of burn-in steps for each chain.
        init_strategy (str): How to find the starting points for the chains.

    Returns:
        torch.Tensor: A tensor of posterior samples, with shape 
                      [num_chains * num_pos_samples, d_theta].
    """

    posterior = inference.build_posterior(
        sample_with="mcmc",
        mcmc_method=mcmc_method
    )
    
    posterior_samples = posterior.sample(
        sample_shape=(num_pos_samples,), 
        x=x_obs,
        num_chains=num_chains,
        num_workers=num_workers,
        thin=thin,
        warmup_steps=warmup_steps,
        init_strategy=init_strategy
    )
    
    return posterior_samples

def compute_mmd_lengthscale(y: torch.Tensor) -> torch.Tensor:
    """
    Computes the MMD kernel lengthscale using the median heuristic.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
        
    sq_dists = torch.pdist(y) ** 2 
    median_sq_dist = torch.median(sq_dists)
    
    return torch.sqrt(median_sq_dist / 2.0)

def kernel_matrix(x: torch.Tensor, y: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Computes the Gaussian RBF kernel matrix."""
    sq_dists = torch.cdist(x, y, p=2)**2
    return torch.exp(-sq_dists / (2 * l**2))

def compute_mmd(x: torch.Tensor, y: torch.Tensor, lengthscale: torch.Tensor) -> float:
    """
    Computes the V-statistic estimator of the squared MMD.
    """
    if x.dim() == 1: x = x.unsqueeze(-1)
    if y.dim() == 1: y = y.unsqueeze(-1)
    
    m, n = x.shape[0], y.shape[0]

    K_xx = kernel_matrix(x, x, lengthscale)
    K_yy = k-ernel_matrix(y, y, lengthscale)
    K_xy = kernel_matrix(x, y, lengthscale)

    mmd_sq = (1 / (m * m)) * K_xx.sum() + (1 / (n * n)) * K_yy.sum() - (2 / (m * n)) * K_xy.sum()
    
    return mmd_sq.item()

def sample_from_case1_gaussian(mu: torch.Tensor,
                               Sigma: torch.Tensor,
                               num_samples: int) -> torch.Tensor:
    """
    Draw samples from N(mu, Sigma) with small jitter on Sigma if needed.
    Returns (num_samples, d_theta) torch.Tensor.
    """
    mu = mu.float()
    Sigma = Sigma.float()

    try:
        dist = MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    except RuntimeError:
        # add tiny jitter if covariance is near-singular
        d = Sigma.shape[0]
        jitter = 1e-6 * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
        dist = MultivariateNormal(loc=mu, covariance_matrix=Sigma + jitter)

    samps = dist.sample((num_samples,))  # (num_samples, d_theta)
    return samps

# Function to compute the inverse covariance matrix of the data used in objective J(phi)
def compute_inverse_covariance(x, regularize_eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the inverse covariance matrix of the data x in a robust way.

    Args:
        x (torch.Tensor or np.ndarray): 
            Input data. Can be 1D with shape (n_samples,) or 
            2D with shape (n_samples, n_features).
        
        regularize_eps (float): 
            A small epsilon (jitter) added to the diagonal of the covariance 
            matrix for numerical stability before inversion. This helps prevent
            errors from singular matrices.

    Returns:
        torch.Tensor: 
            The inverse covariance matrix (Sigma_inv).
            Shape is (1, 1) for 1D input, or (n_features, n_features) for 2D.
    """
    # 1. Ensure x is a PyTorch tensor and is at least 2D
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        # Ensure float type for covariance calculation
        x = x.to(dtype=torch.float32)

    # Handle the 1D case by reshaping to a 2D column vector
    if x.dim() == 1:
        x = x.unsqueeze(1)

    # 2. Check for sufficient samples
    n_samples, n_features = x.shape
    if n_samples < 2:
        return torch.eye(n_features, device=x.device, dtype=x.dtype)

    # 3. Compute the covariance matrix
    covariance_matrix = torch.cov(x.T)

    # 4. Regularize and Invert the matrix
    identity_matrix = torch.eye(n_features, device=x.device, dtype=x.dtype)
    reg_covariance_matrix = covariance_matrix + regularize_eps * identity_matrix
    
    inverse_cov = torch.linalg.inv(reg_covariance_matrix)

    return inverse_cov


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
