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
