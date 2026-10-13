"""Shared utilities for MCMC sampling and robust statistics."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Binomial, MultivariateNormal, StudentT


def run_mcmc(
    target_log_prob,
    initial_state,
    n_samples=10000,
    burn_in=2000,
):
    """Simplified MCMC sampler for testing purposes.

    In a real scenario, this would be replaced by a proper NUTS or Metropolis
    sampler.
    """
    samples = []
    current_state = initial_state
    current_log_prob = target_log_prob(current_state)

    for i in range(n_samples):
        # Propose a new state
        proposal = current_state + torch.randn_like(current_state) * 0.1
        proposal_log_prob = target_log_prob(proposal)

        # Accept or reject
        if torch.log(torch.rand(1)) < proposal_log_prob - current_log_prob:
            current_state = proposal
            current_log_prob = proposal_log_prob

        if i >= burn_in:
            samples.append(current_state.clone())

    return torch.stack(samples)


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

