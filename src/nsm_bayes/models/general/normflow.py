"""Normalising-flow model for the general NSM-Bayes method.

    The general method uses a normalising flow as its likelihood estimator:
    an sbi ``SNLE`` trainer with a masked autoregressive flow (``"maf"``)
    density estimator. :func:`train_normflow` trains the flow on simulated
    ``(theta, x)`` pairs and :func:`make_nle_logprob` wraps a trained flow
    into the ``f(x, theta) -> (1,)`` log-probability callable expected by the
    shared calibration and score-matching utilities.

    The flow itself is a neural network in ``sbi`` (``NFlowsFlow``), so this
    module only contains the training entry point and the shape-conversion
    helper; no flow parameters live in this repository.
"""
from __future__ import annotations

from typing import Callable

import torch
from sbi.inference import SNLE
from torch.distributions import MultivariateNormal

__all__ = ["train_normflow", "make_nle_logprob"]


def train_normflow(
    theta_sim: torch.Tensor,
    x_sim: torch.Tensor,
    prior: MultivariateNormal,
    density_estimator: str = "maf",
) -> "SNLE":
    """Train a normalising-flow likelihood estimator on simulated data.

    Parameters
    ----------
    theta_sim : torch.Tensor
        Simulated parameters with shape ``(m, d_theta)``, drawn from `prior`.
    x_sim : torch.Tensor
        Simulated data with shape ``(m, d_x)`` corresponding to `theta_sim`.
    prior : torch.distributions.MultivariateNormal
        Prior distribution over the parameters.
    density_estimator : str, default "maf"
        Name of the sbi density estimator, e.g. ``"maf"`` or ``"mdn"``.

        Returns
    -------
    torch.nn.Module
        Trained sbi density estimator whose ``log_prob(x, theta)`` evaluates
        ``log q_phi(x | theta)``.
    """
    inference = SNLE(prior, density_estimator=density_estimator)
    return inference.append_simulations(theta_sim, x_sim).train()


def make_nle_logprob(estimator) -> Callable:
    """Wrap a trained flow into an ``f(x, theta) -> (1,)`` log-probability callable.

    The returned callable is compatible with the shared utilities
    (``calibrate_beta`` in ``gpc.py`` and the score-matching loss in
    ``method.py``):

    - accepts `x` with shape ``(d_x,)`` or ``(1, d_x)``
    - accepts `theta` with shape ``(d_theta,)`` or ``(1, d_theta)``
    - handles device/dtype conversion
    - reshapes to the ``(sample_dim, batch_dim, d_x)`` convention used by
      SBI's ``NFlowsFlow``

    Parameters
    ----------
    estimator : sbi.inference.SNLE
        Trained likelihood estimator whose ``log_prob`` method is wrapped.

    Returns
    -------
    Callable
        ``f(x, theta)`` returning a tensor of shape ``(1,)`` with
        ``log q_phi(x | theta)``.
    """
    # pick device/dtype from the estimator
    p = next(estimator.parameters())
    dev, dt = p.device, p.dtype

    def f(x, theta):
        # ensure 1D event shapes
        x_row = x.reshape(-1)
        th_row = theta.reshape(-1)

        # reshape to (sample_dim=1, batch_dim=1, d_x)
        x_b = x_row.to(device=dev, dtype=dt).reshape(1, 1, -1).contiguous()
        th_b = th_row.to(device=dev, dtype=dt).reshape(1, -1).contiguous()

        # NFlowsFlow.log_prob returns shape (sample_dim, batch_dim) = (1,1)
        out = estimator.log_prob(x_b, th_b).reshape(-1)  # -> (1,)
        return out

    return f

####----Function to run MCMC sampling using the NLE likelihood estimator----####
#this samples the posterior for the normalising flow method. 
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


