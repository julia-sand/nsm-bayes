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

from nsm_bayes.gpc import calibrate_beta
from nsm_bayes.method import ScoreMatchingLogPosterior
from nsm_bayes.models.general.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.utils.statistics import estimate_stats

__all__ = [
    "train_normflow",
    "make_nle_logprob",
    "run_mcmc",
    "calibrate_beta_normflow",
]

def make_normflow_model(
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
    return SNLE(prior, density_estimator=density_estimator)

def train_normflow(theta_sim: torch.Tensor,
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
    
    inference = make_normflow_model(prior,density_estimator)

    #train the inference model
    inference.append_simulations(theta_sim, x_sim).train()
    
    return inference


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


def calibrate_beta_normflow(
    flow,
    x_obs: torch.Tensor,
    prior: MultivariateNormal,
    beta_base: float,
    robust: bool = True,
    c: float = 1.0,
    alpha: float = 0.05,
    B: int = 100,
    T: int = 20,
    beta_min: float = 0.01,
    num_samples: int = 500,
    num_chains: int = 1,
    warmup_steps: int = 500,
    thin: int = 1,
) -> tuple[float, dict]:
    """Calibrate ``beta`` for the general (normalising-flow) method.

    Runs the bootstrap-coverage calibration (``gpc.calibrate_beta``) with
    the flow as likelihood estimator.  ``mu_hat`` and ``Sigma_inv`` are
    computed the same way the general posterior uses them (robust or
    sample statistics, then ``sigma_inverse``) and the posterior samples
    for the initial run and every refresh are drawn with the same
    slice-sampler call the final posterior uses, so the calibration and
    the posterior cannot drift apart.

    Parameters
    ----------
    flow : torch.nn.Module
        Trained likelihood estimator (``log_prob(x, theta)``).
    x_obs : torch.Tensor
        Observed data, shape ``(n, d_x)``.
    prior : torch.distributions.MultivariateNormal
        Prior distribution over the parameters.
    beta_base : float
        Starting value of ``beta``; the initial slice-sampler run is drawn
        at this value.
    robust : bool, default True
        If true use robust (median / MCD) data statistics, otherwise plain
        sample moments.
    c : float, default 1.0
        Scale parameter of the IMQ weight.
    alpha : float, default 0.05
        Significance level; ``1 - alpha`` is the target coverage.
    B : int, default 100
        Number of bootstraps per calibration iteration.
    T : int, default 20
        Number of stochastic-approximation iterations.
    beta_min : float, default 0.01
        Lower bound for ``beta``.
    num_samples : int, default 500
        Posterior samples per slice-sampler chain.
    num_chains : int, default 1
        Number of slice-sampler chains.
    warmup_steps : int, default 500
        Warmup steps per chain.
    thin : int, default 1
        Thinning factor.

    Returns
    -------
    tuple[float, dict]
        The calibrated ``beta`` and the history of betas and coverages.
    """
    mu_hat, Sigma_inv = estimate_stats(x_obs, robust)

    theta_samples_base = sample_general_posterior(
        x_obs, prior, flow, beta_base, mu_hat, Sigma_inv,
        num_samples, num_chains, warmup_steps, thin,
    )

    def refresh_sampler_at(beta_new: float) -> torch.Tensor:
        return sample_general_posterior(
            x_obs, prior, flow, beta_new, mu_hat, Sigma_inv,
            num_samples, num_chains, warmup_steps, thin,
        )

    return calibrate_beta(
        theta_samples_base=theta_samples_base,
        beta_base=beta_base,
        x_obs=x_obs,
        q_phi_log_prob=make_nle_logprob(flow),
        mu_hat=mu_hat,
        Sigma_inv=Sigma_inv,
        weight_type="imq",
        c=c,
        alpha=alpha,
        B=B,
        T=T,
        beta_min=beta_min,
        refresh_sampler=refresh_sampler_at,
    )
