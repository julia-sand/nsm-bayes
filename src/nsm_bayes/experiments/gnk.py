"""The g-and-k experiment: problem setup and data generation.

The misspecification is additive outliers: a fraction ``epsilon`` of the
observed values is replaced by values from ``cfg.outlier_values``.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.distributions import MultivariateNormal

from nsm_bayes.experiments.common import ExperimentData
from nsm_bayes.simulators.benchmark_simulators.simulators import GandKSimulator

__all__ = ["build_problem", "generate_data"]


def build_problem(cfg: DictConfig) -> tuple:
    """Build the g-and-k simulator, prior, true parameter and data size.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration (``configs/gnk.yaml``).

    Returns
    -------
    sim : GandKSimulator
        Simulator with the configured prior.
    prior : torch.distributions.MultivariateNormal
        Gaussian prior on ``[A, log(B), g, log(k)]``.
    theta_true : torch.Tensor
        True parameter vector in the same space, shape ``(4,)``.
    d_x : int
        Data dimension, always ``1``.
    """
    dtype = torch.float32
    prior_mean = torch.tensor(cfg.prior_mean, dtype=dtype)
    prior_cov = torch.tensor(cfg.prior_cov, dtype=dtype)
    theta_true = torch.tensor(cfg.theta_true, dtype=dtype)

    sim = GandKSimulator(prior_mean=prior_mean, prior_cov=prior_cov)
    prior = sim.prior
    return sim, prior, theta_true, 1


def generate_data(
    sim: GandKSimulator,
    prior: MultivariateNormal,
    theta_true: torch.Tensor,
    d_x: int,
    cfg: DictConfig,
) -> ExperimentData:
    """Generate one repetition of the g-and-k data.

    Parameters
    ----------
    sim : nsm_bayes.simulators.benchmark_simulators.simulators.GandKSimulator
        The simulator returned by :func:`build_problem`.
    prior : torch.distributions.MultivariateNormal
        The prior returned by :func:`build_problem`.
    theta_true : torch.Tensor
        True parameter vector (shape ``(4,)``).
    d_x : int
        Data dimension (unused here; kept for a uniform signature).
    cfg : omegaconf.DictConfig
        Experiment configuration.

    Returns
    -------
    ExperimentData
        Training data ``(theta, x_sim)`` and observed data; the extras
        hold ``outlier_indices``.
    """
    theta, x_sim = sim.generate_training_data(cfg.num_samples)
    x_obs, x_obs_mis, outlier_indices = sim.generate_contaminated_data(
        theta_true, cfg.n_obs, cfg.epsilon, cfg.outlier_values
    )

    return ExperimentData(
        theta=theta,
        x_sim=x_sim,
        x_obs_mis=x_obs_mis,
        x_obs=x_obs,
        extras={"outlier_indices": outlier_indices},
    )
