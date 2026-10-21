"""Turin channel-model experiment: problem setup and data generation.

The parameters live in log space.  The misspecification is noise-only
receivers: a fraction ``epsilon`` of the receivers is replaced by
simulations under random prior parameters.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.distributions import MultivariateNormal

from nsm_bayes.experiments.common import ExperimentData
from nsm_bayes.simulators.benchmark_simulators.simulators import TurinSimulator

__all__ = ["build_problem", "generate_data"]


def build_problem(cfg: DictConfig) -> tuple:
    """Build the Turin simulator, prior, true parameter and data size.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration (``configs/turin.yaml``).

    Returns
    -------
    sim : TurinSimulator
        Simulator configured from ``B_hz``, ``Ns``, ``device`` and, if
        present, ``prior_mean`` and ``prior_cov``.
    prior : torch.distributions.MultivariateNormal
        Gaussian prior in log space.
    theta_true : torch.Tensor
        ``log(cfg.theta_true)``, shape ``(4,)``.
    d_x : int
        ``cfg.d_x``.
    """
    dtype = torch.float32
    theta_true = torch.log(torch.tensor(cfg.theta_true, dtype=dtype))

    sim = TurinSimulator(
        B=cfg.B_hz,
        Ns=cfg.Ns,
        device=cfg.device,
        prior_mean=torch.tensor(list(cfg.prior_mean), dtype=dtype)
        if "prior_mean" in cfg
        else None,
        prior_cov=torch.tensor([list(r) for r in cfg.prior_cov], dtype=dtype)
        if "prior_cov" in cfg
        else None,
    )
    prior = sim.prior
    return sim, prior, theta_true, cfg.d_x


def generate_data(
    sim: TurinSimulator,
    prior: MultivariateNormal,
    theta_true: torch.Tensor,
    d_x: int,
    cfg: DictConfig,
) -> ExperimentData:
    """Generate one repetition of the Turin data.

    Parameters
    ----------
    sim : nsm_bayes.simulators.benchmark_simulators.simulators.TurinSimulator
        The simulator returned by :func:`build_problem`.
    prior : torch.distributions.MultivariateNormal
        The prior returned by :func:`build_problem`.
    theta_true : torch.Tensor
        True parameter vector in log space (shape ``(4,)``).
    d_x : int
        Data dimension (unused here; kept for a uniform signature).
    cfg : omegaconf.DictConfig
        Experiment configuration.

    Returns
    -------
    ExperimentData
        Training data ``(theta, x_sim)`` and the contaminated observations
        ``x_obs_mis``; ``x_obs`` is ``None`` and there are no extras.
    """
    theta, x_sim = sim.generate_training_data(cfg.num_samples)
    
    x_obs_mis = sim.generate_observed_data(
        theta_true, n_obs=cfg.n_obs, epsilon=cfg.epsilon
    )

    return ExperimentData(
        theta=theta,
        x_sim=x_sim,
        x_obs_mis=x_obs_mis,
    )
