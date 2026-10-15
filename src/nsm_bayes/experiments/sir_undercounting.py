"""SIR undercounting experiment: data generation.

Reuses :func:`nsm_bayes.experiments.sir.build_problem`; the
misspecification is one-sided undercounting of the incidence trajectories
(binomial thinning with retention probability ``cfg.q``).
"""
from __future__ import annotations

from omegaconf import DictConfig
from torch.distributions import MultivariateNormal

from nsm_bayes.experiments import sir
from nsm_bayes.experiments.common import ExperimentData
from nsm_bayes.simulators.add_noise import apply_undercounting_trajectory
from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator

__all__ = ["build_problem", "generate_data"]


def build_problem(cfg: DictConfig) -> tuple:
    """Build the SIR problem (identical to the contamination experiment).

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration (``configs/sir_undercounting.yaml``).

    Returns
    -------
    tuple
        ``(sim, prior, theta_true, d_x)`` as returned by
        :func:`nsm_bayes.experiments.sir.build_problem`.
    """
    return sir.build_problem(cfg)


def generate_data(
    sim: SIRSimulator,
    prior: MultivariateNormal,
    theta_true,
    d_x: int,
    cfg: DictConfig,
) -> ExperimentData:
    """Generate one repetition of the SIR undercounting data.

    Parameters
    ----------
    sim : nsm_bayes.simulators.benchmark_simulators.sir.SIRSimulator
        The simulator returned by :func:`build_problem`.
    prior : torch.distributions.MultivariateNormal
        The prior returned by :func:`build_problem`.
    theta_true : torch.Tensor
        True parameter vector in unconstrained space (shape ``(4,)``).
    d_x : int
        Data dimension (unused here; kept for a uniform signature).
    cfg : omegaconf.DictConfig
        Experiment configuration.

    Returns
    -------
    ExperimentData
        Training data ``(theta, x_sim)`` and the corrupted observed data;
        the extras hold the clean trajectories ``y_obs`` and the
        contamination mask ``is_contam``.
    """
    N = cfg.N_sir
    T = cfg.T_sir
    theta = prior.sample((cfg.num_samples,))
    y_sim = sim.simulate(theta, T=T, N=N)
    x_sim = sim.summary(y_sim, N)

    theta_batch = theta_true.unsqueeze(0).repeat(cfg.n_obs, 1)
    y_obs = sim.simulate(theta_batch, T=T, N=N)
    y_cor, is_contam = apply_undercounting_trajectory(
        y_obs, epsilon=cfg.epsilon, q=cfg.q, per_time=False
    )
    x_obs_mis = sim.summary(y_cor, N)

    return ExperimentData(
        theta=theta,
        x_sim=x_sim,
        x_obs_mis=x_obs_mis,
        extras={"y_obs": y_obs, "is_contam": is_contam},
    )
