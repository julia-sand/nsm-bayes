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
    sim : SIRSimulator
        See :func:`nsm_bayes.experiments.sir.build_problem`.
    prior : torch.distributions.MultivariateNormal
        Gaussian prior in the unconstrained space.
    theta_true : torch.Tensor
        True parameter vector in the unconstrained space.
    d_x : int
        ``cfg.d_x``.
    """
    return sir.build_problem(cfg)


def generate_data(
    sim: SIRSimulator,
    prior,
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
        Summary statistics for training (``x_sim``), clean observations
        (``x_obs``) and their undercounted version (``x_obs_mis``); the
        extras hold the clean trajectories ``y_obs`` and the contamination
        mask ``is_contam``.

    Notes
    -----
    Unlike the contamination experiment, ``x_obs_mis`` is derived from the
    same trajectories as ``x_obs``.
    """
    
    #training data for surrogate density 
    theta, y_sim = sim.generate_training_data(cfg.num_samples, )
    x_sim = sim.summary(y_sim)

    #clean data set 
    theta_batch = theta_true.unsqueeze(0).repeat(cfg.n_obs, 1)
    y_obs = sim.simulate(theta_batch)
    x_obs = sim.summary(y_obs)

    #mis-specified data set
    y_obs_mis, is_contam = sim.apply_undercounting_trajectory(y_obs,
                                               epsilon=cfg.epsilon,
                                               q = cfg.q)
    x_obs_mis = sim.summary(y_obs_mis)

    return ExperimentData(
        theta=theta,
        x_sim=x_sim,
        x_obs_mis=x_obs_mis,
        x_obs=x_obs,
        extras={"y_obs": y_obs, "is_contam": is_contam}, 
    )


