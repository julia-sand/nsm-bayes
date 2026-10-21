"""SIR epidemic experiment: problem setup and data generation.

The prior and the true parameter are expressed in the unconstrained space
(log beta, log gamma, logit rho, log I0); this module owns that transform.
The misspecification is prior contamination: a fraction ``epsilon`` of the
observed trajectories is simulated under parameters drawn from the prior.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.distributions import MultivariateNormal

from nsm_bayes.experiments.common import ExperimentData
from nsm_bayes.simulators.add_noise import simulate_contaminated_dataset
from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator

__all__ = ["build_problem", "generate_data"]


def build_problem(cfg: DictConfig) -> tuple:
    """Build the SIR simulator, prior, true parameter and data size.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration (``configs/sir.yaml``).

    Returns
    -------
    tuple
        ``(sim, prior, theta_true, d_x)`` with ``theta_true`` in the
        unconstrained (log/logit) parameter space.
    """
    dtype = torch.float32
    prior_std = torch.tensor(cfg.prior_std, dtype=dtype)
    prior_cov = torch.diag(prior_std**2)

    sim = SIRSimulator(
        obs_model=cfg.obs_model if "obs_model" in cfg else "poisson",
        prior_mean=torch.tensor(list(cfg.prior_mean),dtype=dtype),
        prior_cov=prior_cov,
        T=cfg.T_sir,
        N=cfg.N_sir,
    )

    prior = sim.prior()

    # True parameter in unconstrained space
    theta_true_constrained = torch.tensor(cfg.theta_true, dtype=dtype).unsqueeze(0)
    theta_true = sim.to_unconstrained(theta_true_constrained).squeeze(0)

    return sim, prior, theta_true, cfg.d_x


def generate_data(
    sim: SIRSimulator,
    prior,
    theta_true: torch.Tensor,
    d_x: int,
    cfg: DictConfig,
) -> ExperimentData:
    """Generate one repetition of the SIR data.

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
        Training data ``(theta, x_sim)`` and observed data; the extras
        hold the contamination mask ``is_contam``.
    """

    #training data for surrogate density 
    theta, y_sim = sim.generate_training_data(cfg.num_samples, )
    x_sim = sim.summary(y_sim)

    #clean data set 
    theta_batch = theta_true.unsqueeze(0).repeat(cfg.n_obs, 1)
    y_obs = sim.simulate(theta_batch)
    x_obs = sim.summary(y_obs)

    #mis-specified data set
    y, is_contam = sim.generate_contaminated_data(theta_true=theta_true,
                                               n_obs=cfg.n_obs,
                                               epsilon=cfg.epsilon,)
    x_obs_mis = sim.summary(y)

    return ExperimentData(
        theta=theta,
        x_sim=x_sim,
        x_obs_mis=x_obs_mis,
        x_obs=x_obs,
        extras={"is_contam": is_contam},
    )
