"""Shared driver, method steps and helpers for the experiment scripts.

Every experiment under ``scripts/`` is a thin wrapper around
:func:`run_experiment`: it builds the problem once, then for each
repetition it seeds the RNGs, generates *all* data (training and observed),
saves it and runs one method step.  All data is generated before any
training, so the conjugate and normalising-flow variants of the same
experiment produce bit-identical data for the same seed.
"""
from __future__ import annotations

import logging
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator

import torch
from omegaconf import DictConfig

from nsm_bayes.inference.conjugate.calibrate_beta import calibrate_beta_gpc
from nsm_bayes.inference.general.calibrate import calibrate_beta
from nsm_bayes.models.conjugate import ConjugateModel
from nsm_bayes.models.conjugate.conj_posterior import ConjugatePosterior
from nsm_bayes.models.general.normflow import GeneralModel
from nsm_bayes.models.general.posterior import GeneralPosterior
from nsm_bayes.models.general.mcmc_settings import MCMCSettings
from nsm_bayes.utils.statistics import estimate_stats
from nsm_bayes.utils.linalg import sigma_inverse

from nsm_bayes.utils import set_seed

__all__ = [
    "ExperimentData",
    "run_experiment",
    "run_normflow_methods",
    "run_conjugate_method",
    "save",
    "set_seed",
    "timed",
]

_log = logging.getLogger(__name__)


@dataclass
class ExperimentData:
    """All data generated for one repetition of an experiment.

    Attributes
    ----------
    theta : torch.Tensor
        Training parameters, shape ``(num_samples, d_theta)``.
    x_sim : torch.Tensor
        Training data, shape ``(num_samples, d_x)``.
    x_obs_mis : torch.Tensor
        Misspecified observed data, shape ``(n_obs, d_x)``.
    x_obs : torch.Tensor, optional
        Clean (unmisspecified) observed data, if the experiment has one.
    extras : dict
        Experiment-specific objects to save, e.g. ``outlier_indices``,
        ``y_obs`` or ``is_contam``.
    """

    theta: torch.Tensor
    x_sim: torch.Tensor
    x_obs_mis: torch.Tensor
    x_obs: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)

# The method step signature: (data, prior, cfg, save_dir, rep) -> None.
MethodStep = Callable[[ExperimentData, Any, DictConfig, Path, int], None]

def save(save_dir: Path, name: str, rep: int, obj: Any) -> None:
    """Pickle ``obj`` to ``save_dir / f"{name}_{rep}.pkl"``."""
    with open(save_dir / f"{name}_{rep}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


@contextmanager
def timed(save_dir: Path, name: str, rep: int) -> Generator[None]:
    """Time a block and save the elapsed seconds as ``cost_{name}_{rep}.pkl``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        save(save_dir, f"cost_{name}", rep, time.perf_counter() - start)


def run_normflow_methods(
    data: ExperimentData,
    prior,
    cfg: DictConfig,
    save_dir: Path,
    rep: int,
) -> None:
    """Run the general (normalising-flow) method for one repetition.

    Trains the flow on the simulation data, samples the baseline NLE
    posterior, calibrates ``beta`` and finally samples the calibrated
    general posterior.  Saves ``likelihood_estimator_full``,
    ``cost_nle_training``, ``samples_nle_mis``, ``cost_nle_mcmc``,
    ``samples_sm``, ``cost_sm_mcmc``, ``beta``, ``gpc_history``,
    ``posterior_mean`` and ``posterior_covariance`` for this repetition.
    """
    x_obs_mis = data.x_obs_mis


    #set mcmc config
    mcmc = MCMCSettings.from_cfg(cfg)

    flow_model = GeneralModel(prior, mcmc_settings=mcmc, density_estimator=cfg.get("density_estimator", "maf"))

    #train the surrogate model
    with timed(save_dir, "nle_training", rep):
        flow_model.fit(
            data.theta,
            data.x_sim,
        )
    torch.save(flow_model.estimator, save_dir / f"likelihood_estimator_full_{rep}.pt")

    #sample the baseline posterior
    with timed(save_dir, "nle_mcmc", rep):
        samples_nle_mis = flow_model.sample_nle_posterior(
            x_obs_mis,
        )

    save(save_dir, "samples_nle_mis", rep, samples_nle_mis)

    #estimate the robust statistics
    #set up a posterior for calibration
    beta_posterior = GeneralPosterior(flow_model=flow_model, x_obs=x_obs_mis, prior=prior,
                                mcmc=mcmc, robust=cfg.robust_flag)
    
    #calibrate beta / temperature
    with timed(save_dir, "sm_mcmc", rep):
        theta_samples_base = beta_posterior.sample(cfg.beta_base_general)

        beta, history = calibrate_beta(beta_posterior, theta_samples_base, 
                                        beta_base=cfg.beta_base_general,
                                        alpha=cfg.alpha, B=cfg.B, T=cfg.T,
                                        beta_min=cfg.get("beta_min_general", 0.01))
        
        samples_sm_general = beta_posterior.sample(beta)

    save(save_dir, "samples_sm", rep, samples_sm_general)
    save(save_dir, "gpc_history", rep, history)
    save(save_dir, "beta", rep, beta)

    # save the posterior moments of the calibrated general samples
    if samples_sm_general.ndim == 2 and samples_sm_general.shape[0] > 1:
        mu_hat, Sigma_hat = estimate_stats(samples_sm_general, cfg.robust_flag)
        save(save_dir, "posterior_mean", rep, mu_hat)
        save(save_dir, "posterior_covariance", rep, Sigma_hat)


def run_conjugate_method(
    data: ExperimentData,
    prior,
    cfg: DictConfig,
    save_dir: Path,
    rep: int,
) -> None:
    """Run the conjugate (score-matching) method for one repetition.

    Fits a :class:`~nsm_bayes.models.conjugate.conj_model.ConjugateModel`
    on the simulation data, calibrates ``beta`` with GPC and computes the
    calibrated posterior.  Saves ``conjugate_nets``,
    ``cost_sm_train``, ``posterior_mean``, ``posterior_covariance``,
    ``gpc_history``, ``beta`` and ``cost_sm_posterior`` for this repetition.
    """
    x_obs_mis = data.x_obs_mis
    d_theta = data.theta.shape[1]

    #define the energy model
    model = ConjugateModel(
        d_x=cfg.d_x, d_theta=d_theta, hidden_dim=cfg.hidden_dim)
    dev = x_obs_mis.device
    model = model.to(device=dev, dtype=x_obs_mis.dtype)

    #train the model
    with timed(save_dir, "sm_train", rep):
        model.fit(x_sim=data.x_sim, theta_sim=data.theta)
    
    model.save_checkpoint(cfg, save_dir, rep)

    posterior = ConjugatePosterior(
        model,
        prior.loc,
        prior.covariance_matrix,
        robust=cfg.robust_flag,
        weight_type="imq",
    )

    with timed(save_dir, "sm_posterior", rep):
        #calibrate the temperature
        beta, history = calibrate_beta_gpc(
            x_obs_mis,
            posterior,
            initial_beta=cfg.beta_base_case1,
            target_coverage=1.0 - cfg.alpha,
            num_iterations=cfg.T,
            num_bootstraps=cfg.B,
            beta_min=cfg.get("beta_min_case1", 0.01),
        )

        #compute robust posterior
        mu_n, Sigma_n = posterior.calibrated_posterior(
            x_obs_mis,
            beta)

    save(save_dir, "posterior_mean", rep, mu_n)
    save(save_dir, "posterior_covariance", rep, Sigma_n)
    save(save_dir, "gpc_history", rep, history)
    save(save_dir, "beta", rep, beta)


def run_experiment(
    cfg: DictConfig,
    problem_module,
    method_step: MethodStep,
    save_dir: Path,
) -> None:
    """Run one experiment: loop over repetitions and drive the method step.

    For each repetition the RNGs are seeded, all data (training and
    observed) is generated and saved first, and only then is `method_step`
    called, so both variants of an experiment get the same data.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration (see ``configs/*.yaml``).
    problem_module : module
        One of :mod:`nsm_bayes.experiments.gnk`, ``sir``,
        ``sir_undercounting`` or ``turin``; provides ``build_problem`` and
        ``generate_data``.
    method_step : callable
        ``(data, prior, cfg, save_dir, rep) -> None``; the method variant
        for this script.
    save_dir : pathlib.Path
        Directory the data and method artifacts are written to.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    sim, prior, theta_true, d_x = problem_module.build_problem(cfg)

    for rep in range(cfg.num_repeat):
        set_seed(cfg.seed_offset + rep)
        data = problem_module.generate_data(sim, prior, theta_true, d_x, cfg)

        save(save_dir, "theta", rep, data.theta)
        save(save_dir, "x_sim", rep, data.x_sim)
        if data.x_obs is not None:
            save(save_dir, "x_obs", rep, data.x_obs)
        save(save_dir, "x_obs_mis", rep, data.x_obs_mis)
        for name, obj in data.extras.items():
            save(save_dir, name, rep, obj)

        _log.info("Iteration number: %d", rep)
        method_step(data, prior, cfg, save_dir, rep)
