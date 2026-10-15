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

from nsm_bayes.gpc import calibrate_beta_gpc
from nsm_bayes.models.conjugate import ConjugateModel
from nsm_bayes.models.general import train_normflow
from nsm_bayes.models.general.normflow import (
    sample_general_posterior,
    calibrate_beta_normflow,
    run_mcmc,
)
from nsm_bayes.utils import set_seed
from nsm_bayes.utils.statistics import estimate_stats

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
    ``samples_sm_general``, ``cost_sm_general_mcmc``, ``beta_general`` and
    ``gpc_history_general`` for this repetition.
    """
    x_obs_mis = data.x_obs_mis

    #train the surrogate model
    with timed(save_dir, "nle_training", rep):
        flow = train_normflow(
            data.theta,
            data.x_sim,
            prior,
            density_estimator=cfg.get("density_estimator", "maf"),
        )
    torch.save(flow, save_dir / f"likelihood_estimator_full_{rep}.pt")

    #sample the posterior
    with timed(save_dir, "nle_mcmc", rep):
        samples_nle_mis = run_mcmc(
            x_obs_mis,
            flow,
            num_pos_samples=cfg.num_posterior_samples,
            num_chains=cfg.num_chains,
            num_workers=cfg.num_chains,
            thin=cfg.thin,
            warmup_steps=cfg.warmup_steps,
        )
    save(save_dir, "samples_nle_mis", rep, samples_nle_mis)

    #estimate the robust statistics
    robust = cfg.get("robust_flag", True)
    mu_hat, Sigma_inv = estimate_stats(x_obs_mis, robust)

    #calibrate beta / temperature
    with timed(save_dir, "sm_general_mcmc", rep):
        beta, history = calibrate_beta_normflow(
            flow,
            x_obs_mis,
            prior,
            beta_base=cfg.beta_base_general,
            robust=robust,
            alpha=cfg.alpha,
            B=cfg.B,
            T=cfg.T,
            beta_min=cfg.get("beta_min_general", 0.01),
            num_samples=cfg.num_posterior_samples,
            num_chains=cfg.num_chains,
            warmup_steps=cfg.warmup_steps,
            thin=cfg.thin,
        )
        samples_sm_general = sample_general_posterior(
            x_obs_mis, prior, flow, beta, mu_hat, Sigma_inv, cfg
        )
    save(save_dir, "samples_sm_general", rep, samples_sm_general)
    save(save_dir, "gpc_history_general", rep, history)
    save(save_dir, "beta_general", rep, beta)


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
    ``cost_sm_case1_train``, ``posterior_mean_case1``,
    ``posterior_covariance_case1``, ``gpc_history_case1``,
    ``beta_case1`` and ``cost_sm_case1_posterior`` for this repetition.
    """
    x_obs_mis = data.x_obs_mis
    prior_mean = prior.loc
    prior_cov = prior.covariance_matrix
    d_theta = prior_mean.shape[0]

    #define the energy model
    model = ConjugateModel(
        d_x=cfg.d_x, d_theta=d_theta, hidden_dim=cfg.hidden_dim
    )
    dev = x_obs_mis.device
    model = model.to(device=dev, dtype=x_obs_mis.dtype)

    #train the model
    with timed(save_dir, "sm_case1_train", rep):
        model.fit(x_sim=data.x_sim, theta_sim=data.theta)
    model.save_checkpoint(cfg, save_dir, rep)

    with timed(save_dir, "sm_case1_posterior", rep):
        #calibrate the temperature
        beta, history = calibrate_beta_gpc(
            x_obs_mis,
            model,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            initial_beta=cfg.beta_base_case1,
            target_coverage=1.0 - cfg.alpha,
            num_iterations=cfg.T,
            num_bootstraps=cfg.B,
            beta_min=cfg.get("beta_min_case1", 0.01),
        )
        #compute robust posterior
        mu_n, Sigma_n = model.calibrated_posterior(
            x_obs_mis,
            beta,
            prior_mean,
            prior_cov,
            cfg.get("robust_flag", True),
        )
    save(save_dir, "posterior_mean_case1", rep, mu_n)
    save(save_dir, "posterior_covariance_case1", rep, Sigma_n)
    save(save_dir, "gpc_history_case1", rep, history)
    save(save_dir, "beta_case1", rep, beta)


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
