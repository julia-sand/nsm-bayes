"""Standalone entry points for the two NSM-Bayes methods.

    nsm_bayes_general  -- general NSM-Bayes: weighted score-matching posterior over a
                          trained likelihood estimator (sbi SNLE/MAF), sampled with the
                          slice sampler, beta calibrated by bootstrap coverage matching.
    nsm_bayes_conj     -- NSM-Bayes-conj ("case 1"): exponential-family likelihood
                          T_phi(x).theta + b_phi(x) with a closed-form Gaussian posterior,
                          beta calibrated by bootstrap coverage matching.

Both functions take exactly the same inputs and return the same NSMResult, so they can
be swapped for one another:

    result = nsm_bayes_general(x_obs, theta_sim, x_sim, prior_mean, prior_cov, config)
    result = nsm_bayes_conj   (x_obs, theta_sim, x_sim, prior_mean, prior_cov, config)

Everything is a thin wrapper around code that already lives in the nsm_bayes package
(method.py, gpc.py, nn_case1.py, slice_sampler.py); the logic follows run_sir.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from sbi.inference import SNLE
from sbi.utils.sbiutils import standardizing_net

from nsm_bayes.gpc import calibrate_beta, calibrate_beta_gpc
from nsm_bayes.method import (
    ScoreMatchingLogPosterior,
    compute_posterior_case1,
    robust_mean_cov,
    w_imq_squared,
)
from nsm_bayes.conj import BphiNet, TphiNet, train_q_phi
from nsm_bayes.slice_sampler import run_multivariate_slice_sampler_tuned

# Method-specific starting values for beta and its lower clamp. These are the values
# used by the SIR / Turin configs; beta is problem dependent, so override them through
# NSMConfig.beta_base / NSMConfig.beta_min when your problem needs a different scale.
_DEFAULT_BETA = {
    "general": {"beta_base": 1e-6, "beta_min": 1e-8},
    "conj": {"beta_base": 1e-2, "beta_min": 1e-4},
}


@dataclass
class NSMConfig:
    """Settings shared by both methods. Fields a method does not use are ignored."""

    # --- beta calibration (both methods) ---
    alpha: float = 0.05            # target credible-set coverage is 1 - alpha
    B: int = 100                   # bootstrap resamples per calibration iteration
    T: int = 20                    # number of calibration iterations
    beta_base: Optional[float] = None   # starting beta; None -> method-specific default
    beta_min: Optional[float] = None    # lower clamp on beta; None -> method-specific default

    # --- posterior draws ---
    # general: slice-sampler settings. conj: only num_posterior_samples is used, to draw
    # samples from the closed-form Gaussian posterior.
    num_posterior_samples: int = 500
    num_chains: int = 1
    warmup_steps: int = 500
    thin: int = 1

    # --- conj only ---
    hidden_dim: int = 128          # hidden width of T_phi / b_phi

    seed: Optional[int] = None


@dataclass
class NSMResult:
    """Common output of both methods (parameters live on the scale of the inputs)."""

    samples: torch.Tensor          # (num_posterior_samples, d_theta)
    mean: torch.Tensor             # (d_theta,)
    cov: torch.Tensor              # (d_theta, d_theta)
    beta: float                    # calibrated beta
    history: dict                  # {'betas': [...], 'coverages': [...]} from calibration
    model: Any = field(repr=False, default=None)
    # `model` is the trained density model; pass it back via `model=` to skip retraining.
    #   general: the sbi likelihood estimator
    #   conj:    dict(T_phi, b_phi, standardizer_x, standardizer_theta, train_history)


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _prepare(x_obs, theta_sim, x_sim, prior_mean, prior_cov, config):
    cfg = config if config is not None else NSMConfig()
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    f32 = dict(dtype=torch.float32)
    x_obs = torch.as_tensor(x_obs).to(**f32)
    theta_sim = torch.as_tensor(theta_sim).to(**f32)
    x_sim = torch.as_tensor(x_sim).to(**f32)
    prior_mean = torch.as_tensor(prior_mean).to(**f32)
    prior_cov = torch.as_tensor(prior_cov).to(**f32)

    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(1)
    if x_sim.dim() == 1:
        x_sim = x_sim.unsqueeze(1)
    d_theta, d_x = prior_mean.shape[0], x_obs.shape[1]
    assert x_obs.dim() == 2 and x_sim.dim() == 2 and theta_sim.dim() == 2, "expected 2-D tensors"
    assert x_sim.shape[1] == d_x, "x_sim and x_obs must have the same dimension"
    assert theta_sim.shape == (x_sim.shape[0], d_theta), "theta_sim must be (m, d_theta)"
    assert prior_cov.shape == (d_theta, d_theta), "prior_cov must be (d_theta, d_theta)"
    return cfg, x_obs, theta_sim, x_sim, prior_mean, prior_cov, d_theta, d_x


def _beta_settings(cfg: NSMConfig, method: str):
    d = _DEFAULT_BETA[method]
    beta_base = cfg.beta_base if cfg.beta_base is not None else d["beta_base"]
    beta_min = cfg.beta_min if cfg.beta_min is not None else d["beta_min"]
    return beta_base, beta_min


def _weights_stats(x: torch.Tensor):
    """Robust location/scale used by the IMQ weight function."""
    mu_hat, Sigma_hat = robust_mean_cov(x)
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    Sigma_inv = torch.linalg.inv(Sigma_hat + 1e-6 * eye)
    return mu_hat, Sigma_inv


def _nle_logprob(estimator):
    """f(x, theta) -> shape (1,), the form calibrate_beta expects (same as run_sir.py)."""
    p = next(estimator.parameters())
    dev, dt = p.device, p.dtype

    def f(x, theta):
        x_b = x.reshape(-1).to(device=dev, dtype=dt).reshape(1, 1, -1).contiguous()
        th_b = theta.reshape(-1).to(device=dev, dtype=dt).reshape(1, -1).contiguous()
        return estimator.log_prob(x_b, th_b).reshape(-1)

    return f


def _summarise(samples: torch.Tensor):
    mean = samples.mean(dim=0)
    cov = torch.cov(samples.T) if samples.shape[1] > 1 else samples.var(dim=0).reshape(1, 1)
    return mean, cov


# ----------------------------------------------------------------------------------
# General NSM-Bayes
# ----------------------------------------------------------------------------------
def nsm_bayes_general(
    x_obs: torch.Tensor,        # (n, d_x)  observed data (possibly contaminated)
    theta_sim: torch.Tensor,    # (m, d_theta) simulation parameters, drawn from the prior
    x_sim: torch.Tensor,        # (m, d_x)  simulated data for theta_sim
    prior_mean: torch.Tensor,   # (d_theta,)
    prior_cov: torch.Tensor,    # (d_theta, d_theta)
    config: Optional[NSMConfig] = None,
    model: Optional[Any] = None,   # trained likelihood estimator from a previous result
) -> NSMResult:
    """General NSM-Bayes. Trains an sbi SNLE/MAF likelihood estimator on (theta_sim, x_sim)
    unless `model` is given, then samples the weighted score-matching posterior with the
    slice sampler and calibrates beta by bootstrap coverage matching."""
    cfg, x_obs, theta_sim, x_sim, prior_mean, prior_cov, d_theta, d_x = _prepare(
        x_obs, theta_sim, x_sim, prior_mean, prior_cov, config
    )
    beta_base, beta_min = _beta_settings(cfg, "general")
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    c = 1.0

    if model is None:
        inference = SNLE(prior, density_estimator="maf")
        model = inference.append_simulations(theta_sim, x_sim).train()

    mu_hat, Sigma_inv = _weights_stats(x_obs)

    def sample_at(beta: float) -> torch.Tensor:
        log_post = ScoreMatchingLogPosterior(
            x_obs=x_obs, prior=prior, beta=beta, q_phi_log_prob=model,
            mu_hat=mu_hat, Sigma_inv=Sigma_inv, c=c, weight_type="imq",
        )
        arr = run_multivariate_slice_sampler_tuned(
            log_posterior_fn=log_post, prior=prior,
            num_samples=cfg.num_posterior_samples, num_chains=cfg.num_chains,
            warmup_steps=cfg.warmup_steps, thin=cfg.thin,
        )
        return torch.from_numpy(arr).to(x_obs.dtype).to(x_obs.device)

    theta_samples_base = sample_at(beta_base)

    beta, history = calibrate_beta(
        theta_samples_base=theta_samples_base,
        beta_base=beta_base,
        x_obs=x_obs,
        q_phi_log_prob=_nle_logprob(model),
        mu_hat=mu_hat,
        Sigma_inv=Sigma_inv,
        weight_type="imq",
        c=c,
        alpha=cfg.alpha,
        B=cfg.B,
        T=cfg.T,
        step_schedule=lambda t: 10.0 / (t + 10.0),
        beta_min=beta_min,
        refresh_sampler=sample_at,
    )

    samples = sample_at(beta)
    mean, cov = _summarise(samples)
    return NSMResult(samples=samples, mean=mean, cov=cov, beta=float(beta),
                     history=history, model=model)


# ----------------------------------------------------------------------------------
# NSM-Bayes-conj
# ----------------------------------------------------------------------------------
def nsm_bayes_conj(
    x_obs: torch.Tensor,        # (n, d_x)  observed data (possibly contaminated)
    theta_sim: torch.Tensor,    # (m, d_theta) simulation parameters, drawn from the prior
    x_sim: torch.Tensor,        # (m, d_x)  simulated data for theta_sim
    prior_mean: torch.Tensor,   # (d_theta,)
    prior_cov: torch.Tensor,    # (d_theta, d_theta)
    config: Optional[NSMConfig] = None,
    model: Optional[Any] = None,   # trained networks from a previous result
) -> NSMResult:
    """NSM-Bayes-conj. Trains T_phi / b_phi by score matching on standardized
    (theta_sim, x_sim) unless `model` is given, calibrates beta, and returns the closed-form
    Gaussian posterior. `samples` are draws from that Gaussian (num_posterior_samples)."""
    cfg, x_obs, theta_sim, x_sim, prior_mean, prior_cov, d_theta, d_x = _prepare(
        x_obs, theta_sim, x_sim, prior_mean, prior_cov, config
    )
    beta_base, beta_min = _beta_settings(cfg, "conj")

    if model is None:
        T_phi = TphiNet(d_x, cfg.hidden_dim, d_theta)
        b_phi = BphiNet(d_x, cfg.hidden_dim)
        standardizer_x = standardizing_net(x_sim)
        standardizer_theta = standardizing_net(theta_sim)
        train_history = train_q_phi(
            x_sim=standardizer_x(x_sim),
            theta=standardizer_theta(theta_sim),
            T_phi_net=T_phi,
            b_phi_net=b_phi,
        )
        model = dict(T_phi=T_phi, b_phi=b_phi, standardizer_x=standardizer_x,
                     standardizer_theta=standardizer_theta, train_history=train_history)

    T_phi, b_phi = model["T_phi"], model["b_phi"]
    standardizer_x, standardizer_theta = model["standardizer_x"], model["standardizer_theta"]

    beta, history = calibrate_beta_gpc(
        x_obs=x_obs,
        T_phi_net=T_phi,
        b_phi_net=b_phi,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        standardizer_x=standardizer_x,
        standardizer_theta=standardizer_theta,
        initial_beta=beta_base,
        target_coverage=1.0 - cfg.alpha,
        num_iterations=cfg.T,
        num_bootstraps=cfg.B,
        learning_rate_fn=lambda t: 10.0 / (t + 10.0),
        beta_min=beta_min,
    )

    # Closed-form posterior in standardized space, then map back to the original scale.
    x_obs_n = standardizer_x(x_obs)
    prior_mean_n = standardizer_theta(prior_mean)
    scales = standardizer_theta.std
    prior_cov_n = prior_cov / torch.outer(scales, scales)
    mu_hat_obs, Sigma_inv_obs = _weights_stats(x_obs_n)

    mu_n, Sigma_n = compute_posterior_case1(
        x_obs_n, T_phi, b_phi, beta, prior_mean_n, prior_cov_n,
        w_imq_squared, mu_hat_obs, Sigma_inv_obs, 1.0,
    )
    mean = (mu_n * standardizer_theta.std + standardizer_theta.mean).detach()
    cov = (Sigma_n * torch.outer(scales, scales)).detach()
    cov = 0.5 * (cov + cov.T)

    samples = MultivariateNormal(mean, covariance_matrix=cov).sample((cfg.num_posterior_samples,))
    return NSMResult(samples=samples, mean=mean, cov=cov, beta=float(beta),
                     history=history, model=model)