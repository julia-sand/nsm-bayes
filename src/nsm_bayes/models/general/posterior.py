
"""General (normalising-flow) posterior sampling for NSM-Bayes."""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from nsm_bayes.method import ScoreMatchingLogPosterior
from nsm_bayes.models.general.slice_sampler import run_multivariate_slice_sampler_tuned

def sample_general_posterior(
    x_obs: torch.Tensor,
    prior,
    flow,
    beta: float,
    mu_hat: torch.Tensor,
    Sigma_inv: torch.Tensor,
    cfg: DictConfig,
) -> torch.Tensor:
    """Draw posterior samples for the general method with the slice sampler."""
    log_posterior = ScoreMatchingLogPosterior(
                    x_obs=x_obs,
                    prior=prior,
                    beta=beta,
                    q_phi_log_prob=flow,
                    mu_hat=mu_hat,
                    Sigma_inv=Sigma_inv,
                    weight_type="imq",
                )
    samples = run_multivariate_slice_sampler_tuned(
        log_posterior_fn=log_posterior,
        prior=prior,
        num_samples=cfg.num_posterior_samples,
        num_chains=cfg.num_chains,
        warmup_steps=cfg.warmup_steps,
        thin=cfg.thin,
    )
    return torch.from_numpy(samples).to(x_obs.dtype).to(x_obs.device)
