"""Generalised (score-matching) posterior for the normalising-flow method."""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.func import vmap

from nsm_bayes.method import nle_log_prob_safe
from nsm_bayes.inference.general.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.utils.linalg import jacobian_and_laplacian
from nsm_bayes.utils.statistics import estimate_stats
from nsm_bayes.utils.weights import weight_function

_log = logging.getLogger(__name__)

@dataclass(frozen=True)
class MCMCSettings:
    """General configuration of an MCMC run."""

    num_samples: int = 500
    num_chains: int = 1
    warmup_steps: int = 500
    thin: int = 1

    @classmethod
    def from_cfg(cls, cfg) -> "MCMCSettings":
        """Build from the existing top-level config keys."""
        return cls(cfg.num_posterior_samples, cfg.num_chains, cfg.warmup_steps, cfg.thin)


class GeneralPosterior:
    """Generalised posterior  p_β(θ | x) ∝ π(θ) · exp(-β · n · L_NSM(θ; x)).

    Bundles everything the general method needs. The robust weights are fitted once, on raw
    x_obs (the flow's score is taken with respect to raw x).
    """

    def __init__(self, x_obs,
                    flow,  #trained density estimate
                    prior, 
                    robust: bool,
                    weight_type: str = "imq", 
                    c: float = 1.0):
        
        self.flow = flow
        self.x_obs = x_obs
        self.prior = prior
        self.mcmc = MCMCSettings() #get the mcmc settings
        self.robust, self.weight_type, self.c = robust, weight_type, c
        self.n_obs = x_obs.shape[0]

        self.mu_hat, self.Sigma_inv = estimate_stats(x_obs, robust)
        self.w2, self.grad_w2 = weight_function(
            weight_type, self.x_obs, self.mu_hat, self.Sigma_inv, c
        )

    def _per_x_sm_loss(self, xi, w2_i, grad_w2_i, theta):
        """Weighted SM loss for one observation `xi` at one `theta`.

        Parameters
        ----------
        xi : torch.Tensor
            A single observation of shape ``(d_x,)``.
        w2_i : torch.Tensor
            Squared weight at `xi`, a scalar.
        grad_w2_i : torch.Tensor
            Weight gradient at `xi`, shape ``(d_x,)``.
        theta : torch.Tensor
            Parameter vector of shape ``(d_theta,)``.

        Returns
        -------
        torch.Tensor
            Scalar score-matching loss contribution for this observation.
        """
        log_q = functools.partial(nle_log_prob_safe, self.flow, theta_vec=theta)
        score, lap = jacobian_and_laplacian(xi, log_q)  # (d_x,), scalar
        return w2_i * (score @ score) + 2.0 * (grad_w2_i @ score) + 2.0 * w2_i * lap

    def _losses_one_theta(self, theta):
        """(n,) per-observation losses at one `theta`; differentiable in `theta`."""
        return vmap(self._per_x_sm_loss, in_dims=(0, 0, 0, None))(
            self.x_obs, self.w2, self.grad_w2, theta
        )

    def sm_loss(self, theta):
        """Scalar `L_NSM(theta)`; used by `log_prob` and `nsm_minimiser`."""
        return self._losses_one_theta(theta).mean()

    @torch.no_grad()
    def per_x_losses(self, theta_samples):
        """ell (M, n) and L_orig (M,) for the importance-reweighting cache.

        Vectorised over both the M sampled parameters and the n observations
        via a nested ``vmap`` (``theta`` outer, ``x`` inner), avoiding the
        previous Python loop over M ``vmap``ed batches.
        """
        ell = vmap(lambda th: self._losses_one_theta(th))(theta_samples)
        return ell, ell.mean(dim=1)

    @torch.no_grad()
    def log_prob(self, theta_np: np.ndarray, beta: float) -> float:
        """log π(θ) − β·n·L_NSM(θ); -inf outside the prior's support. For the slice sampler."""
        theta = torch.as_tensor(theta_np, dtype=self.x_obs.dtype, device=self.x_obs.device)
        lp = self.prior.log_prob(theta).sum()
        if torch.isneginf(lp):
            return -np.inf
        return float(lp - beta * self.n_obs * self.sm_loss(theta))

    def sample(self, beta: float) -> Tensor:
        """Draw posterior samples at `beta` with the slice sampler."""
        samples = run_multivariate_slice_sampler_tuned(
                            log_posterior_fn=functools.partial(self.log_prob, beta=beta),
                            prior=self.prior,
                            num_samples=self.mcmc.num_samples,
                            num_chains=self.mcmc.num_chains,
                            warmup_steps=self.mcmc.warmup_steps,
                            thin=self.mcmc.thin,
                        )
        return torch.from_numpy(samples).to(dtype=self.x_obs.dtype, device=self.x_obs.device)

    def compute_theta_hat_true(self, theta_init: Tensor, num_steps: int = 200, lr: float = 1e-2) -> Tensor:
        """θ̂ = argmin_θ L_NSM(θ; x_obs) by Adam, started from `theta_init`."""
        theta = theta_init.detach().clone().to(dtype=self.x_obs.dtype, device=self.x_obs.device)
        theta.requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self.sm_loss(theta)
            loss.backward()
            optimizer.step()
            if step % 20 == 0 or step == num_steps - 1:
                _log.debug("nsm_minimiser step %d | L_NSM = %.4f", step, loss.item())
        return theta.detach()
