"""Normalising-flow model for the general NSM-Bayes method.

    The general method uses a normalising flow as its likelihood estimator:
    an sbi ``SNLE`` trainer with a masked autoregressive flow (``"maf"``)
    density estimator. :func:`train_normflow` trains the flow on simulated
    ``(theta, x)`` pairs and :func:`make_nle_logprob` wraps a trained flow
    into the ``f(x, theta) -> (1,)`` log-probability callable expected by the
    shared calibration and score-matching utilities.

    The flow itself is a neural network in ``sbi`` (``NFlowsFlow``), so this
    module only contains the training entry point and the shape-conversion
    helper; no flow parameters live in this repository.
"""
from __future__ import annotations

import torch

from sbi.inference import SNLE
from nsm_bayes.models.general.mcmc_settings import MCMCSettings

class GeneralModel():
    """Generalised model
    """

    def __init__(self,
            prior,
            mcmc_settings,
            density_estimator="maf",
    ):
        self.prior = prior
        if self.prior.loc.dtype != torch.float32:
            raise TypeError("Mismatch in data type. SBI requires a float32 prior")
        self.trainer = SNLE(self.prior,density_estimator=density_estimator)
        self.mcmc = mcmc_settings
        self.estimator = None

    def fit(self, theta_sim, x_sim):
        #fit the model, return trained model 
        #  
        self.estimator = self.trainer.append_simulations(theta_sim.float(), x_sim.float()).train()
        return self

    def log_prob(self, theta_vec, x_obs=None, **kwargs):
        """Evaluate the flow's log-probability, clamping non-finite values.

        Parameters
        ----------
        theta_vec : torch.Tensor
            Parameter vector(s) of shape ``(..., d_theta)``.
        x_obs : torch.Tensor, optional
            Conditioning context of shape ``(n_obs, d_x)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities with a trailing ``(1,)`` dimension,
            clamped to a minimum of -1e10.
        """
        return nle_log_prob_safe(self.estimator, theta_vec, x_obs=x_obs)

    def sample_nle_posterior(self, x_obs, mcmc_method="slice_np"):
        """Sample from the NLE posterior q(theta | x_obs) via MCMC.

        Parameters
        ----------
        x_obs : torch.Tensor
            Observation data of shape ``(n_obs, d_x)``.
        mcmc_method : str, default "slice_np"
            Name of the MCMC method to use.

        Returns
        -------
        torch.Tensor
            Samples from the posterior.
        """
        posterior = self.trainer.build_posterior(
            sample_with="mcmc", mcmc_method=mcmc_method
        )
        return posterior.sample(
            sample_shape=(self.mcmc.num_samples,), x=x_obs,
            num_chains=self.mcmc.num_chains, num_workers=self.mcmc.num_chains,
            thin=self.mcmc.thin, warmup_steps=self.mcmc.warmup_steps,
            init_strategy="proposal",
        )


def nle_log_prob_safe(flow, theta_vec, *, x_obs=None):
    """Safe NLE log-probability: clamp non-finite values to -1e10.

    Parameters
    ----------
    flow : posterior_based_estimator
        A trained NLE posterior estimator with a ``.log_prob`` method.
    theta_vec : torch.Tensor
        Parameter vector(s) of shape ``(..., d_theta)``.
    x_obs : torch.Tensor, optional
        Conditioning context of shape ``(n_obs, d_x)``.

    Returns
    -------
    torch.Tensor
        Log-probabilities with the same leading shape as ``theta_vec``
        and a trailing ``(1,)`` dimension, clamped to a minimum of -1e10.
    """
    if x_obs is not None:
        lp = flow.log_prob(theta_vec, condition=x_obs)
    else:
        lp = flow.log_prob(theta_vec)
    return torch.clamp(lp, min=-1e10)


def make_nle_logprob(flow):
    """Return a callable that evaluates the safe NLE log-probability.

    Parameters
    ----------
    flow : posterior_based_estimator
        A trained NLE posterior estimator with a ``.log_prob`` method.

    Returns
    -------
    callable
        A function with signature ``(theta_vec, x_obs=None) -> torch.Tensor``
        that returns the clamped log-probability.
    """
    def _logprob(theta_vec, x_obs=None, **kwargs):
        return nle_log_prob_safe(flow, theta_vec, x_obs=x_obs)
    return _logprob

