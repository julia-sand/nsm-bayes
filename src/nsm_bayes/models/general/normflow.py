"""Normalising-flow likelihood model for the general NSM-Bayes method.

The general method learns a conditional density ``q(x | theta)`` with sbi's
neural likelihood estimation (``SNLE``), by default with a masked
autoregressive flow (``"maf"``). The trained flow is used in two ways:

* as an ordinary likelihood, to sample the baseline NLE posterior
  (:meth:`GeneralModel.sample_nle_posterior`); and
* through its score ``grad_x log q(x | theta)`` in the robust score-matching
  posterior of :class:`~nsm_bayes.models.general.posterior.GeneralPosterior`.

The flow architecture and training loop come from ``sbi``; this module only
wraps them.
"""
from __future__ import annotations

import torch

from sbi.inference import SNLE
from nsm_bayes.models.general.mcmc_settings import MCMCSettings

class GeneralModel():
    """Neural likelihood estimator from sbi, with save/load and sampling.

    Parameters
    ----------
    prior : torch.distributions.Distribution
        Prior over ``theta``. Must expose ``loc`` with dtype
        ``torch.float32``, as sbi requires.
    mcmc_settings : MCMCSettings
        Sampler settings for :meth:`sample_nle_posterior`.
    density_estimator : str, default "maf"
        Density estimator name passed to ``sbi.inference.SNLE``
        (e.g. ``"maf"``, ``"nsf"``, ``"mdn"``).

    Attributes
    ----------
    trainer : sbi.inference.SNLE
        The sbi inference object.
    estimator : torch.nn.Module or None
        Trained conditional density estimator; ``None`` until :meth:`fit`
        or :meth:`load_checkpoint` is called.
    mcmc : MCMCSettings
        The sampler settings.

    Raises
    ------
    TypeError
        If ``prior.loc`` is not ``float32``.
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
        """Train the likelihood estimator on simulated pairs.

        Parameters
        ----------
        theta_sim : torch.Tensor
            Parameters of shape ``(N, d_theta)``.
        x_sim : torch.Tensor
            Simulated data of shape ``(N, d_x)``.

        Returns
        -------
        GeneralModel
            ``self``, with :attr:`estimator` set.

        Notes
        -----
        Both inputs are cast to ``float32``. Training uses sbi's defaults
        (including its own validation split and early stopping).
        """
        #fit the model, return trained model 
        #  
        self.estimator = self.trainer.append_simulations(theta_sim.float(), x_sim.float()).train()
        return self

    def load_checkpoint(self, path):
        """Load a likelihood estimator saved with ``torch.save``.

        Parameters
        ----------
        path : str or pathlib.Path
            File written by ``torch.save(model.estimator, path)``.

        Returns
        -------
        GeneralModel
            ``self``, with :attr:`estimator` set.

        Notes
        -----
        The whole estimator module is pickled, not just a state dict, so the
        file is read with ``weights_only=False``; only load files you trust.
        The estimator is mapped onto the CPU, where it is trained.
        """
        self.estimator = torch.load(path, map_location="cpu", weights_only=False)
        return self

    def log_prob(self, x_obs, theta_vec=None):
        """Evaluate ``log q(x_obs | theta_vec)`` under the trained flow.

        Parameters
        ----------
        x_obs : torch.Tensor
            Data at which the density is evaluated, shape ``(..., d_x)``.
        theta_vec : torch.Tensor
            Conditioning parameters, shape ``(..., d_theta)``. Despite the
            ``None`` default it is required by the estimator.

        Returns
        -------
        torch.Tensor
            Log-densities, in the shape returned by the sbi estimator's
            ``log_prob``, clamped from below at ``-1e10`` so ``-inf`` does
            not propagate.
        """
        lp = self.estimator.log_prob(x_obs,condition=theta_vec)
        
        return torch.clamp(lp, min=-1e10)
    
    def sample_nle_posterior(self, x_obs, mcmc_method="slice_np"):
        """Sample the standard (non-robust) NLE posterior with MCMC.

        The rows of `x_obs` are treated as i.i.d. observations, so the
        target is ``prior(theta) * prod_i q(x_i | theta)``.

        Parameters
        ----------
        x_obs : torch.Tensor
            Observed data of shape ``(n_obs, d_x)``.
        mcmc_method : str, default "slice_np"
            Name of the sbi MCMC method.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(num_samples, d_theta)``.

        Notes
        -----
        Uses :attr:`mcmc` for the number of samples, chains, thinning and
        warm-up, with one worker per chain and chains initialised from the
        proposal (the prior).
        """
        # Pass the estimator explicitly so this also works when it was loaded
        # from a checkpoint rather than trained by ``self.trainer``.
        posterior = self.trainer.build_posterior(
            density_estimator=self.estimator,
            sample_with="mcmc", mcmc_method=mcmc_method
        )
        return posterior.sample(
            sample_shape=(self.mcmc.num_samples,), x=x_obs,
            num_chains=self.mcmc.num_chains, num_workers=self.mcmc.num_chains,
            thin=self.mcmc.thin, warmup_steps=self.mcmc.warmup_steps,
            init_strategy="proposal",
        )

