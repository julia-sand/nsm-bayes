"""Settings shared by the MCMC samplers of the general method."""
from dataclasses import dataclass

#general settings for using mcmc sampling

@dataclass(frozen=True)
class MCMCSettings:
    """Immutable configuration of an MCMC run.

    Used both for sbi's NLE posterior sampling and for the slice sampler of
    :class:`~nsm_bayes.models.general.posterior.GeneralPosterior`.

    Attributes
    ----------
    num_samples : int, default 500
        Number of samples to return. For the slice sampler this is per
        chain (exact only when ``thin == 1``); for sbi it is the total.
    num_chains : int, default 1
        Number of independent chains.
    warmup_steps : int, default 500
        Number of warm-up (burn-in) steps per chain.
    thin : int, default 1
        Keep every ``thin``-th state.
    """

    num_samples: int = 500
    num_chains: int = 1
    warmup_steps: int = 500
    thin: int = 1

    @classmethod
    def from_cfg(cls, cfg) -> "MCMCSettings":
        """Build settings from an experiment configuration.

        Parameters
        ----------
        cfg : omegaconf.DictConfig or object
            Must provide ``num_posterior_samples``, ``num_chains``,
            ``warmup_steps`` and ``thin`` as attributes.

        Returns
        -------
        MCMCSettings
            Settings with ``num_samples = cfg.num_posterior_samples``.
        """
        return cls(cfg.num_posterior_samples, cfg.num_chains, cfg.warmup_steps, cfg.thin)
