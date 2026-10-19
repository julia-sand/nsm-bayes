from dataclasses import dataclass

#general settings for using mcmc sampling

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
