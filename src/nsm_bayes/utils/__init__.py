"""Shared utilities for MCMC sampling and robust statistics."""
from nsm_bayes.utils.misc import set_seed
from nsm_bayes.utils.statistics import sample_mean_and_covariance

__all__ = [
    "sample_mean_and_covariance",
    "set_seed",
]
