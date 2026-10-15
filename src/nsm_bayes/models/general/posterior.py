
"""General (normalising-flow) posterior sampling for NSM-Bayes."""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from nsm_bayes.method import ScoreMatchingLogPosterior
from nsm_bayes.models.general.slice_sampler import run_multivariate_slice_sampler_tuned
