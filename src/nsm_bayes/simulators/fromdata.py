"""Helpers for reading pre-simulated data from ``.pkl`` files.

Each file holds one array (``torch.Tensor``, ``numpy.ndarray``, nested list
or anything with a ``to_numpy()`` method); arrays are converted to
``float32``.  Files are read with :mod:`pickle`, which can execute
arbitrary code: only load files you trust.
"""
import pickle

import torch

__all__ = ["load_pkl", "gaussian_mle_from_data"]


def load_pkl(path: str):
    """Load the pickled array at `path` as a float32 tensor."""
    with open(path, "rb") as f:
        return torch.as_tensor(pickle.load(f), dtype=torch.float32)


def gaussian_mle_from_data(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MLE mean ``(d,)`` and covariance ``(d, d)`` of samples of shape ``(n, d)``."""
    mean = theta.mean(dim=0)
    xc = theta - mean
    cov = xc.T @ xc / theta.shape[0]
    return mean, cov
