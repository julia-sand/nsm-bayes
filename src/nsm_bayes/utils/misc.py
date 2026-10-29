"""Miscellaneous helpers."""
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Seed the ``numpy`` and ``torch`` global RNGs.

    Parameters
    ----------
    seed : int
        Seed value passed to each of the two RNGs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
