import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Seed the random, numpy and torch global RNGs with `seed`.

    Parameters
    ----------
    seed : int
        Seed value passed to each of the three RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

