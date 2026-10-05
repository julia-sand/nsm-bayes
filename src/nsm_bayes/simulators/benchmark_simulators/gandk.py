"""
This module contains functions for simulating the g-and-k distribution.
"""

import torch


def sample_gandk_fully_reparameterized(gamma: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Sample from the g-and-k distribution using a fully reparameterized vector.

    Parameters
    ----------
    gamma : torch.Tensor
        A 1D tensor of shape (4,) containing the
        TRANSFORMED parameters [A, log(B), g, log(k)].
    n : int, default 1
        Number of samples to draw.

    Returns
    -------
    torch.Tensor
        A single scalar data point x sampled from the distribution.
    """
    # --- 1. Unpack and Transform Parameters ---
    # The model works with gamma = [A, log(B), g, log(k)]
    A = gamma[0]
    logB = gamma[1]
    g = gamma[2]
    logk = gamma[3]

    # Transform constrained parameters back to their original space
    B = torch.exp(logB)
    k = torch.exp(logk)

    z = torch.randn(n, device=gamma.device)
    # exp_gz = torch.exp(-g * z).clamp(min=1e-30, max=1e30)
    # term_g = (1 - exp_gz) / (1 + exp_gz)
    term_g = torch.tanh(0.5 * g * z)
    term_k = (1 + z**2)**k
    x = A + B * (1 + 0.8 * term_g) * term_k * z
    return x.squeeze()
