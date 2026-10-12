"""
This module contains functions for simulating the g-and-k distribution.
"""

from __future__ import annotations

import torch
from torch.distributions import MultivariateNormal

__all__ = ["GandKSimulator"]


class GandKSimulator:
    """Simulator for the G and K model.

    The model works in the reparameterised space
    ``gamma = [A, log(B), g, log(k)]``, so the prior is defined on the
    same 4-dimensional parameter vector as :meth:`simulate`.

    Attributes
    ----------
    dt : float
        Time step for the simulation.
    """

    def __init__(self, dt: float = 1.0):
        """Initialize the G and K simulator.

        Parameters
        ----------
        dt : float, default 1.0
            Time step for the simulation.
        """
        self.dt = dt

    def prior(self) -> MultivariateNormal:
        """Return the Gaussian prior used in the g-and-k experiments.

        This is the prior defined in ``configs/gnk.yaml`` (``prior_mean``
        and ``prior_cov``), in the reparameterised parameter space
        ``[A, log(B), g, log(k)]``.

        Returns
        -------
        MultivariateNormal
            Prior distribution over the 4 parameters.
        """
        mean = torch.tensor([0.0, 0.7, 0.0, -1.5], dtype=torch.float32)
        cov = torch.tensor(
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.25],
            ],
            dtype=torch.float32,
        )
        return MultivariateNormal(loc=mean, covariance_matrix=cov)

    def simulate(
        self,
        theta: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        """Sample from the g-and-k distribution.

        For each row of `theta` one (or `n`) scalar data points are
        drawn, so the output has the same number of rows as the input.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector (1D, shape ``(4,)``) or batch of parameter
            vectors (2D, shape ``(n_theta, 4)``) containing the
            transformed parameters ``[A, log(B), g, log(k)]``.
        n : int, default 1
            Number of samples to draw per parameter vector.

        Returns
        -------
        torch.Tensor
            Simulated data of shape ``(n_theta,)`` for `n == 1` or
            ``(n_theta, n)`` for `n > 1`.
        """
        theta = torch.as_tensor(theta)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        x = torch.stack(
            [self.sample_gandk_fully_reparameterized(row, n=n) for row in theta],
            dim=0,
        )
        if n == 1:
            return x.squeeze(-1)
        if x.dim() == 2 and x.shape[0] == 1:
            return x.squeeze(0)
        return x

    def sample_gandk_fully_reparameterized(
        self,
        gamma: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        """Sample from the g-and-k distribution using a reparameterised parameter.

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
        term_k = (1 + z**2) ** k
        x = A + B * (1 + 0.8 * term_g) * term_k * z
        return x.squeeze()

