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

    def __init__(
        self,
        dt: float = 1.0,
        prior_mean: torch.Tensor | None = None,
        prior_cov: torch.Tensor | None = None,
    ) -> None:
        """Initialize the G and K simulator.

        Parameters
        ----------
        dt : float, default 1.0
            Time step for the simulation.
        prior_mean : torch.Tensor, optional
            Prior mean vector. If None, uses default values.
        prior_cov : torch.Tensor, optional
            Prior covariance matrix. If None, uses default values.
        """
        self.dt = dt

        # Log-space prior mean and covariance
        if prior_mean is not None:
            self.prior_mean = prior_mean
        else:
            self.prior_mean = torch.tensor([0.0, 0.7, 0.0, -1.5], dtype=torch.float32)

        if prior_cov is not None:
            self.prior_cov = prior_cov
        else:
            self.prior_cov = torch.tensor(
                [
                    [5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.25],
                ],
                dtype=torch.float32,
            )
        self.prior = MultivariateNormal(loc=self.prior_mean, 
                                            covariance_matrix=prior_cov)
    
    def simulate(
        self,
        theta: torch.Tensor,
        n: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Simulate data for the g-and-k distribution.

        Parameters
        ----------
        theta : array_like
            Parameter vector [A, log(B), g, log(k)].
        n : int, default 1
            Number of samples to draw per parameter vector.
        **kwargs : dict
            Additional arguments (ignored).

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

    def generate_training_data(
        self,
        n: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate training data by sampling from the prior.

        Parameters
        ----------
        prior : MultivariateNormal
            Prior distribution.
        n : int
            Number of samples.
        **kwargs : dict
            Additional arguments (ignored).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (theta, x_sim).
        """
        theta = self.prior.sample((n,))
        x_sim = self.simulate(theta, n=1)
        return theta, x_sim

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
        term_g = torch.tanh(0.5 * g * z)
        term_k = (1 + z**2) ** k
        x = A + B * (1 + 0.8 * term_g) * term_k * z
        return x.squeeze()

    def generate_observed_data(
        self,
        theta: torch.Tensor,
        n_obs: int,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Generate observed data with potential outliers.

        Parameters
        ----------
        theta : torch.Tensor
            True parameter vector in log space.
        n_obs : int
            Number of receivers to simulate.
        epsilon : float, default 0.0
            Fraction of receivers to replace with pure noise.

        Returns
        -------
        torch.Tensor
            Simulated observed data of shape (n_obs, 1).
        """
        theta_batch = theta.expand(n_obs, -1)
        x_obs = self.simulate(theta_batch, n=1).squeeze(-1)
        
        n_out = int(torch.floor(torch.tensor(epsilon * n_obs)).item())
        outlier_indices = torch.randperm(n_obs)[:n_out]
        
        if n_out > 0:
            theta_out = self.prior.sample((n_out,))
            x_noise = self.simulate(theta_out, n=1).squeeze(-1)
            x_obs[outlier_indices] = x_noise
            
        return x_obs

