r"""The g-and-k distribution simulator.

The g-and-k distribution is defined through its quantile function: with
:math:`z \sim N(0, 1)`,

.. math::

    x = A + B \left(1 + 0.8 \tanh\!\left(\tfrac{g z}{2}\right)\right)
        \left(1 + z^2\right)^k z .

Parameters are handled in the unconstrained space
:math:`\gamma = (A, \log B, g, \log k)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import MultivariateNormal

from nsm_bayes.simulators.add_noise import add_outliers_by_proportion

__all__ = ["GandKSimulator"]


class GandKSimulator:
    """Simulator for the g-and-k distribution.

    The model works in the reparameterised space
    ``gamma = [A, log(B), g, log(k)]``, and the Gaussian prior is defined on
    the same 4-dimensional vector.

    Parameters
    ----------
    dt : float, default 1.0
        Stored for interface compatibility; not used by the simulator.
    prior_mean : torch.Tensor, optional
        Prior mean of shape ``(4,)``. Defaults to ``[0, 0.7, 0, -1.5]``.
    prior_cov : torch.Tensor, optional
        Prior covariance of shape ``(4, 4)``. Defaults to
        ``diag([5, 0.5, 4, 0.25])``.

    Attributes
    ----------
    prior : torch.distributions.MultivariateNormal
        Prior over ``gamma``.
    """

    def __init__(
        self,
        dt: float = 1.0,
        prior_mean: torch.Tensor | None = None,
        prior_cov: torch.Tensor | None = None,
    ) -> None:
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
                                            covariance_matrix=self.prior_cov)
    
    def simulate(
        self,
        theta: torch.Tensor,
        n: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Draw samples from the g-and-k distribution.

        Parameters
        ----------
        theta : array_like
            Parameters ``[A, log(B), g, log(k)]``, shape ``(4,)`` or
            ``(n_theta, 4)``.
        n : int, default 1
            Number of samples per parameter vector.
        **kwargs
            Ignored.

        Returns
        -------
        torch.Tensor
            Samples. Leading and trailing singleton dimensions are squeezed:
            for ``(n_theta, 4)`` input the shape is ``(n_theta,)`` when
            ``n == 1`` and ``(n_theta, n)`` otherwise; for a single
            parameter vector it is ``(n,)``, or a scalar when ``n == 1``.
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
        """Simulate training pairs with parameters drawn from the prior.

        Parameters
        ----------
        n : int
            Number of pairs.
        **kwargs
            Ignored.

        Returns
        -------
        theta : torch.Tensor
            Prior draws of shape ``(n, 4)``.
        x_sim : torch.Tensor
            One sample per parameter vector, shape ``(n, 1)``.
        """
        theta = self.prior.sample((n,))
        x_sim = self.simulate(theta, n=1).unsqueeze(-1)
        return theta, x_sim

    def generate_contaminated_data(
        self,
        theta_true: torch.Tensor,
        n_obs: int,
        epsilon: float,
        outlier_values: Sequence[float] = (-50.0,),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate observed data at ``theta_true`` with additive outliers.

        ``n_obs`` i.i.d. draws are simulated at ``theta_true``; then
        ``floor(epsilon * n_obs)`` of them, chosen at random, are shifted by
        a value drawn uniformly from ``outlier_values``.

        Parameters
        ----------
        theta_true : torch.Tensor
            True parameter ``[A, log B, g, log k]``, shape ``(4,)`` or ``(1, 4)``.
        n_obs : int
            Number of observations.
        epsilon : float
            Fraction of observations to contaminate.
        outlier_values : sequence of float, default (-50.0,)
            Shifts to add to the contaminated observations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(x_obs, x_obs_mis, outlier_indices)``: the clean data and the
            contaminated data, both ``(n_obs, 1)``, and the indices of the
            contaminated rows.
        """
        theta_true = torch.as_tensor(theta_true).reshape(-1)
        x_obs = self.simulate(theta_true, n=n_obs).reshape(n_obs, 1)
        x_obs_mis, outlier_indices = add_outliers_by_proportion(
            x_obs, epsilon, outlier_values
        )
        return x_obs, x_obs_mis, outlier_indices

    def sample_gandk_fully_reparameterized(
        self,
        gamma: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        """Draw samples for one parameter vector via the quantile function.

        Parameters
        ----------
        gamma : torch.Tensor
            Parameters ``[A, log(B), g, log(k)]``, shape ``(4,)``.
        n : int, default 1
            Number of samples.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(n,)``, or a scalar when ``n == 1``.
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
        """Simulate observed data with prior-contaminated samples.

        Parameters
        ----------
        theta : torch.Tensor
            True parameters ``[A, log(B), g, log(k)]``, shape ``(4,)``.
        n_obs : int
            Number of observations.
        epsilon : float, default 0.0
            Fraction of observations, ``floor(epsilon * n_obs)``, that are
            replaced by samples simulated at parameters drawn from the
            prior.

        Returns
        -------
        torch.Tensor
            Observed data of shape ``(n_obs,)``.

        See Also
        --------
        nsm_bayes.simulators.add_noise.add_outliers_by_proportion :
            The additive-outlier contamination used by the g-and-k
            experiment.
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

