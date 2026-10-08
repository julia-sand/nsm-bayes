"""
This module contains the SIR simulator class for generating epidemiological data.
"""

from __future__ import annotations

import torch
from torch.distributions import Binomial, MultivariateNormal, NegativeBinomial, Poisson


class SIRSimulator:
    """Simulator for the SIR epidemiological model.

    This class provides methods to simulate the stochastic SIR model, transform
    parameters between constrained and unconstrained spaces, and generate
    synthetic datasets for training and observation.
    """

    def __init__(self, obs_model: str = "poisson", dt: float = 1.0):
        """Initialize the SIR simulator.

        Parameters
        ----------
        obs_model : str, default "poisson"
            Observation model: "poisson" or "negbin".
        dt : float, default 1.0
            Time step for the simulation.
        """
        self.obs_model = obs_model
        self.dt = dt

    def to_unconstrained(self, vals: torch.Tensor) -> torch.Tensor:
        """Transform constrained parameters to unconstrained space.

        Transforms (beta, gamma, rho, I0, [phi]) -> (log_beta, log_gamma, logit_rho, log_I0, [log_phi]).

        Parameters
        ----------
        vals : torch.Tensor
            Constrained parameters of shape (n, d).

        Returns
        -------
        torch.Tensor
            Unconstrained parameters of shape (n, d).
        """
        n, d = vals.shape
        unconstrained = torch.empty_like(vals)
        unconstrained[:, 0] = torch.log(vals[:, 0])  # beta
        unconstrained[:, 1] = torch.log(vals[:, 1])  # gamma
        unconstrained[:, 2] = torch.logit(vals[:, 2])  # rho
        unconstrained[:, 3] = torch.log(vals[:, 3])  # I0
        if d == 5:
            unconstrained[:, 4] = torch.log(vals[:, 4])  # phi
        return unconstrained

    def to_constrained(self, theta: torch.Tensor) -> torch.Tensor:
        """Transform unconstrained parameters to constrained space.

        Transforms (log_beta, log_gamma, logit_rho, log_I0, [log_phi]) -> (beta, gamma, rho, I0, [phi]).

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters of shape (n, d).

        Returns
        -------
        torch.Tensor
            Constrained parameters of shape (n, d).
        """
        n, d = theta.shape
        constrained = torch.empty_like(theta)
        constrained[:, 0] = torch.exp(theta[:, 0])  # beta
        constrained[:, 1] = torch.exp(theta[:, 1])  # gamma
        constrained[:, 2] = torch.sigmoid(theta[:, 2])  # rho
        constrained[:, 3] = torch.clamp(torch.round(torch.exp(theta[:, 3])), min=1.0)  # I0
        if d == 5:
            constrained[:, 4] = torch.exp(theta[:, 4])  # phi
        return constrained

    def simulate_sir(
        self,
        theta: torch.Tensor,
        T: int = 100,
        N: int = 10_000,
    ) -> torch.Tensor:
        """Stochastic discrete-time SIR with Binomial transitions.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter tensor of shape (n, d_theta) in unconstrained space.
            If d_theta == 4: (log_beta, log_gamma, logit_rho, log_I0).
            If d_theta == 5: add log_phi for NegBin overdispersion.
        T : int, default 100
            Time horizon.
        N : int, default 10_000
            Population size.

        Returns
        -------
        torch.Tensor
            y: (n, T) observed new cases per day.
        """
        assert theta.dim() == 2
        n, d = theta.shape
        device = theta.device
        dtype = theta.dtype

        params = self.to_constrained(theta)
        beta = params[:, 0]
        gamma = params[:, 1]
        rho = params[:, 2]
        I0 = params[:, 3]

        if d == 5:
            phi = params[:, 4]

        # State arrays
        S = torch.empty(n, T + 1, device=device, dtype=dtype)
        I = torch.empty(n, T + 1, device=device, dtype=dtype)
        R = torch.empty(n, T + 1, device=device, dtype=dtype)

        S[:, 0] = float(N) - I0
        I[:, 0] = I0
        R[:, 0] = 0.0

        y = torch.empty(n, T, device=device, dtype=dtype)

        for t in range(T):
            St = S[:, t].clamp_min(0.0)
            It = I[:, t].clamp_min(0.0)

            # Infection probability over dt (mass-action)
            p_inf = 1.0 - torch.exp(-beta * (It / float(N)) * self.dt)
            p_inf = p_inf.clamp(0.0, 1.0)

            # Recovery probability over dt
            p_rec = 1.0 - torch.exp(-gamma * self.dt)
            p_rec = p_rec.clamp(0.0, 1.0)

            new_inf = Binomial(total_count=St, probs=p_inf).sample()
            new_rec = Binomial(total_count=It, probs=p_rec).sample()

            S[:, t + 1] = St - new_inf
            I[:, t + 1] = It + new_inf - new_rec
            R[:, t + 1] = R[:, t] + new_rec

            # Observations: reported incident infections
            mu = (rho * new_inf).clamp_min(0.0)

            if self.obs_model == "poisson" or d == 4:
                y[:, t] = Poisson(mu).sample()
            else:
                # One common NegBin parameterization: mean mu, variance mu + mu^2/phi
                probs = (phi / (phi + mu + 1e-8)).clamp(1e-6, 1 - 1e-6)
                y[:, t] = NegativeBinomial(total_count=phi, probs=probs).sample()

        return y

    def sir_summary(self, y: torch.Tensor, N: float) -> torch.Tensor:
        """Compute summary statistics for SIR incidence.

        Parameters
        ----------
        y : torch.Tensor
            Incidence data of shape (n, T).
        N : float
            Population size.

        Returns
        -------
        torch.Tensor
            Summary statistics of shape (n, 3): [attack_rate, t_peak_scaled, peak_scaled].
        """
        y = y.to(torch.float32)
        n, T = y.shape

        attack = y.sum(dim=1) / float(N)
        peak = y.max(dim=1).values / float(N)
        t_peak = y.argmax(dim=1).to(torch.float32) / float(max(T - 1, 1))

        return torch.stack([attack, t_peak, peak], dim=1)

    def sir_summary(self, y: torch.Tensor, N: float) -> torch.Tensor:
        """Compute summary statistics for SIR incidence.

        Parameters
        ----------
        y : torch.Tensor
            Incidence data of shape (n, T).
        N : float
            Population size.

        Returns
        -------
        torch.Tensor
            Summary statistics of shape (n, 3): [attack_rate, t_peak_scaled, peak_scaled].
        """
        y = y.to(torch.float32)
        n, T = y.shape

        attack = y.sum(dim=1) / float(N)
        peak = y.max(dim=1).values / float(N)
        t_peak = y.argmax(dim=1).to(torch.float32) / float(max(T - 1, 1))

        return torch.stack([attack, t_peak, peak], dim=1)


    def sir_summary(self, y: torch.Tensor, N: float) -> torch.Tensor:
        """Compute summary statistics for SIR incidence.

        Parameters
        ----------
        y : torch.Tensor
            Incidence data of shape (n, T).
        N : float
            Population size.

        Returns
        -------
        torch.Tensor
            Summary statistics of shape (n, 3): [attack_rate, t_peak_scaled, peak_scaled].
        """
        y = y.to(torch.float32)
        n, T = y.shape

        attack = y.sum(dim=1) / float(N)
        peak = y.max(dim=1).values / float(N)
        t_peak = y.argmax(dim=1).to(torch.float32) / float(max(T - 1, 1))

        return torch.stack([attack, t_peak, peak], dim=1)

    def generate_training_data(
        self, prior: MultivariateNormal, n: int, T: int, N: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate training data by sampling from the prior.

        Parameters
        ----------
        prior : MultivariateNormal
            Prior distribution for unconstrained parameters.
        n : int
            Number of samples.
        T : int
            Time horizon.
        N : int
            Population size.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (observed_data, summary_stats), where observed_data is (n, T)
            and summary_stats is (n, 3).
        """
        theta = prior.sample((n,))
        y = self.simulate_sir(theta, T=T, N=N)
        summary = self.sir_summary(y, N)
        return y, summary

    def generate_observed_data(
        self,
        theta_true: torch.Tensor,
        n_obs: int,
        epsilon: float,
        prior: MultivariateNormal,
        T: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate observed data with a mixture of clean and contaminated samples.

        Parameters
        ----------
        theta_true : torch.Tensor
            True unconstrained parameters of shape (1, d).
        n_obs : int
            Total number of observed samples.
        epsilon : float
            Fraction of contaminated samples.
        prior : MultivariateNormal
            Prior distribution for contaminated parameters.
        T : int
            Time horizon.
        N : int
            Population size.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (observed_data, summary_stats).
        """
        n_clean = int((1 - epsilon) * n_obs)
        n_contam = n_obs - n_clean

        # Clean data
        theta_clean = theta_true.repeat(n_clean, 1)
        y_clean = self.simulate_sir(theta_clean, T=T, N=N)

        # Contaminated data
        theta_contam = prior.sample((n_contam,))
        y_contam = self.simulate_sir(theta_contam, T=T, N=N)

        y = torch.cat([y_clean, y_contam], dim=0)
        summary = self.sir_summary(y, N)

        return y, summary



def sir_summary(y: torch.Tensor, N: float) -> torch.Tensor:
    """Compute summary statistics for SIR incidence.

    Parameters
    ----------
    y : torch.Tensor
        Incidence data of shape (n, T).
    N : float
        Population size.

    Returns
    -------
    torch.Tensor
        Summary statistics of shape (n, 3): [attack_rate, t_peak_scaled, peak_scaled].
    """
    y = y.to(torch.float32)
    n, T = y.shape

    attack = y.sum(dim=1) / float(N)
    peak = y.max(dim=1).values / float(N)
    t_peak = y.argmax(dim=1).to(torch.float32) / float(max(T - 1, 1))

    return torch.stack([attack, t_peak, peak], dim=1)


def make_sir_prior(cfg: dict) -> MultivariateNormal:
    """Create a MultivariateNormal prior for SIR parameters.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing 'mean' and 'cov' for the prior.

    Returns
    -------
    MultivariateNormal
        The prior distribution.
    """
    mean = torch.tensor(cfg["mean"], dtype=torch.float32)
    cov = torch.tensor(cfg["cov"], dtype=torch.float32)
    return MultivariateNormal(mean, cov)
