"""
This module contains the SIR simulator class for generating epidemiological data.
"""

from __future__ import annotations

import torch
from torch.distributions import Binomial, MultivariateNormal, NegativeBinomial, Poisson

# Small constants that only guard against division by zero / clamp bounds.

class SIRSimulator:
    """Simulator for the SIR epidemiological model.

    This class provides methods to simulate the stochastic SIR model, transform
    parameters between constrained and unconstrained spaces, and generate
    synthetic datasets for training and observation.

    Methods
    -------
    to_unconstrained
        Transform constrained parameters to unconstrained space.
    to_constrained
        Transform unconstrained parameters to constrained space.
    simulate
        Simulate a stochastic discrete-time SIR process.
    summary
        Compute summary statistics for SIR incidence.
    generate_training_data
        Generate training data by sampling from the prior.
    generate_observed_data
        Generate observed data with a mixture of clean and contaminated
        samples.
    """

    def __init__(
        self,
        obs_model: str = "poisson",
        dt: float = 1.0,
        prior_mean: torch.Tensor | None = None,
        prior_cov: torch.Tensor | None = None,
        T=150,
        N=1000,
    ) -> None:
        """Initialize the SIR simulator.

        Parameters
        ----------
        obs_model : str, default "poisson"
            Observation model: "poisson" or "negbin".
        dt : float, default 1.0
            Time step for the simulation.
        prior_mean : torch.Tensor, optional
            Prior mean vector. If None, uses default values.
        prior_cov : torch.Tensor, optional
            Prior covariance matrix. If None, uses default values.
        """
        self.obs_model = obs_model
        self.dt = dt

        self.T = T
        self.N = N

        # Log-space prior mean and covariance
        if prior_mean is not None:
            self.prior_mean = prior_mean
        else:
            self.prior_mean = torch.tensor([0.5, 0.2, 0.5, 20.0], dtype=torch.float32)
        #self.prior_mean = self.to_constrained(self.prior_mean)

        if prior_cov is not None:
            self.prior_cov = prior_cov
        else:
            std = torch.tensor([0.5, 0.5, 1.0, 0.7], dtype=torch.float32)
            self.prior_cov = torch.diag(std**2)
        
        self.prior = MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_cov)


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

    def simulate(
        self,
        theta: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Stochastic discrete-time SIR with Binomial transitions.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter tensor of shape (n, d_theta) in unconstrained space.
            If d_theta == 4: (log_beta, log_gamma, logit_rho, log_I0).
            If d_theta == 5: add log_phi for NegBin overdispersion.
        **kwargs : dict
            Additional arguments (ignored).

        Returns
        -------
        torch.Tensor
            Simulated trajectories of shape (n, T).
        """
        assert theta.dim() == 2
        n, d = theta.shape
        device = theta.device
        dtype = theta.dtype

        #move to unconstrained space
        params = self.to_constrained(theta)
        beta = params[:, 0]
        gamma = params[:, 1]
        rho = params[:, 2]
        I0 = params[:, 3]

        if d == 5:
            phi = params[:, 4]

        # State arrays
        S = torch.empty(n, self.T + 1, device=device, dtype=dtype)
        I = torch.empty(n, self.T + 1, device=device, dtype=dtype)
        R = torch.empty(n, self.T + 1, device=device, dtype=dtype)

        S[:, 0] = float(self.N) - I0
        I[:, 0] = I0
        R[:, 0] = 0.0

        y = torch.empty(n, self.T, device=device, dtype=dtype)

        for t in range(self.T):
            St = S[:, t].clamp_min(0.0)
            It = I[:, t].clamp_min(0.0)

            # Infection probability over dt (mass-action)
            p_inf = 1.0 - torch.exp(-beta * (It / float(self.N)) * self.dt)
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
                #probs = (phi / (phi + mu)).clamp(0, 1 - EPS6)
                probs = (mu / (phi + mu)).clamp(max=1 - 1e-6)
                y[:, t] = NegativeBinomial(total_count=phi, probs=probs).sample()

        return y

    def summary(self, y: torch.Tensor, N: float | None = None) -> torch.Tensor:
        """Compute summary statistics for SIR incidence.

        Parameters
        ----------
        y : torch.Tensor
            Incidence data of shape (n, T).

        Returns
        -------
        torch.Tensor
            Summary statistics of shape (n, 3): [attack_rate, t_peak_scaled, peak_scaled].

        Raises
        ------
        ValueError
            If `y` is already summarized (i.e., has 3 columns).
        """
        y = y.to(torch.float32)

        if y.shape[1] == 3:
            raise ValueError(
                "Input data `y` appears to be already summarized (shape (n, 3)). "
                "Summary statistics should be computed from raw incidence data (n, T)."
            )

        attack = y.sum(dim=1) / float(self.N)
        peak = y.max(dim=1).values / float(self.N)
        t_peak = y.argmax(dim=1).to(torch.float32) / float(max(self.T - 1, 1))


        return torch.stack([attack, t_peak, peak], dim=1)

    def generate_training_data(self,n ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate training data by sampling from the prior.

        Parameters
        ----------
        prior : MultivariateNormal
            Prior distribution for unconstrained parameters.
        n : int
            Number of samples.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (theta, observed_data, summary_stats), where theta is
            (n, d), observed_data is (n, T) and summary_stats is (n, 3).
        """
        theta = self.prior.sample((n,))
        y = self.simulate(theta)
        #summary = self.summary(y)
        
        return theta, y #, summary

    def generate_contaminated_data(
        self,
        theta_true: torch.Tensor,
        n_obs: int,
        epsilon: float,
        contaminant="prior",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate observed data with theta true, and 
        add a fraction epsilon of different thetas

        Parameters
        ----------
        theta_true : torch.Tensor
            True unconstrained parameters of shape (1, d).
        n_obs : int
            Total number of observed samples.
        epsilon : float
            Fraction of contaminated samples.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (observations, contamination index, thetas used for simulations).
        """

        device = theta_true.device


        # Determine which samples are contaminated
        # randomly select a proportion of epsilon times 100 percent 
        is_contam = torch.rand(n_obs, device=device) < epsilon
        n_contam = is_contam.sum().item()

        #store thetas used for simulations
        theta_used = torch.zeros((n_obs, theta_true.shape[0]), device=device, dtype=theta_true.dtype)

        # Clean samples use theta_true
        theta_used[~is_contam] = theta_true

        # Contaminated samples
        if contaminant == "prior":
            theta_contam = self.prior.sample((n_contam,))
            theta_used[is_contam] = theta_contam
        else:
            # Fallback: just use theta_true for now or a fixed offset
            print("No contamination, only fixed offset.")
            theta_used[is_contam] = theta_true+1

        y = self.simulate(theta_used)

        return y, is_contam


    def apply_undercounting_trajectory(self,
        y_obs, #sampled clean data
        epsilon: float,   # fraction of contaminated trajectories
        q: float = 0.3,            # retention probability (one-sided: downward)
        per_time: bool = False,    # if True: contaminate per time point, else per trajectory
    ):
        """
        One-sided undercounting by binomial thinning.
        applies the thinning to clean data, given by yobs.
        Returns: y_corrupted, is_contam_mask
        - if per_time=False: is_contam_mask is (n_obs,) for contaminated trajectories
        - if per_time=True:  is_contam_mask is (n_obs, T) for contaminated time points
        
        theta_true: torch.Tensor,

        """
      
        y_cor = y_obs.clone()

        if not per_time:
            is_contam = (torch.rand(y_obs.shape[0], device=y_obs.device) < epsilon)  # (n_obs,)
            if is_contam.any():
                idx = is_contam.nonzero(as_tuple=True)[0]
                y_sub = y_cor[idx].float()
                y_cor[idx] = Binomial(total_count=y_sub, probs=torch.tensor(q, device=y_obs.device)).sample().to(y_obs.dtype)
            return y_cor, is_contam
        else:
            is_contam = (torch.rand(y_obs.shape[0], y_obs.shape[1], device=y_obs.device) < epsilon)  # (n_obs,T)
            if is_contam.any():
                y_sub = y_cor[is_contam].float()
                y_cor[is_contam] = Binomial(total_count=y_sub, probs=torch.tensor(q, device=y_obs.device)).sample().to(y_obs.dtype)
            return y_cor, is_contam
        