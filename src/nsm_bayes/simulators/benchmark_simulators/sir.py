"""Stochastic discrete-time SIR epidemic simulator.

Parameters are handled in the unconstrained space
``(log beta, log gamma, logit rho, log I0[, log phi])``: infection rate,
recovery rate, reporting probability, initial number infected and, for the
negative-binomial observation model, overdispersion.
"""

from __future__ import annotations

import torch
from torch.distributions import Binomial, MultivariateNormal, NegativeBinomial, Poisson

# Small constants that only guard against division by zero / clamp bounds.

class SIRSimulator:
    """Simulator for the stochastic SIR epidemic model.

    Simulates reported daily incidence in a closed population, transforms
    parameters between constrained and unconstrained spaces, computes
    summary statistics and generates (contaminated) data sets.

    Parameters
    ----------
    obs_model : {"poisson", "negbin"}, default "poisson"
        Observation model for reported cases. ``"negbin"`` takes effect only
        for 5-dimensional parameters (which include ``log phi``); with 4
        parameters observations are always Poisson.
    dt : float, default 1.0
        Length of one time step.
    prior_mean : torch.Tensor, optional
        Prior mean in the unconstrained space, shape ``(4,)``. Defaults to
        ``[0.5, 0.2, 0.5, 20.0]``.
    prior_cov : torch.Tensor, optional
        Prior covariance, shape ``(4, 4)``. Defaults to
        ``diag([0.5, 0.5, 1.0, 0.7] ** 2)``.
    T : int, default 150
        Number of time steps per trajectory.
    N : int, default 1000
        Population size.

    Attributes
    ----------
    prior : torch.distributions.MultivariateNormal
        Prior over the unconstrained parameters.

    Methods
    -------
    to_unconstrained
        Map constrained parameters to the unconstrained space.
    to_constrained
        Map unconstrained parameters to the constrained space.
    simulate
        Simulate incidence trajectories.
    summary
        Compute summary statistics of trajectories.
    generate_training_data
        Simulate trajectories at prior draws.
    generate_contaminated_data
        Simulate observed data with prior-contaminated trajectories.
    apply_undercounting_trajectory
        Undercount trajectories by binomial thinning.
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
        self.obs_model = obs_model
        self.dt = dt

        self.T = T
        self.N = N

        # Log-space prior mean and covariance
        if prior_mean is not None:
            self.prior_mean = prior_mean
        else:
            self.prior_mean = torch.tensor([0.5, 0.2, 0.5, 20.0], dtype=torch.float32)
        self.prior_mean = self.to_constrained(self.prior_mean)

        if prior_cov is not None:
            self.prior_cov = prior_cov
        else:
            std = torch.tensor([0.5, 0.5, 1.0, 0.7], dtype=torch.float32)
            self.prior_cov = torch.diag(std**2)
        
        self.prior = MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_cov)


    def to_unconstrained(self, vals: torch.Tensor) -> torch.Tensor:
        """Map constrained parameters to the unconstrained space.

        ``(beta, gamma, rho, I0[, phi]) -> (log beta, log gamma, logit rho,
        log I0[, log phi])``.

        Parameters
        ----------
        vals : torch.Tensor
            Constrained parameters of shape ``(n, d)`` with ``d`` 4 or 5.

        Returns
        -------
        torch.Tensor
            Unconstrained parameters of shape ``(n, d)``.
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
        """Map unconstrained parameters to the constrained space.

        ``(log beta, log gamma, logit rho, log I0[, log phi]) -> (beta,
        gamma, rho, I0[, phi])``. ``I0`` is rounded to an integer and
        clamped to at least 1, so this is not an exact inverse of
        :meth:`to_unconstrained`.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters of shape ``(n, d)`` with ``d`` 4 or 5.

        Returns
        -------
        torch.Tensor
            Constrained parameters of shape ``(n, d)``.
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
        """Simulate reported incidence with binomial transitions.

        At each step, new infections are ``Binomial(S, 1 - exp(-beta I / N
        dt))`` and recoveries ``Binomial(I, 1 - exp(-gamma dt))``. Reported
        cases are ``Poisson(rho * new_infections)`` or, for 5-dimensional
        parameters with ``obs_model="negbin"``, negative binomial with mean
        ``rho * new_infections`` and dispersion ``phi``.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters of shape ``(n, d)``:
            ``(log beta, log gamma, logit rho, log I0)`` for ``d == 4``, plus
            ``log phi`` for ``d == 5``.
        **kwargs
            Ignored.

        Returns
        -------
        torch.Tensor
            Reported incidence of shape ``(n, T)``, with the dtype and device
            of `theta`.

        Raises
        ------
        AssertionError
            If `theta` is not 2-D.
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
        """Summarise incidence trajectories.

        Parameters
        ----------
        y : torch.Tensor
            Incidence of shape ``(n, T)``.
        N : float, optional
            Unused; the simulator's population size ``self.N`` is always
            used.

        Returns
        -------
        torch.Tensor
            ``float32`` summaries of shape ``(n, 3)``:

            * attack rate, ``sum_t y_t / N``;
            * time of the peak, ``argmax_t y_t / (T - 1)``;
            * peak height, ``max_t y_t / N``.

        Raises
        ------
        ValueError
            If `y` has 3 columns, i.e. appears to be summarised already.
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
        """Simulate trajectories at parameters drawn from the prior.

        Parameters
        ----------
        n : int
            Number of trajectories.

        Returns
        -------
        theta : torch.Tensor
            Unconstrained prior draws of shape ``(n, d)``.
        y : torch.Tensor
            Raw incidence of shape ``(n, T)``; apply :meth:`summary` to get
            the training features.
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
        """Simulate observed data in which some trajectories use prior draws.

        Each of the `n_obs` trajectories is independently contaminated with
        probability `epsilon`; contaminated trajectories are simulated at a
        parameter drawn from the prior, the rest at `theta_true`.

        Parameters
        ----------
        theta_true : torch.Tensor
            True unconstrained parameters, shape ``(d,)``.
        n_obs : int
            Number of trajectories.
        epsilon : float
            Contamination probability per trajectory.
        contaminant : str, default "prior"
            ``"prior"`` draws contaminating parameters from the prior; any
            other value uses the fixed offset ``theta_true + 1``.

        Returns
        -------
        y : torch.Tensor
            Incidence of shape ``(n_obs, T)``.
        is_contam : torch.Tensor
            Boolean mask of shape ``(n_obs,)``.
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
        """Undercount clean trajectories by binomial thinning.

        Each selected count ``y`` is replaced by a draw from
        ``Binomial(y, q)``, so counts can only decrease.

        Parameters
        ----------
        y_obs : torch.Tensor
            Clean incidence of shape ``(n_obs, T)``; not modified.
        epsilon : float
            Probability that a trajectory (or a time point, see
            `per_time`) is thinned.
        q : float, default 0.3
            Retention probability of each reported case.
        per_time : bool, default False
            If false, whole trajectories are selected; if true, individual
            time points are.

        Returns
        -------
        y_corrupted : torch.Tensor
            Thinned copy of `y_obs` with the same dtype.
        is_contam : torch.Tensor
            Boolean mask of shape ``(n_obs,)`` or ``(n_obs, T)``.
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
        