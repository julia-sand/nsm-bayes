"""Turin model of a radio propagation channel.

Multipath components arrive as a homogeneous Poisson process with rate
``lambda_0`` over the observation window. Their complex gains are Gaussian
with variance decaying exponentially in the delay (reverberation time
``T``, gain ``G0``), and complex white Gaussian noise of variance
``sigma2_N`` is added to the frequency response. Parameters are handled in
log space: ``theta = log([G0, T, lambda_0, sigma2_N])``.
"""

import torch
from torch.distributions import MultivariateNormal

# Small constant that only guards log(0).
EPS = 1e-12
class TurinSimulator:
    """Simulator for the Turin channel model.

    Simulates received frequency responses, summarises them by log temporal
    moments, and generates training and observed data sets.

    Parameters
    ----------
    B : float, default 4e9
        Bandwidth in Hz.
    Ns : int, default 801
        Number of frequency samples.
    N : int, default 50
        Default number of receivers simulated for a 1-D `theta`.
    tau0 : float, default 0.0
        Multipath components with delay below `tau0` (seconds) get zero
        gain.
    device : str, default "cpu"
        Device for the simulation (``"cpu"`` or ``"cuda"``).
    prior_mean : torch.Tensor, optional
        Prior mean in log space, shape ``(4,)``. Defaults to
        ``[-19, -19, 22, -22]``.
    prior_cov : torch.Tensor, optional
        Prior covariance, shape ``(4, 4)``. Defaults to the identity.

    Attributes
    ----------
    prior : torch.distributions.MultivariateNormal
        Prior over the log parameters.
    """

    def __init__(
        self,
        B: float = 4e9,
        Ns: int = 801,
        N: int = 50,
        tau0: float = 0.0,
        device: str = "cpu",
        prior_mean: torch.Tensor | None = None,
        prior_cov: torch.Tensor | None = None,
    ) -> None:
        self.B = B
        self.Ns = Ns
        self.N = N
        self.tau0 = tau0
        self.device = device

        # Log-space prior mean and covariance
        if prior_mean is not None:
            self.prior_mean = prior_mean
        else:
            self.prior_mean = torch.tensor([-19.0, -19.0, 22.0, -22.0], dtype=torch.float32, device=self.device)

        if prior_cov is not None:
            self.prior_cov = prior_cov
        else:
            self.prior_cov = torch.eye(4, dtype=torch.float32, device=self.device)

        self.prior = MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_cov)


    def _simulate_channel(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate the received complex frequency response.

        Parameters
        ----------
        theta : torch.Tensor
            Log parameters ``[G0, T, lambda_0, sigma2_N]``: shape ``(4,)``
            (shared by ``self.N`` receivers) or ``(nRx, 4)`` (one row per
            receiver).

        Returns
        -------
        Y : torch.Tensor
            Complex frequency response of shape ``(nRx, Ns)``.
        """
        # batch support: theta (d_theta,) broadcasts to all nRx receivers,
        # theta (nRx, d_theta) gives each receiver its own parameters
        theta = torch.as_tensor(theta)
        if theta.dim() == 2:
            nRx = theta.shape[0]
            theta_rows = theta
        else:
            nRx = self.N
            theta_rows = theta.unsqueeze(0).expand(nRx, -1)
        exp_theta = torch.exp(theta_rows.to(self.device))
        G0, T, lambda_0, sigma2_N = exp_theta.unbind(dim=1)

        delta_f = self.B / (self.Ns - 1)
        t_max = 1 / delta_f

        # channel matrix H
        H = torch.zeros((nRx, self.Ns), dtype=torch.cfloat, device=self.device)
        mu_poisson = lambda_0 * t_max

        for jR in range(nRx):
            n_points = int(torch.poisson(mu_poisson[jR]))
            delays = torch.sort(torch.rand(n_points, device=self.device) * t_max)[0]

            alpha = torch.zeros(n_points, dtype=torch.cfloat, device=self.device)
            sigma2 = G0[jR] * torch.exp(-delays / T[jR]) / lambda_0[jR] * self.B

            for l in range(n_points):
                if delays[l] < self.tau0:
                    alpha[l] = 0
                else:
                    std = torch.sqrt(sigma2[l] / 2)
                    alpha[l] = torch.normal(0, std) + 1j * torch.normal(0, std)

            # H[jR,f]
            H[jR, :] = torch.matmul(
                torch.exp(
                    -1j * 2 * torch.pi * delta_f
                    * torch.outer(
                        torch.arange(self.Ns, device=self.device), delays
                    )
                ),
                alpha,
            )

        # AWGN
        std_noise = torch.sqrt(sigma2_N / 2)
        Noise = (
            torch.normal(0.0, std_noise.unsqueeze(-1).expand(-1, self.Ns))
            + 1j
            * torch.normal(0.0, std_noise.unsqueeze(-1).expand(-1, self.Ns))
        ).to(self.device)

        # received freq response
        return H + Noise

    def temporal_moments(
        self, Y: torch.Tensor, K: int = 3
    ) -> torch.Tensor:
        """Log temporal moments of the received power delay profile.

        For each receiver the time-domain response ``y = ifft(Y)`` is formed
        and the moments ``m_k = integral tau^k |y(tau)|^2 dtau`` for
        ``k = 0, ..., K - 1`` are computed with the trapezoidal rule.

        Parameters
        ----------
        Y : torch.Tensor
            Complex frequency response of shape ``(n, Ns)``.
        K : int, default 3
            Number of moments.

        Returns
        -------
        torch.Tensor
            ``log(m_k)`` as ``float32``, shape ``(n, K)``.
        """
        delta_f = self.B / (self.Ns - 1)
        t_max = 1 / delta_f
        tau = torch.linspace(0, t_max, self.Ns, device=Y.device)
        n = Y.shape[0]
        out = torch.zeros((n, K), dtype=torch.float32, device=Y.device)

        for k in range(K):
            for i in range(n):
                y = torch.fft.ifft(Y[i, :])
                out[i, k] = torch.trapz(tau**k * (torch.abs(y) ** 2), tau)

        return torch.log(out)

    def simulate(
        self,
        theta: torch.Tensor,
        T: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Simulate receivers and return the log temporal moments.

        Parameters
        ----------
        theta : torch.Tensor
            Log parameters ``[G0, T, lambda_0, sigma2_N]``: shape ``(4,)``
            (shared by all receivers) or ``(n_theta, 4)`` (one row per
            receiver).
        T : int, optional
            Ignored; accepted for interface compatibility.
        **kwargs
            Optional keyword arguments:

            ``N`` : int
                Number of receivers to return. Defaults to
                ``theta.shape[0]`` for 2-D `theta`, else ``self.N``. Extra
                simulated receivers are discarded.

        Returns
        -------
        torch.Tensor
            Log temporal moments of shape ``(N, 3)``.
        """
        N = kwargs.get("N")

        nRx = (
            N
            if N is not None
            else (theta.shape[0] if theta.dim() == 2 else self.N)
        )

        Y = self._simulate_channel(theta)
        # summary over the first nRx receivers (any extra receivers
        # only draw random numbers and are discarded)
        return self.temporal_moments(Y[:nRx, :])

    def generate_training_data(
        self,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate training pairs with parameters drawn from the prior.

        Parameters
        ----------
        n : int
            Number of pairs; one receiver is simulated per prior draw, in a
            single batched call.

        Returns
        -------
        theta : torch.Tensor
            Log-parameter draws of shape ``(n, 4)``.
        x_sim : torch.Tensor
            Log temporal moments of shape ``(n, 3)``.
        """
        theta = self.prior.sample((n,))
        # One batch draw: `n` receivers, each with its own theta, in a
        # single call to :meth:`simulate`.
        x_sim = self.simulate(theta, N=n)
        return theta, x_sim

    def generate_observed_data(
        self,
        theta: torch.Tensor,
        n_obs: int,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Simulate observed data with prior-contaminated receivers.

        Parameters
        ----------
        theta : torch.Tensor
            True log parameters, shape ``(4,)``.
        n_obs : int
            Number of receivers.
        epsilon : float, default 0.0
            Fraction of receivers, ``floor(epsilon * n_obs)``, replaced by
            receivers simulated at parameters drawn from the prior.

        Returns
        -------
        torch.Tensor
            Log temporal moments of shape ``(n_obs, 3)``.
        """
        # Draw theta for each receiver (identical to theta_true)
        theta_batch = theta.expand(n_obs, -1)
        # Simulate
        x_obs = self.simulate(theta_batch, N=n_obs)
        
        # Introduce outliers: replace epsilon * n_obs receivers with noise
        n_out = int(torch.floor(torch.tensor(epsilon * n_obs)).item())
        outlier_indices = torch.randperm(n_obs)[:n_out]
        
        # Create noise (using a random theta or just random normal if appropriate)
        # For consistency with this model, we can sample a random theta from the prior
        if n_out > 0:
            theta_out = self.prior.sample((n_out,))
            x_noise = self.simulate(theta_out, N=n_out)
            x_obs[outlier_indices] = x_noise
            
        return x_obs

