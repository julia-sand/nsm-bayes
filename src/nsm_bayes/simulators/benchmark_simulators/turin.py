"""
This module contains the Turin channel model simulator class.
"""

from __future__ import annotations

import torch
from torch.distributions import MultivariateNormal

# Small constant that only guards log(0).
EPS = 1e-12
class TurinSimulator:
    """Simulator for the Turin channel model.

    This class provides methods to simulate the stochastic Turin channel
    model, compute the temporal moment summaries of the received time-domain
    response, and generate synthetic training and observation datasets.
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
        """Initialize the Turin channel model simulator.

        Parameters
        ----------
        B : float, default 4e9
            Bandwidth in Hz.
        Ns : int, default 801
            Number of frequency samples.
        N : int, default 50
            Number of time series (receivers) drawn per call to
            :meth:`simulate` when no explicit `N` is given.
        tau0 : float, default 0.0
            Minimum delay threshold in seconds.
        device : str, default "cpu"
            Device to use ("cpu" or "cuda").
        prior_mean : torch.Tensor, optional
            Prior mean vector. If None, uses default values.
        prior_cov : torch.Tensor, optional
            Prior covariance matrix. If None, uses default values.
        """
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

    def prior(self, num_samples: int | None = None) -> torch.Tensor | MultivariateNormal:
        """Return the Gaussian prior used in the Turin experiments.

        This is the prior defined in ``configs/turin.yaml`` (``prior_mean``
        and ``prior_cov``), in the log-space parameter vector
        ``[log(G0), log(T), log(lambda_0), log(sigma2_N)]``.

        Parameters
        ----------
        num_samples : int or None, default None
            If provided, returns `num_samples` samples from the prior.
            If None, returns the MultivariateNormal distribution object.

        Returns
        -------
        torch.Tensor or MultivariateNormal
            If `num_samples` is provided, returns a tensor of shape
            (num_samples, 4). Otherwise, returns the prior distribution.
        """
        dist = MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_cov)
        if num_samples is not None:
            return dist.sample((num_samples,))
        return dist

    def _simulate_channel(
        self, theta: torch.Tensor, epsilon: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate the complex channel frequency response with AWGN.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector [G0, T, lambda_0, sigma2_N] in log space.
        epsilon : float, default 0.0
            Fraction (0-1) of outlier receivers to replace with pure noise.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (Y, outlier_indices), where Y is the received frequency response
            of shape (self.N, Ns) and outlier_indices indexes the receivers
            that received noise only.
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

        # choose outlier receivers
        outlier_indices = torch.empty(0, dtype=torch.long, device=self.device)
        if epsilon > 0.0:
            n_outliers = int(epsilon * nRx)
            if n_outliers > 0:
                outlier_indices = torch.randperm(nRx, device=self.device)[:n_outliers]
                H[outlier_indices, :] = 0.0  # kill the channel, leave only noise

        # received freq response
        return H + Noise, outlier_indices

    def temporal_moments(
        self, Y: torch.Tensor, K: int = 3
    ) -> torch.Tensor:
        """Compute log temporal moment summaries of a received response.

        Parameters
        ----------
        Y : torch.Tensor
            Received frequency response of shape (N, Ns), complex.
        K : int, default 3
            Number of temporal moments to compute.

        Returns
        -------
        torch.Tensor
            Log temporal moments of shape (N, K), float64.
        """
        delta_f = self.B / (self.Ns - 1)
        t_max = 1 / delta_f
        tau = torch.linspace(0, t_max, self.Ns, device=Y.device)
        n = Y.shape[0]
        out = torch.zeros((n, K), dtype=torch.float64, device=Y.device)

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate the Turin channel model and summarize the response.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector [G0, T, lambda_0, sigma2_N] in log space,
            either 1D (broadcast to all receivers) or 2D of shape
            (n_theta, d_theta) to give each receiver its own parameters.
        T : int, optional
            Time horizon. For the Turin simulator, this is ignored as the
            number of samples is governed by `Ns`.
        **kwargs
            Additional simulation arguments. Supported:
            - N : int, optional. Number of receivers. If None, uses `self.N`
              or `theta.shape[0]` if theta is 2D.
            - Ns : int, optional. Number of frequency samples. Defaults to `self.Ns`.
            - output : str, default "moments". "moments" to return 3D log
              moment summaries, "data" for the full time-domain power (dB).
            - epsilon : float, default 0.0. Fraction (0-1) of outlier time
              series to replace with pure noise.

        Returns
        -------
        torch.Tensor or tuple
            If output == "moments", returns temporal moments of shape
            (N, K). If output == "data", returns (p_dB, outlier_indices, tau).
        """
        N = kwargs.get("N")
        Ns = kwargs.get("Ns")
        output = kwargs.get("output", "moments")
        epsilon = kwargs.get("epsilon", 0.0)

        nRx = (
            N
            if N is not None
            else (theta.shape[0] if theta.dim() == 2 else self.N)
        )
        if Ns is not None and Ns != self.Ns:
            raise ValueError(
                f"Ns must match the simulator's Ns ({self.Ns}), got {Ns}"
            )

        Y, outlier_indices = self._simulate_channel(theta, epsilon=epsilon)

        if output == "moments":
            # summary over the first nRx receivers (any extra receivers
            # only draw random numbers and are discarded)
            Y = Y[:nRx, :]
            return self.temporal_moments(Y)
        elif output == "data":
            y_td = torch.fft.ifft(Y[:nRx, :], dim=1)
            p = torch.abs(y_td) ** 2
            p_dB = 10 * torch.log10(p + EPS)
            delta_f = self.B / (self.Ns - 1)
            tau = torch.linspace(0, 1 / delta_f, self.Ns, device=self.device)
            return (
                p_dB.detach().cpu(),
                outlier_indices.detach().cpu(),
                tau.detach().cpu(),
            )
        raise ValueError(f"Unknown output {output!r}; expected 'moments' or 'data'")

    def generate_training_data(
        self,
        prior: MultivariateNormal,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate training data by sampling from the prior.

        Parameters
        ----------
        prior : MultivariateNormal
            Prior distribution for the log-space parameters.
        n : int
            Number of training samples; the simulator draws one batch of
            `n` receivers, each with its own theta.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (theta, x_sim), where theta is (n, d_theta) and
            x_sim is (n, K) log temporal moments.
        """
        theta = prior.sample((n,))
        # One batch draw: `n` receivers, each with its own theta, in a
        # single call to :meth:`simulate`.
        x_sim = self.simulate(theta, N=n)
        return theta, x_sim

