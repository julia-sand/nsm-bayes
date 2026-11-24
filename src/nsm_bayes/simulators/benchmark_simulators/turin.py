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
        
        self.delta_f = self.B / (self.Ns - 1)
        self.t_max = 1 / self.delta_f
        self.tau = torch.linspace(0, self.t_max, self.Ns)

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

    def generate_training_data(
        self,
        n_samples: int,
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
        # One batch draw: `n` receivers, each with its own theta, in a
        # single call to :meth:`simulate`.
        theta = self.prior.sample((n_samples,))
        H = self.simulate_channel(theta,n_samples)
        noise = self.sample_noise(torch.exp(theta[:, 3]), n_samples)
        x_sim = self.summary(H + noise)
        return theta, x_sim

    def generate_contaminated_data(
        self,
        theta_true: torch.Tensor,
        n_obs: int,
        epsilon: float,
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
        # One batch draw: `n` receivers, each with its own theta, in a
        # single call to :meth:`simulate`.
        H_sim = self.simulate_channel(theta_true, n_obs=n_obs)
        noise = self.sample_noise(torch.exp(theta_true[3]), n_obs)
        H, outlier_indices = self.zero_channels(epsilon, H_sim)
        x_obs_mis = self.summary(H + noise)
        return x_obs_mis, outlier_indices

    # Summary function
    def temporalMomentsGeneral(self, Y, K=3):
        
        power = torch.fft.ifft(Y, dim=1).abs() ** 2                        # (n, Ns): all receivers in one FFT
        tau_k = self.tau ** torch.arange(K, device=self.tau.device).unsqueeze(1)  # (K, Ns): tau^0, tau^1, tau^2
        moments = torch.trapz(power.unsqueeze(1) * tau_k, self.tau, dim=-1)       # (n, K)
        
        return torch.log(moments.double()).float()
        
    
    def simulate_channel(self,theta,n_obs,):
        
        theta = torch.as_tensor(theta, dtype=torch.float32)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        if theta.shape[0] == 1:
            theta = theta.expand(n_obs, -1)
        G0, T, lambda_0, _ = torch.exp(theta).unbind(dim=-1)

        # channel matrix H
        H = torch.zeros((n_obs, self.Ns), dtype=torch.float32)

        for jR in range(n_obs):
            mu_poisson = lambda_0[jR] * self.t_max
            n_points = int(torch.poisson(mu_poisson))
            delays = torch.sort(torch.rand(n_points) * self.t_max)[0]

            alpha = torch.zeros(n_points, dtype=torch.float32)
            sigma2 = G0[jR] * torch.exp(-delays / T[jR]) / lambda_0[jR] * self.B

            std = torch.sqrt(sigma2 / 2)                                           # (n_points,)
            alpha = torch.complex(torch.normal(0.0, std), torch.normal(0.0, std))  # complex64, (n_points,)
            alpha[delays < self.tau0] = 0
            
            # H[jR,f]
            H[jR, :] = torch.matmul(
                torch.exp(
                    -1j * 2 * torch.pi * self.delta_f
                    * torch.outer(torch.arange(self.Ns), delays)
                ),
                alpha,
            )
        return H 

    def zero_channels(self, epsilon, H):

        # choose outlier receivers
        outlier_indices = torch.empty(0, dtype=torch.float32)
        if epsilon > 0.0:
            n_outliers = int(epsilon * H.shape[0])
            if n_outliers > 0:
                outlier_indices = torch.randperm(H.shape[0])[:n_outliers]
                H[outlier_indices, :] = 0.0  # kill the channel, leave only noise
        
        return H, outlier_indices

    def sample_noise(self, sigma2_N: torch.Tensor, n: int) -> torch.Tensor:
        """Draw complex white Gaussian noise for `n` receivers.

        Parameters
        ----------
        sigma2_N : torch.Tensor
            Noise variance (natural units, not log): a scalar shared by all
            receivers, or shape ``(n,)``, one value per receiver.
        n : int
            Number of receivers.

        Returns
        -------
        torch.Tensor
            Complex noise of shape ``(n, Ns)``; real and imaginary parts
            each have variance ``sigma2_N / 2``.
        """
        std = torch.sqrt(torch.as_tensor(sigma2_N, device=self.device) / 2)
        if std.dim() == 1:
            std = std.unsqueeze(-1).expand(n, self.Ns)
            normal = torch.distributions.Normal(0, std)
            return normal.sample() + 1j * normal.sample()
        normal = torch.distributions.Normal(0, std)
        return normal.sample((n, self.Ns)) + 1j * normal.sample((n, self.Ns))

    def summary(self, Y, outlier_indices=None, output="moments"):

        if output == "moments":
        
            temporal_moments =self.temporalMomentsGeneral(Y)
            return temporal_moments
        
        elif output == "data":
                
            # go to time domain, compute power
            power = torch.fft.ifft(Y, dim=1).abs() ** 2       # (n, Ns): all receivers in one FFT
            p_dB = 10 * torch.log10(power + 1e-12)            # avoid log(0)
                
            return p_dB.detach().cpu(), outlier_indices.detach().cpu(), self.tau.detach().cpu()
