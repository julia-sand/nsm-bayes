"""
This module contains simulator functions used in the project.
"""

import torch
from torch.distributions import Binomial, Poisson, NegativeBinomial

def sample_gandk_fully_reparameterized(gamma, n=1):
    """
    Samples from the g-and-k distribution using a fully reparameterized vector.

    Args:
        gamma (torch.Tensor): A 1D tensor of shape (4,) containing the
                              TRANSFORMED parameters [A, log(B), g, log(k)].
                                    
    Returns:
        torch.Tensor: A single scalar data point x sampled from the distribution.
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
    exp_gz = torch.exp(-g * z).clamp(min=1e-30, max=1e30)
    # term_g = (1 - exp_gz) / (1 + exp_gz)
    term_g = torch.tanh(0.5 * g * z)
    term_k = (1 + z**2)**k
    x = A + B * (1 + 0.8 * term_g) * term_k * z
    return x.squeeze()


def simulate_sir(
    theta: torch.Tensor,
    T: int = 100,
    N: int = 10_000,
    dt: float = 1.0,
    obs_model: str = "poisson",   # "poisson" or "negbin"
) -> torch.Tensor:
    """
    Stochastic discrete-time SIR with Binomial transitions.

    theta: (n, d_theta)
      If d_theta==4: (log_beta, log_gamma, logit_rho, log_I0)
      If d_theta==5: add log_phi for NegBin overdispersion
    Returns:
      y: (n, T) observed new cases per day (or per dt)
    """
    assert theta.dim() == 2
    n, d = theta.shape
    device = theta.device
    dtype = theta.dtype

    log_beta = theta[:, 0]
    log_gamma = theta[:, 1]
    logit_rho = theta[:, 2]
    log_I0 = theta[:, 3]

    beta = torch.exp(log_beta)
    gamma = torch.exp(log_gamma)
    rho = torch.sigmoid(logit_rho)
    I0 = torch.clamp(torch.round(torch.exp(log_I0)), min=1.0).to(dtype=dtype)

    if d == 5:
        log_phi = theta[:, 4]
        phi = torch.exp(log_phi)  # overdispersion/shape-like, depending on parameterization

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
        # p_inf = 1 - exp(-beta * I/N * dt)
        p_inf = 1.0 - torch.exp(-beta * (It / float(N)) * dt)
        p_inf = p_inf.clamp(0.0, 1.0)

        # Recovery probability over dt
        # p_rec = 1 - exp(-gamma * dt)
        p_rec = 1.0 - torch.exp(-gamma * dt)
        p_rec = p_rec.clamp(0.0, 1.0)

        new_inf = Binomial(total_count=St, probs=p_inf).sample()
        new_rec = Binomial(total_count=It, probs=p_rec).sample()

        S[:, t + 1] = St - new_inf
        I[:, t + 1] = It + new_inf - new_rec
        R[:, t + 1] = R[:, t] + new_rec

        # Observations: reported incident infections
        mu = (rho * new_inf).clamp_min(0.0)

        if obs_model == "poisson" or d == 4:
            y[:, t] = Poisson(mu).sample()
        else:
            # One common NegBin parameterization: mean mu, variance mu + mu^2/phi
            # Convert to total_count (phi) and probs = phi/(phi+mu)
            probs = (phi / (phi + mu + 1e-8)).clamp(1e-6, 1 - 1e-6)
            y[:, t] = NegativeBinomial(total_count=phi, probs=probs).sample()

    return y

def sir_summary(y: torch.Tensor, N: float) -> torch.Tensor:
    """
    y: (n, T) incidence
    Returns: (n, 3) [attack_rate, t_peak_scaled, peak_scaled]
    """
    y = y.to(torch.float32)
    n, T = y.shape

    attack = y.sum(dim=1) / float(N)
    peak = y.max(dim=1).values / float(N)
    t_peak = y.argmax(dim=1).to(torch.float32) / float(max(T - 1, 1))

    return torch.stack([attack, t_peak, peak], dim=1)


def TurinModel(
    theta,
    B=4e9,
    Ns=801,
    N=50,
    tau0=0,
    output = "moments",
    epsilon=0.0,
    device="cpu"
):
    """
    Simulates the Turin channel model and optionally introduces outlier time series.

    Parameters
    ----------
    theta : torch.Tensor
        Parameter vector [G0, T, lambda_0, sigma2_N].
    B : float
        Bandwidth in Hz.
    Ns : int
        Number of frequency samples.
    N : int
        Number of time series (receivers).
    tau0 : float
        Minimum delay threshold.
    output : str
        "moments" to return 6D summary statistics, "data" for full time series.
    epsilon : float
        Fraction (0–1) of outlier time series to replace with pure noise.
    device : str
        Device to use ("cpu" or "cuda").
    """
    # unpack theta
    G0, T, lambda_0, sigma2_N = torch.exp(theta.to(device))
    nRx = N

    delta_f = B / (Ns - 1)
    t_max = 1 / delta_f
    tau = torch.linspace(0, t_max, Ns, device=device)

    # channel matrix H
    H = torch.zeros((nRx, Ns), dtype=torch.cfloat, device=device)
    mu_poisson = lambda_0 * t_max

    for jR in range(nRx):
        n_points = int(torch.poisson(mu_poisson))
        delays = torch.sort(torch.rand(n_points, device=device) * t_max)[0]

        alpha = torch.zeros(n_points, dtype=torch.cfloat, device=device)
        sigma2 = G0 * torch.exp(-delays / T) / lambda_0 * B

        for l in range(n_points):
            if delays[l] < tau0:
                alpha[l] = 0
            else:
                std = torch.sqrt(sigma2[l] / 2)
                alpha[l] = torch.normal(0, std) + 1j * torch.normal(0, std)

        # H[jR,f]
        H[jR, :] = torch.matmul(
            torch.exp(
                -1j * 2 * torch.pi * delta_f
                * torch.outer(torch.arange(Ns, device=device), delays)
            ),
            alpha,
        )

    # AWGN
    normal = torch.distributions.normal.Normal(0, torch.sqrt(sigma2_N / 2))
    Noise = normal.sample((nRx, Ns)) + 1j * normal.sample((nRx, Ns))
    Noise = Noise.to(device)

    # choose outlier receivers
    outlier_indices = torch.empty(0, dtype=torch.long, device=device)
    if epsilon > 0.0:
        n_outliers = int(epsilon * nRx)
        if n_outliers > 0:
            outlier_indices = torch.randperm(nRx, device=device)[:n_outliers]
            H[outlier_indices, :] = 0.0  # kill the channel, leave only noise

    # received freq response
    Y = H + Noise

    # go to time domain, compute power
    y_td = torch.fft.ifft(Y, dim=1)               # shape (N, Ns), complex
    p = torch.abs(y_td) ** 2                      # power
    p_dB = 10 * torch.log10(p + 1e-12)            # avoid log(0)

    # Summary function
    def temporalMomentsGeneral(Y, K=3, B=4e9):
        N, Ns = Y.shape
        delta_f = B / (Ns - 1)
        t_max = 1 / delta_f
        tau = torch.linspace(0, t_max, Ns, device=Y.device)
        out = torch.zeros((N, K), dtype=torch.float64, device=Y.device)

        for k in range(K):
            for i in range(N):
                y = torch.fft.ifft(Y[i, :])
                out[i, k] = torch.trapz(tau**k * (torch.abs(y) ** 2), tau)

        return torch.log(out)
    
    if output == "moments":
        temporal_moments = temporalMomentsGeneral(Y)
        return temporal_moments
    elif output == "data":
        return p_dB.detach().cpu(), outlier_indices.detach().cpu(), tau.detach().cpu()

