"""
This module contains functions for simulating the SIR model.
"""

import torch
from torch.distributions import Binomial, NegativeBinomial, Poisson


def simulate_sir(
    theta: torch.Tensor,
    T: int = 100,
    N: int = 10_000,
    dt: float = 1.0,
    obs_model: str = "poisson",  # "poisson" or "negbin"
) -> torch.Tensor:
    """Stochastic discrete-time SIR with Binomial transitions.

    Parameters
    ----------
    theta : torch.Tensor
        Parameter tensor of shape (n, d_theta).
        If d_theta == 4: (log_beta, log_gamma, logit_rho, log_I0).
        If d_theta == 5: add log_phi for NegBin overdispersion.
    T : int, default 100
        Time horizon.
    N : int, default 10_000
        Population size.
    dt : float, default 1.0
        Time step.
    obs_model : str, default "poisson"
        Observation model: "poisson" or "negbin".

    Returns
    -------
    torch.Tensor
        y: (n, T) observed new cases per day (or per dt).
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
