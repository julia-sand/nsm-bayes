"""
This module contains functions for simulating the Turin channel model.
"""

import torch


def TurinModel(
    theta,
    B=4e9,
    Ns=801,
    N=50,
    tau0=0,
    output="moments",
    epsilon=0.0,
    device="cpu",
):
    """Simulate the Turin channel model and optionally introduce outlier time series.

    Parameters
    ----------
    theta : torch.Tensor
        Parameter vector [G0, T, lambda_0, sigma2_N].
    B : float, default 4e9
        Bandwidth in Hz.
    Ns : int, default 801
        Number of frequency samples.
    N : int, default 50
        Number of time series (receivers).
    tau0 : float, default 0
        Minimum delay threshold.
    output : str, default "moments"
        "moments" to return 6D summary statistics, "data" for full time series.
    epsilon : float, default 0.0
        Fraction (0–1) of outlier time series to replace with pure noise.
    device : str, default "cpu"
        Device to use ("cpu" or "cuda").

    Returns
    -------
    torch.Tensor or tuple
        If output == "moments", returns temporal moments.
        If output == "data", returns (p_dB, outlier_indices, tau).
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
    y_td = torch.fft.ifft(Y, dim=1)  # shape (N, Ns), complex
    p = torch.abs(y_td) ** 2  # power
    p_dB = 10 * torch.log10(p + 1e-12)  # avoid log(0)

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
