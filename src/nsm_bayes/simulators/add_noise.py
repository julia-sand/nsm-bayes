"""Contains functions for adding outliers and noise to datasets for testing inference"""
import torch
from torch.distributions import Binomial, StudentT


def add_student_t_noise(
    x_obs: torch.Tensor,
    epsilon: float,
    df: float = 3.0,
    noise_scale: float = 1.0,
    use_robust_scale: bool = False,
    eps: float = 1e-8,
):
    """
    Add Student-t heavy-tailed noise to an epsilon fraction of samples.

    Args
    ----
    x_obs : [n_obs, d]
        Observed summary statistics.
    epsilon : float in [0, 1]
        Fraction of samples to corrupt.
    df : float
        Degrees of freedom (df=1 -> Cauchy).
    noise_scale : float
        Multiplier on the estimated scale.
    use_robust_scale : bool
        If True, use MAD instead of std for scaling.
    eps : float
        Numerical stability.

    Returns
    -------
    x_noisy : [n_obs, d]
        Observed data with sparse heavy-tailed corruption.
    corrupted_indices : LongTensor
        Indices of samples that were corrupted.
    """
    assert x_obs.dim() == 2
    assert 0.0 <= epsilon <= 1.0

    n_obs, d = x_obs.shape
    device, dtype = x_obs.device, x_obs.dtype

    x_noisy = x_obs.clone()

    # Number of samples to corrupt
    num_corrupt = int(torch.floor(torch.tensor(epsilon * n_obs)).item())
    if num_corrupt == 0:
        return x_noisy, torch.empty(0, dtype=torch.long, device=device)

    # Randomly choose which samples to corrupt
    perm = torch.randperm(n_obs, device=device)
    corrupted_indices = perm[:num_corrupt]

    # Estimate scale per dimension (using all samples)
    if use_robust_scale:
        median = x_obs.median(dim=0).values
        mad = (x_obs - median).abs().median(dim=0).values
        scale = 1.4826 * mad + eps
    else:
        scale = x_obs.std(dim=0, unbiased=False) + eps

    # Student-t noise
    t_dist = StudentT(df=df)
    noise = t_dist.sample((num_corrupt, d)).to(device=device, dtype=dtype)

    # Scale noise
    noise = noise * scale * noise_scale

    # Corrupt selected samples
    x_noisy[corrupted_indices] += noise

    return x_noisy

def simulate_contaminated_dataset(
    theta_true,
    n_obs,
    simulate_fn,
    T,
    N,
    epsilon,
    contaminant="prior",
    prior=None,
):
    """
    Generate a dataset where a fraction epsilon of the samples are contaminated.

    Args
    ----
    theta_true : Tensor [d]
        True parameter vector.
    n_obs : int
        Number of observed samples.
    simulate_fn : callable
        Function to simulate data given theta, T, N.
    T : int
        Simulation time.
    N : int
        Population size.
    epsilon : float
        Fraction of contaminated samples.
    contaminant : str
        Type of contamination ("prior" or "fixed").
    prior : Distribution
        Prior distribution for contamination if contaminant="prior".

    Returns
    -------
    y_obs_mis : Tensor
        Simulated data (potentially contaminated).
    is_contam : Tensor [n_obs]
        Boolean mask of contaminated samples.
    theta_used : Tensor [n_obs, d]
        Parameters used for each sample.
    """
    device = theta_true.device
    dtype = theta_true.dtype

    # Determine which samples are contaminated
    is_contam = torch.rand(n_obs, device=device) < epsilon
    num_contam = is_contam.sum().item()
    num_clean = n_obs - num_contam

    theta_used = torch.zeros((n_obs, theta_true.shape[0]), device=device, dtype=dtype)

    # Clean samples use theta_true
    theta_used[~is_contam] = theta_true

    # Contaminated samples
    if contaminant == "prior" and prior is not None:
        theta_contam = prior.sample((num_contam,))
        theta_used[is_contam] = theta_contam
    else:
        # Fallback: just use theta_true for now or a fixed offset
        theta_used[is_contam] = theta_true

    # Simulate data
    y_obs_mis = simulate_fn(theta_used, T=T, N=N)

    return y_obs_mis, is_contam, theta_used
