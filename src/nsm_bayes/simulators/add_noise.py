"""Misspecification helpers: outliers, heavy-tailed noise and contamination.

These functions corrupt a fraction ``epsilon`` of a clean observed data set
so that the robustness of the posteriors can be tested.
"""
import torch

from torch.distributions import Binomial, StudentT


def add_outliers_by_proportion(data, epsilon, outlier_values):
    """Shift a random fraction of observations by a fixed outlier offset.

    ``floor(epsilon * n_obs)`` rows are chosen without replacement. For each
    one, a value is drawn uniformly from `outlier_values` and **added** to
    every coordinate of the row.

    Parameters
    ----------
    data : torch.Tensor
        Clean data of shape ``(n_obs, d_x)``; not modified.
    epsilon : float
        Fraction of rows to corrupt, e.g. ``0.1`` for 10%.
    outlier_values : sequence of float
        Offsets to choose from, e.g. ``[10.0, -10.0]``.

    Returns
    -------
    modified_data : torch.Tensor
        Copy of `data` with the selected rows shifted.
    outlier_indices : torch.Tensor
        Indices of the shifted rows; an empty float tensor if no row is
        corrupted (a warning is printed in that case).
    """
    modified_data = data.clone()
    
    n_obs, d_x = data.shape
    
    # Calculate the number of outliers to add
    num_outliers = int(torch.floor(torch.tensor(epsilon * n_obs)).item())
    
    if num_outliers == 0:
        print("Warning: Epsilon is too small to add any outliers for this dataset size.")
        return modified_data, torch.tensor([])
        
    all_indices = torch.randperm(n_obs)
    outlier_indices = all_indices[:num_outliers]
    
    for idx in outlier_indices:
        # Randomly choose one of the provided outlier values
        random_outlier_value = outlier_values[torch.randint(0,len(outlier_values),size=(1,)).item()]
        
        # Create the outlier tensor
        outlier_tensor = torch.full((d_x,), fill_value=float(random_outlier_value), 
                                    dtype=data.dtype, device=data.device)
        
        # Replace the data at the chosen index
        modified_data[idx, :] += outlier_tensor 
        
    return modified_data, outlier_indices

def add_student_t_noise(
    x_obs: torch.Tensor,
    epsilon: float,
    df: float = 3.0,
    noise_scale: float = 1.0,
    use_robust_scale: bool = False,
    eps: float = 1e-8,
):
    """Add Student-t noise to a random fraction of the observations.

    Parameters
    ----------
    x_obs : torch.Tensor
        Observed data of shape ``(n_obs, d)``; not modified.
    epsilon : float
        Fraction in ``[0, 1]`` of rows to corrupt; ``floor(epsilon * n_obs)``
        rows are chosen without replacement.
    df : float, default 3.0
        Degrees of freedom of the noise (``df=1`` gives Cauchy noise).
    noise_scale : float, default 1.0
        Multiplier on the per-dimension scale estimate.
    use_robust_scale : bool, default False
        Scale the noise by ``1.4826 * MAD`` instead of the (biased) standard
        deviation of each dimension.
    eps : float, default 1e-8
        Added to the scale estimate.

    Returns
    -------
    x_noisy : torch.Tensor or tuple
        Copy of `x_obs` with the selected rows perturbed, shape
        ``(n_obs, d)``. If no row is selected the function instead returns
        the tuple ``(x_noisy, corrupted_indices)`` with an empty index
        tensor (see Notes).

    Raises
    ------
    AssertionError
        If `x_obs` is not 2-D or `epsilon` is outside ``[0, 1]``.

    Notes
    -----
    The return type is inconsistent: a single tensor when noise is added,
    a 2-tuple when ``floor(epsilon * n_obs) == 0``.
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
    if use_robust_scale: #does this ever get used? 
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
    """Simulate a data set in which a random fraction uses other parameters.

    Each of the `n_obs` samples is independently marked as contaminated
    with probability `epsilon`. Clean samples are simulated at
    `theta_true`; contaminated ones at parameters drawn from `prior`.

    Parameters
    ----------
    theta_true : torch.Tensor
        True parameter vector of shape ``(d,)``.
    n_obs : int
        Number of samples.
    simulate_fn : callable
        Called as ``simulate_fn(theta, T=T, N=N)`` with ``theta`` of shape
        ``(n_obs, d)``.
    T : int
        Time horizon passed to `simulate_fn`.
    N : int
        Population size passed to `simulate_fn`.
    epsilon : float
        Probability that a sample is contaminated.
    contaminant : str, default "prior"
        ``"prior"`` draws contaminating parameters from `prior`. Any other
        value (or ``prior=None``) uses the fixed offset ``theta_true + 1``.
    prior : torch.distributions.Distribution, optional
        Distribution of the contaminating parameters.

    Returns
    -------
    y_obs_mis : torch.Tensor
        Output of `simulate_fn` for all samples.
    is_contam : torch.Tensor
        Boolean mask of shape ``(n_obs,)``.
    theta_used : torch.Tensor
        Parameters used for each sample, shape ``(n_obs, d)``.

    See Also
    --------
    nsm_bayes.simulators.benchmark_simulators.sir.SIRSimulator.generate_contaminated_data :
        Method version used by the SIR experiment.
    """
    device = theta_true.device
    dtype = theta_true.dtype

    # Determine which samples are contaminated
    is_contam = torch.rand(n_obs, device=device) < epsilon
    num_contam = is_contam.sum().item()

    theta_used = torch.zeros((n_obs, theta_true.shape[0]), device=device, dtype=dtype)

    # Clean samples use theta_true
    theta_used[~is_contam] = theta_true

    # Contaminated samples
    if contaminant == "prior" and prior is not None:
        theta_contam = prior.sample((num_contam,))
        theta_used[is_contam] = theta_contam
    else:
        # Fallback: just use theta_true for now or a fixed offset
        print("No contamination, only fixed offset.")
        theta_used[is_contam] = theta_true+1

    # Simulate data
    y_obs_mis = simulate_fn(theta_used, T=T, N=N)

    return y_obs_mis, is_contam, theta_used


def apply_undercounting_trajectory(
    y: torch.Tensor,          # (n_obs, T) integer counts
    epsilon: float = 0.05,     # fraction of contaminated trajectories
    q: float = 0.3,            # retention probability (one-sided: downward)
    per_time: bool = False,    # if True: contaminate per time point, else per trajectory
):
    """Undercount counts by binomial thinning (one-sided misspecification).

    Each selected count ``y`` is replaced by a draw from
    ``Binomial(y, q)``, so counts can only decrease.

    Parameters
    ----------
    y : torch.Tensor
        Integer-valued counts of shape ``(n_obs, T)``.
    epsilon : float, default 0.05
        Probability that a trajectory (or a single time point, see
        `per_time`) is thinned.
    q : float, default 0.3
        Retention probability of each counted case.
    per_time : bool, default False
        If false, whole trajectories are selected; if true, individual time
        points are.

    Returns
    -------
    y_corrupted : torch.Tensor
        Thinned copy of `y` with the same dtype.
    is_contam : torch.Tensor
        Boolean mask of shape ``(n_obs,)`` (trajectories) or
        ``(n_obs, T)`` (time points).

    See Also
    --------
    nsm_bayes.simulators.benchmark_simulators.sir.SIRSimulator.apply_undercounting_trajectory :
        Method version used by the experiments.
    """
    assert y.dim() == 2
    n_obs, T = y.shape
    device = y.device

    y_cor = y.clone()

    if not per_time:
        is_contam = (torch.rand(n_obs, device=device) < epsilon)  # (n_obs,)
        if is_contam.any():
            idx = is_contam.nonzero(as_tuple=True)[0]
            y_sub = y_cor[idx].float()
            y_cor[idx] = Binomial(total_count=y_sub, probs=torch.tensor(q, device=device)).sample().to(y.dtype)
        return y_cor, is_contam
    else:
        is_contam = (torch.rand(n_obs, T, device=device) < epsilon)  # (n_obs,T)
        if is_contam.any():
            y_sub = y_cor[is_contam].float()
            y_cor[is_contam] = Binomial(total_count=y_sub, probs=torch.tensor(q, device=device)).sample().to(y.dtype)
        return y_cor, is_contam
    