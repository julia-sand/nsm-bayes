"""Contains functions for adding outliers and noise to datasets for testing inference"""
import numpy as np  
import torch
from torch.distributions import Binomial, MultivariateNormal, StudentT

####----Function to add outliers to the observed data----####
def add_outliers_by_proportion(data, epsilon, outlier_values):
    """
    Randomly replaces a proportion `epsilon` of observations with outliers.

    Args:
        data (torch.Tensor): The original dataset of shape (n_obs, d_x).
        epsilon (float): The proportion of data to replace (e.g., 0.1 for 10%).
        outlier_values (list or tuple): A list of values to sample outliers from
                                        (e.g., [10.0, -10.0]).

    Returns:
        tuple: A tuple containing:
            - modified_data (torch.Tensor): A new tensor with outliers.
            - outlier_indices (torch.Tensor): A tensor of the indices that were replaced.
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
        random_outlier_value = outlier_values[torch.randint(len(outlier_values), (1,)).item()]
        #np.random.choice(outlier_values)
        
        # Create the outlier tensor
        outlier_tensor = torch.full((d_x,), fill_value=float(random_outlier_value), 
                                    dtype=data.dtype, device=data.device)
        
        # Replace the data at the chosen index
        modified_data[idx, :] += outlier_tensor 
        
    return modified_data, outlier_indices

def simulate_contaminated_dataset(
    theta_true: torch.Tensor,          
    n_obs: int,
    simulate_fn,
    T: int,
    N: int,
    epsilon: float = 0.1,
    contaminant: str = "prior",
    prior=None,
    theta_contam: torch.Tensor | None = None,
):
    device, dtype = theta_true.device, theta_true.dtype
    d_theta = theta_true.numel()

    is_contam = (torch.rand(n_obs, device=device) < epsilon)

    theta_batch = theta_true.unsqueeze(0).repeat(n_obs, 1)

    if contaminant == "prior":
        assert prior is not None
        theta_bad = prior.sample((n_obs,)).to(device=device, dtype=dtype)
        theta_batch[is_contam] = theta_bad[is_contam]
    elif contaminant == "fixed":
        assert theta_contam is not None
        theta_batch[is_contam] = theta_contam.unsqueeze(0).expand(is_contam.sum(), d_theta)
    else:
        raise ValueError("contaminant must be 'prior' or 'fixed'")

    y = simulate_fn(theta_batch, T=T, N=N)
    return y, is_contam, theta_batch


def apply_undercounting_trajectory(
    y: torch.Tensor,          # (n_obs, T) integer counts
    epsilon: float = 0.05,     # fraction of contaminated trajectories
    q: float = 0.3,            # retention probability (one-sided: downward)
    per_time: bool = False,    # if True: contaminate per time point, else per trajectory
):
    """
    One-sided undercounting by binomial thinning.
    Returns: y_corrupted, is_contam_mask
      - if per_time=False: is_contam_mask is (n_obs,) for contaminated trajectories
      - if per_time=True:  is_contam_mask is (n_obs, T) for contaminated time points
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