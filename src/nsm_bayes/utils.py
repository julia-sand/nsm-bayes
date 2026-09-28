import torch
import numpy as np
from torch.distributions import MultivariateNormal, Binomial, StudentT


####----Function to get simulations from a prior and simulator----####
def get_simulations(prior, simulator, num_samples, data_dim):
    # Sample from prior
    theta = prior.sample([num_samples])
    x_sim = torch.zeros(num_samples, data_dim) 
    # Sample from the simulator
    for i in range(num_samples):
        x_sim[i,:] = simulator(theta[i, :])
    return theta, x_sim

####----Function to run MCMC sampling using the NLE likelihood estimator----####
def run_mcmc(
    x_obs: torch.Tensor,
    inference,
    num_pos_samples: int = 250,
    mcmc_method: str = "slice_np",
    num_chains: int = 4,
    num_workers: int = 4,
    thin: int = 10,
    warmup_steps: int = 500,
    init_strategy: str = "proposal"
) -> torch.Tensor:
    """
    Builds a posterior from a trained sbi inference object and runs MCMC.

    Args:
        x_obs (torch.Tensor): The observed data to condition the posterior on.
        inference: The trained sbi inference object (e.g., SNLE).
        num_pos_samples (int): The number of posterior samples to generate per chain.
        mcmc_method (str): The MCMC algorithm to use (e.g., 'slice_np', 'nuts').
        num_chains (int): The number of independent MCMC chains to run.
        thin (int): The thinning factor to reduce sample autocorrelation.
        warmup_steps (int): The number of burn-in steps for each chain.
        init_strategy (str): How to find the starting points for the chains.

    Returns:
        torch.Tensor: A tensor of posterior samples, with shape 
                      [num_chains * num_pos_samples, d_theta].
    """

    posterior = inference.build_posterior(
        sample_with="mcmc",
        mcmc_method=mcmc_method
    )
    
    posterior_samples = posterior.sample(
        sample_shape=(num_pos_samples,), 
        x=x_obs,
        num_chains=num_chains,
        num_workers=num_workers,
        thin=thin,
        warmup_steps=warmup_steps,
        init_strategy=init_strategy
    )
    
    return posterior_samples

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
    num_outliers = int(np.floor(epsilon * n_obs))
    
    if num_outliers == 0:
        print("Warning: Epsilon is too small to add any outliers for this dataset size.")
        return modified_data, torch.tensor([])
        
    all_indices = torch.randperm(n_obs)
    outlier_indices = all_indices[:num_outliers]
    
    for idx in outlier_indices:
        # Randomly choose one of the provided outlier values
        random_outlier_value = np.random.choice(outlier_values)
        
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

def compute_mmd_lengthscale(y: torch.Tensor) -> torch.Tensor:
    """
    Computes the MMD kernel lengthscale using the median heuristic.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
        
    sq_dists = torch.pdist(y) ** 2 
    median_sq_dist = torch.median(sq_dists)
    
    return torch.sqrt(median_sq_dist / 2.0)

def kernel_matrix(x: torch.Tensor, y: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Computes the Gaussian RBF kernel matrix."""
    sq_dists = torch.cdist(x, y, p=2)**2
    return torch.exp(-sq_dists / (2 * l**2))

def compute_mmd(x: torch.Tensor, y: torch.Tensor, lengthscale: torch.Tensor) -> float:
    """
    Computes the V-statistic estimator of the squared MMD.
    """
    if x.dim() == 1: x = x.unsqueeze(-1)
    if y.dim() == 1: y = y.unsqueeze(-1)
    
    m, n = x.shape[0], y.shape[0]

    K_xx = kernel_matrix(x, x, lengthscale)
    K_yy = kernel_matrix(y, y, lengthscale)
    K_xy = kernel_matrix(x, y, lengthscale)

    mmd_sq = (1 / (m * m)) * K_xx.sum() + (1 / (n * n)) * K_yy.sum() - (2 / (m * n)) * K_xy.sum()
    
    return mmd_sq.item()

def to_tensor_matrix(arr: object) -> torch.Tensor:
    """
    Convert loaded samples (numpy or torch) to a 2D torch.Tensor (N, d).
    Flattens leading dims except the last.
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    elif isinstance(arr, list):
        arr = torch.tensor(arr)
    # now arr is torch.Tensor
    arr = arr.float()
    # reshape to (N, d_theta)
    arr = arr.reshape(-1, arr.shape[-1])
    return arr

def sample_from_case1_gaussian(mu: torch.Tensor,
                               Sigma: torch.Tensor,
                               num_samples: int) -> torch.Tensor:
    """
    Draw samples from N(mu, Sigma) with small jitter on Sigma if needed.
    Returns (num_samples, d_theta) torch.Tensor.
    """
    mu = mu.float()
    Sigma = Sigma.float()

    try:
        dist = MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    except RuntimeError:
        # add tiny jitter if covariance is near-singular
        d = Sigma.shape[0]
        jitter = 1e-6 * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
        dist = MultivariateNormal(loc=mu, covariance_matrix=Sigma + jitter)

    samps = dist.sample((num_samples,))  # (num_samples, d_theta)
    return samps

# Function to compute the inverse covariance matrix of the data used in objective J(phi)
def compute_inverse_covariance(x, regularize_eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the inverse covariance matrix of the data x in a robust way.

    Args:
        x (torch.Tensor or np.ndarray): 
            Input data. Can be 1D with shape (n_samples,) or 
            2D with shape (n_samples, n_features).
        
        regularize_eps (float): 
            A small epsilon (jitter) added to the diagonal of the covariance 
            matrix for numerical stability before inversion. This helps prevent
            errors from singular matrices.

    Returns:
        torch.Tensor: 
            The inverse covariance matrix (Sigma_inv).
            Shape is (1, 1) for 1D input, or (n_features, n_features) for 2D.
    """
    # 1. Ensure x is a PyTorch tensor and is at least 2D
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        # Ensure float type for covariance calculation
        x = x.to(dtype=torch.float32)

    # Handle the 1D case by reshaping to a 2D column vector
    if x.dim() == 1:
        x = x.unsqueeze(1)

    # 2. Check for sufficient samples
    n_samples, n_features = x.shape
    if n_samples < 2:
        return torch.eye(n_features, device=x.device, dtype=x.dtype)

    # 3. Compute the covariance matrix
    covariance_matrix = torch.cov(x.T)

    # 4. Regularize and Invert the matrix
    identity_matrix = torch.eye(n_features, device=x.device, dtype=x.dtype)
    reg_covariance_matrix = covariance_matrix + regularize_eps * identity_matrix
    
    inverse_cov = torch.linalg.inv(reg_covariance_matrix)

    return inverse_cov


def sample_mean_and_covariance(x_obs: torch.Tensor):
    """
    Compute the sample mean and (unbiased) sample covariance of x_obs.

    Args:
        x_obs (Tensor): shape (n, d)

    Returns:
        mean (Tensor): shape (d,)
        cov  (Tensor): shape (d, d)
    """
    if x_obs.ndim != 2:
        raise ValueError("x_obs must have shape (n, d)")

    n = x_obs.shape[0]

    # Sample mean
    mean = x_obs.mean(dim=0)

    # Centered data
    x_centered = x_obs - mean

    # Sample covariance (unbiased, divide by n-1)
    cov = (x_centered.T @ x_centered) / (n - 1)

    return mean, cov

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