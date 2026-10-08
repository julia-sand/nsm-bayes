"""Contains functions for adding outliers and noise to datasets for testing inference"""

import torch 

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
    num_outliers = int(torch.floor(epsilon * n_obs))
    
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
