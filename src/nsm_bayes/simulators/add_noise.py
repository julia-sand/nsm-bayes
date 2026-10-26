"""Misspecification helper: additive outliers.

This function corrupts a fraction ``epsilon`` of a clean observed data set
so that the robustness of the posteriors can be tested.
"""
import torch


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

