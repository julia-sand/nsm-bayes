"""Conjugate (neural exponential family) posterior computation."""
import torch
from torch.func import jacrev, vmap


from nsm_bayes.utils.linalg import (
    get_cov_term_single,
    get_non_div_term_single,
    sigma_inverse,
)


#######----Function to compute posterior precision matrix for NSM-Bayes-conj----######
def compute_posterior_precision_conjugate(
    x_obs, model, beta, prior_cov_inv, weight_fn, mu_hat_obs, Sigma_inv_obs, c
):
    """
    Computes the posterior precision matrix (inverse covariance) for the model.

    """

    # Vectorize the calculation over all observations
    batched_cov_terms,_ = vmap(get_cov_term_single)(x_obs,model.T_phi) # shape: (n, d_theta, d_theta)

    # Apply weights and sum
    weights = weight_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c).view(-1, 1, 1)

    sum_term = (weights * batched_cov_terms).sum(dim=0)

    # Assemble the final precision matrix
    Sigma_n_inv = prior_cov_inv + 2 * beta * sum_term

    return Sigma_n_inv

def compute_posterior_conjugate(
    x_obs, model, beta, prior_mean, prior_cov, weight_fn, mu_hat_obs, Sigma_inv_obs, c
):
    """
    Computes the posterior mean and covariance for the neural exponential family model.

    Returns:
    - mu_n: (d_theta,) tensor for the posterior mean.
    - Sigma_n: (d_theta, d_theta) tensor for the posterior covariance.
    """
    
    # Invert prior covariance to get prior precision
    prior_cov_inv = sigma_inverse(prior_cov)
    #torch.linalg.inv(prior_cov)
    
    # 1. Compute the posterior precision matrix
    Sigma_n_inv = compute_posterior_precision_conjugate(
        x_obs, model, beta, prior_cov_inv, weight_fn, mu_hat_obs, Sigma_inv_obs, c
    )

    # --- 2. Calculate terms needed for the posterior mean ---


    batched_non_div_terms = vmap(get_non_div_term_single, in_dims=(0, None))(x_obs,model)
    weights_h,grad_w2 = weight_fn(x_obs, mu_hat_obs, Sigma_inv_obs, c)
    term_non_div = (weights_h * batched_non_div_terms).sum(dim=0)

    batched_divergences = vmap(
        get_div_term_single, in_dims=(0, None, None, None, None)
                 )(x_obs, model, w2=weights_h,grad_w2=grad_w2)          # (n, d_theta)
    
    #batched_divergences = vmap(get_div_term_single)(x_obs)
    term_div = batched_divergences.sum(dim=0)
    
    # --- 3. Solve for the Posterior Mean ---
    
    # Assemble the right-hand side of the linear system: Sigma_n_inv @ mu_n = rhs
    prior_term = prior_cov_inv @ prior_mean
    rhs = prior_term - 2 * beta * (term_div + term_non_div)
    
    # Solve for mu_n using the stable solver.
    mu_n = torch.linalg.solve(Sigma_n_inv, rhs)

    # --- 4. Compute Posterior Covariance from Precision ---
    
    # Use a numerically stable Cholesky-based inverse
    #L = torch.linalg.cholesky(Sigma_n_inv)
    Sigma_n = sigma_inverse(Sigma_n_inv)#torch.cholesky_inverse(L)
    
    return mu_n, Sigma_n


    