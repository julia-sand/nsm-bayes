"""Conjugate (neural exponential family) posterior computation."""
import torch
from torch.func import vmap

from nsm_bayes.utils.linalg import (
    get_cov_term_single,
    get_non_div_term_single,
    sigma_inverse,
    get_div_term_single,
)

from nsm_bayes.utils.weights import weight_function
from nsm_bayes.utils.statistics import estimate_stats

class ConjugatePosterior():

    def __init__(self,
                model,
                prior_mean, 
                prior_cov,
                robust,
                weight_type):

        self.model = model
        self.robust = robust
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.weight_type = weight_type

        #get the normalised model's mean and variance
        self.s, self.m = self.model.standardizer_theta.std, self.model.standardizer_theta.mean
        # Invert prior covariance to get prior precision
        self.prior_cov_norm = self.prior_cov / torch.outer(self.s, self.s)
        self.prior_cov_inv = sigma_inverse(self.prior_cov_norm)
        self.prior_mean_norm = self.model.standardizer_theta(self.prior_mean)

    def _normalise_input(self,x_obs):
        return self.model.standardizer_x(x_obs)

    def _unnormalise(self,mu_n_normalized,Sigma_n_normalized):

        mu_n = mu_n_normalized * self.s + self.m
        
        Sigma_n = Sigma_n_normalized * torch.outer(self.s, self.s) 

        return mu_n,Sigma_n

    def calibrated_posterior(self, 
                x_obs, 
                beta):
        """Posterior mean/cov in original theta units (standardise -> solve -> unstandardise)."""

        #normalise x values 
        #x_n = self.model.standardizer_x(x_obs)
        x_n = self._normalise_input(x_obs)

        mu_hat, Sigma_inv = estimate_stats(x_n, self.robust)
        
        w2, grad_w2 = weight_function(self.weight_type, x_n, mu_hat, Sigma_inv)
        
        mu_n_normalized, Sigma_n_normalized = self._compute_posterior_conjugate(
                    x_n,  
                    beta,
                    w2, 
                    grad_w2)

        mu_n, Sigma_n = self._unnormalise(mu_n_normalized, Sigma_n_normalized)
        return mu_n, Sigma_n

    #######----Function to compute posterior precision matrix for NSM-Bayes-conj----######
    def _compute_posterior_precision_conjugate(self, x_n, beta, w2):
        """
        Computes the posterior precision matrix (inverse covariance) for the model.
        """
        
        # Vectorize the calculation over all observations
        #batched_cov_terms = vmap(get_cov_term_single, in_dims=(0, None))(
        #                                            x_obs, self.model.T_phi
        #                                                )  # shape: (n, d_theta, d_theta)

        #sum_term = (w2.view(-1, 1, 1) * batched_cov_terms).sum(dim=0)
        A = self._compute_A(x_n,w2)
        # Assemble the final precision matrix
        Sigma_n_inv = self.prior_cov_inv + 2 * beta * A#sum_term

        return Sigma_n_inv

    def _compute_posterior_conjugate(self, x_n, beta, w2, grad_w2):
        """
        Computes the posterior mean and covariance for the neural exponential family model.

        Returns:
        - mu_n: (d_theta,) tensor for the posterior mean.
        - Sigma_n: (d_theta, d_theta) tensor for the posterior covariance.
        """

        # 1. Compute the posterior precision matrix
        Sigma_n_inv = self._compute_posterior_precision_conjugate(x_n, beta, w2)

        B = self._compute_B(x_n,w2,grad_w2)

        # Assemble the right-hand side of the linear system: Sigma_n_inv @ mu_n = rhs
        prior_term = self.prior_cov_inv @ self.prior_mean_norm
        rhs = prior_term - 2 * beta * (B)
        
        # Solve for mu_n using the stable solver.
        mu_n = torch.linalg.solve(Sigma_n_inv, rhs)

        # --- 4. Compute Posterior Covariance from Precision ---
        Sigma_n = sigma_inverse(Sigma_n_inv)
        
        return mu_n, Sigma_n

    def _compute_A(self,x_n,w2):

        # A = sum_i w2_i * J_T(x_i)^T J_T(x_i)   (d_theta, d_theta)
        # The networks are not batched: vmap them with in_dim=None. get_cov_term_
        # single(xi, fn) returns (J_T^T J_T, J_T); only the covariance term is kept.
        batched_cov_terms = vmap(get_cov_term_single, in_dims=(0, None))(
            x_n, self.model.T_phi
        )  # (n, d_theta, d_theta)
        A = (w2.view(-1, 1, 1) * batched_cov_terms).sum(dim=0)  # (d_theta, d_theta)
        return A 

    def _compute_B(self,x_n,w2,grad_w2):

        # B = sum_i [ w2_i * div_theta log p(x_i | theta)
        #             + grad_x w2(x_i)^T score_x(x_i) ]   (d_theta,)
        # get_non_div_term_single(xi, fn1, fn2, Jac1, Jac2) = Jac1 @ Jac2; both
        # Jacobians are recomputed per point and w2 is applied outside, since this
        # helper takes no weights.
        batched_non_div_terms = vmap(
            get_non_div_term_single, in_dims=(0, None, None)
        )(x_n, self.model.T_phi, self.model.b_phi)  # (n, d_theta)
        term_non_div = (w2.view(-1, 1) * batched_non_div_terms).sum(dim=0)  # (d_theta,)

        # term_div = sum_i div_x[ w(x_i)^2 * score_x(x_i) ]  (vector in R^{d_theta})
        # get_div_term_single(xi, model, w2, grad_w2) reads model.T_phi itself and
        # weights by w2 inside, so it is NOT re-applied when reducing below.
        batched_divergences = vmap(get_div_term_single, in_dims=(0, None, 0, 0))(
            x_n, self.model.T_phi, w2, grad_w2
        )  # (n, d_theta)
        term_div = batched_divergences.sum(dim=0)  # (d_theta,)

        B = term_non_div + term_div  # (d_theta,)

        return B

    
    def compute_theta_hat_true(self,
        x_obs: torch.Tensor,) -> torch.Tensor:
        """
        Compute the NSM-only minimiser theta_hat_true for NSM-Bayes-conj:
            theta_hat_true = argmin_theta L_NSM(theta; x_obs, phi_hat)

        This corresponds to the M-estimator used as the "true" parameter
        in the Syring & Martin calibration paper.

        Parameters
        ----------
        x_obs : torch.Tensor
            (n, d_x) observed data (already normalized if you use standardizer_x).
        w2,
        grad_w2 :
        mu_hat_obs : torch.Tensor
            Robust location of the data.
        Sigma_inv_obs : torch.Tensor
            Inverse robust covariance of the data.

        Returns
        -------
        torch.Tensor
            ``theta_hat_true`` of shape ``(d_theta,)`` in the same (normalized)
            theta space that T_phi produces.
        """
        x_n = self._normalise_input(x_obs)

        mu_hat, Sigma_inv = estimate_stats(x_n, self.robust)
        
        w2, grad_w2 = weight_function(self.weight_type, x_n, mu_hat, Sigma_inv)

        A = self._compute_A(x_n,w2)
        B = self._compute_B(x_n,w2,grad_w2)

        # ---------- Solve A * theta_hat_true = -B ----------
        d_theta = A.shape[0]
        scale = torch.trace(A) / A.shape[0]
        lam = 1e-2 * scale + 1e-12
        A_reg = 0.5 * (A + A.T) + lam * torch.eye(d_theta, device=A.device, dtype=A.dtype)

        try:
            L = torch.linalg.cholesky(A_reg)
            theta_hat_true = -torch.cholesky_solve(B.unsqueeze(1), L).squeeze(1)
        except RuntimeError:
            theta_hat_true = -torch.linalg.lstsq(A_reg, B).solution

        return theta_hat_true * self.s + self.m
