r"""Closed-form generalised posterior for the conjugate NSM-Bayes method.

With the surrogate :math:`\log q_\phi(x \mid \theta) = \theta^\top
T_\phi(x) + b_\phi(x)`, the weighted score-matching loss summed over the
observations is quadratic in :math:`\theta`,

.. math::

    \sum_{i=1}^n \ell(\theta; x_i) = \theta^\top A\, \theta
        + 2\, \theta^\top B + \text{const},

with

.. math::

    A = \sum_i w^2(x_i)\, J_T(x_i) J_T(x_i)^\top, \qquad
    B = \sum_i \Big[ w^2(x_i)\, J_T(x_i) \nabla b(x_i)
        + J_T(x_i) \nabla w^2(x_i) + w^2(x_i)\, \Delta T(x_i) \Big],

where :math:`J_T` is the ``(d_theta, d_x)`` Jacobian of :math:`T_\phi` and
:math:`\Delta T` its component-wise Laplacian.  With a Gaussian prior
:math:`N(\mu_0, \Sigma_0)`, the generalised posterior
:math:`\pi(\theta) \exp\{-\beta \sum_i \ell(\theta; x_i)\}` is Gaussian with

.. math::

    \Sigma_n^{-1} = \Sigma_0^{-1} + 2\beta A, \qquad
    \Sigma_n^{-1} \mu_n = \Sigma_0^{-1} \mu_0 - 2\beta B.

All of this is computed in the standardised ``x`` and ``theta`` spaces of
the model and mapped back to original units.
"""
import torch
from torch.func import vmap

from nsm_bayes.utils.linalg import (
    jacobian_and_laplacian,
    jacobian,
    get_cov_term_single,
    get_non_div_term_single,
    sigma_inverse,
    get_div_term_single,
)

from nsm_bayes.utils.weights import weight_function
from nsm_bayes.utils.statistics import estimate_stats

class ConjugatePosterior():
    """Gaussian generalised posterior for a fitted :class:`ConjugateModel`.

    Parameters
    ----------
    model : ConjugateModel
        Trained surrogate whose standardisers have been fitted.
    prior_mean : torch.Tensor
        Gaussian prior mean in original ``theta`` units, shape
        ``(d_theta,)``.
    prior_cov : torch.Tensor
        Gaussian prior covariance in original units, shape
        ``(d_theta, d_theta)``.
    robust : bool
        If true, the weight function is centred and scaled with the median
        and the minimum covariance determinant estimate of the
        (standardised) data; otherwise with the sample mean and covariance.
        See :func:`~nsm_bayes.utils.statistics.estimate_stats`.
    weight_type : {"imq", "huber", "none"}
        Weight function ``w`` in the score-matching loss; see
        :func:`~nsm_bayes.utils.weights.weight_function`.

    Attributes
    ----------
    s, m : torch.Tensor
        Standard deviation and mean of the model's ``theta`` standardiser,
        each of shape ``(d_theta,)``.
    prior_mean_norm : torch.Tensor
        Prior mean in standardised ``theta`` space.
    prior_cov_norm : torch.Tensor
        Prior covariance in standardised ``theta`` space.
    prior_cov_inv : torch.Tensor
        Inverse of `prior_cov_norm`.

    See Also
    --------
    nsm_bayes.inference.conjugate.calibrate_beta.calibrate_beta_gpc :
        Chooses ``beta`` for :meth:`calibrated_posterior`.

    Notes
    -----
    The data weights are recomputed from whichever data set is passed to
    :meth:`calibrated_posterior`, so each bootstrap resample gets its own
    robust centre and scale.
    """

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

    def precompute(self, x_obs):
        """Cache per-observation derivatives of the networks at `x_obs`.

        Parameters
        ----------
        x_obs : torch.Tensor
            Observed data of shape ``(n, d_x)`` in original units.

        Notes
        -----
        Sets ``x_n`` (standardised data, ``(n, d_x)``), ``J`` (Jacobians of
        ``T_phi``, ``(n, d_theta, d_x)``), ``lap`` (Laplacians of ``T_phi``,
        ``(n, d_theta)``), ``JJ`` (``J J^T``, ``(n, d_theta, d_theta)``) and
        ``h`` (``J grad b_phi``, ``(n, d_theta)``). These caches are not
        used by the other methods of this class, which recompute what they
        need.
        """
        self._x_obs = x_obs
        self.x_n = self._normalise_input(x_obs)                              # (n, d_x)
        J, lap = vmap(jacobian_and_laplacian, in_dims=(0, None))(self.x_n, self.model.T_phi)
        grad_b = vmap(jacobian, in_dims=(0, None))(self.x_n, self.model.b_phi)   # (n, d_x)
        self.J = J                                                           # (n, d_theta, d_x)
        self.lap = lap                                                       # (n, d_theta)
        self.JJ = J @ J.transpose(1, 2)                                      # (n, d_theta, d_theta)
        self.h = (J @ grad_b.unsqueeze(-1)).squeeze(-1)                      # (n, d_theta)


    def _normalise_input(self,x_obs):
        """Standardise data with the model's ``x`` standardiser.

        Parameters
        ----------
        x_obs : torch.Tensor
            Data of shape ``(n, d_x)`` in original units.

        Returns
        -------
        torch.Tensor
            Standardised data, same shape.
        """
        return self.model.standardizer_x(x_obs)

    def _unnormalise(self,mu_n_normalized,Sigma_n_normalized):
        """Map a Gaussian from standardised to original ``theta`` units.

        Parameters
        ----------
        mu_n_normalized : torch.Tensor
            Mean in standardised space, shape ``(d_theta,)``.
        Sigma_n_normalized : torch.Tensor
            Covariance in standardised space, shape ``(d_theta, d_theta)``.

        Returns
        -------
        mu_n : torch.Tensor
            ``mu * s + m``.
        Sigma_n : torch.Tensor
            ``Sigma * outer(s, s)``.
        """

        mu_n = mu_n_normalized * self.s + self.m
        
        Sigma_n = Sigma_n_normalized * torch.outer(self.s, self.s) 

        return mu_n,Sigma_n

    def calibrated_posterior(self, 
                x_obs, 
                beta):
        """Compute the generalised posterior at a given ``beta``.

        Parameters
        ----------
        x_obs : torch.Tensor
            Observed data of shape ``(n, d_x)`` in original units.
        beta : float
            Learning rate (temperature) of the generalised posterior.

        Returns
        -------
        mu_n : torch.Tensor
            Posterior mean in original ``theta`` units, shape
            ``(d_theta,)``.
        Sigma_n : torch.Tensor
            Posterior covariance in original units, shape
            ``(d_theta, d_theta)``.

        Notes
        -----
        The data are standardised, the weights ``w^2`` and ``grad w^2`` are
        computed from the standardised data, the Gaussian posterior is
        solved in standardised space (see the module docstring) and the
        result is mapped back with :meth:`_unnormalise`.
        """

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
        """Posterior precision ``Sigma_0^{-1} + 2 * beta * A`` (standardised space).

        Parameters
        ----------
        x_n : torch.Tensor
            Standardised data of shape ``(n, d_x)``.
        beta : float
            Learning rate of the generalised posterior.
        w2 : torch.Tensor
            Squared weights ``w(x_i)^2``, shape ``(n,)``.

        Returns
        -------
        torch.Tensor
            Precision matrix of shape ``(d_theta, d_theta)``.
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
        """Posterior mean and covariance in standardised space.

        Parameters
        ----------
        x_n : torch.Tensor
            Standardised data of shape ``(n, d_x)``.
        beta : float
            Learning rate of the generalised posterior.
        w2 : torch.Tensor
            Squared weights, shape ``(n,)``.
        grad_w2 : torch.Tensor
            Gradients of the squared weights, shape ``(n, d_x)``.

        Returns
        -------
        mu_n : torch.Tensor
            Posterior mean, shape ``(d_theta,)``, from solving
            ``Sigma_n^{-1} mu_n = Sigma_0^{-1} mu_0 - 2 * beta * B``.
        Sigma_n : torch.Tensor
            Posterior covariance, shape ``(d_theta, d_theta)``.
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
        """Quadratic coefficient ``A = sum_i w2_i J_T(x_i) J_T(x_i)^T``.

        Parameters
        ----------
        x_n : torch.Tensor
            Standardised data of shape ``(n, d_x)``.
        w2 : torch.Tensor
            Squared weights, shape ``(n,)``.

        Returns
        -------
        torch.Tensor
            Symmetric positive semi-definite matrix of shape
            ``(d_theta, d_theta)``.
        """

        # A = sum_i w2_i * J_T(x_i)^T J_T(x_i)   (d_theta, d_theta)
        # single(xi, fn) returns (J_T^T J_T, J_T); only the covariance term is kept.
        batched_cov_terms = vmap(get_cov_term_single, in_dims=(0, None))(
            x_n, self.model.T_phi
        )  # (n, d_theta, d_theta)
        A = (w2.view(-1, 1, 1) * batched_cov_terms).sum(dim=0)  # (d_theta, d_theta)
        return A 

    def _compute_B(self,x_n,w2,grad_w2):
        """Linear coefficient ``B`` of the summed score-matching loss.

        ``B = sum_i [w2_i J_T(x_i) grad b(x_i) + J_T(x_i) grad_w2_i
        + w2_i lap T(x_i)]``; see the module docstring.

        Parameters
        ----------
        x_n : torch.Tensor
            Standardised data of shape ``(n, d_x)``.
        w2 : torch.Tensor
            Squared weights, shape ``(n,)``.
        grad_w2 : torch.Tensor
            Gradients of the squared weights, shape ``(n, d_x)``.

        Returns
        -------
        torch.Tensor
            Vector of shape ``(d_theta,)``.
        """

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
        """Minimiser of the score-matching loss on the observed data.

        Computes ``theta_hat = argmin_theta L_NSM(theta; x_obs)``, the
        M-estimator that general posterior calibration (Syring and Martin,
        2019) treats as the "true" parameter. Because the loss is quadratic,
        ``theta_hat = -A^{-1} B`` in standardised space.

        Parameters
        ----------
        x_obs : torch.Tensor
            Observed data of shape ``(n, d_x)`` in original units; it is
            standardised internally.

        Returns
        -------
        torch.Tensor
            ``theta_hat`` of shape ``(d_theta,)`` in original ``theta``
            units.

        Notes
        -----
        ``A`` is symmetrised and regularised with a ridge
        ``lam * I``, ``lam = 1e-2 * trace(A) / d_theta + 1e-12``, before
        solving. A Cholesky solve is tried first, falling back to least
        squares if the factorisation fails.
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
