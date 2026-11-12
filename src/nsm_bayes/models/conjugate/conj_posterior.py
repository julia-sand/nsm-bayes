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
from scipy.stats import chi2
from torch import Tensor
from torch.func import vmap

from nsm_bayes.models.conjugate.conj_model import EnergyBasedModel
from nsm_bayes.utils.linalg import (
    get_cov_term_single,
    get_non_div_term_single,
    sigma_inverse,
    get_div_term_single,
)

from nsm_bayes.utils.weights import weight_function
from nsm_bayes.utils.statistics import estimate_stats, mahalanobis_sq

class ConjugatePosterior():
    """Gaussian generalised posterior for a fitted :class:`EnergyBasedModel`.

    Parameters
    ----------
    model : EnergyBasedModel
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
    nsm_bayes.inference.calibrate_beta.calibrate_beta :
        Chooses ``beta`` for :meth:`calibrated_posterior`.

    Notes
    -----
    The data weights are recomputed from whichever data set is passed to
    :meth:`calibrated_posterior`, so each bootstrap resample gets its own
    robust centre and scale.
    """

    def __init__(self,
                model: EnergyBasedModel,
                prior_mean: torch.Tensor,
                prior_cov: torch.Tensor,
                robust: bool = True,
                weight_type: str = "imq") -> None:

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

        # Point estimate used by the beta-calibration steps
        # (see prepare_calibration). None until prepare_calibration is called.
        self.theta_hat: Tensor | None = None

    def _normalise_input(self,x_obs: torch.Tensor) -> torch.Tensor:
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

    def _unnormalise(self,mu_n_normalized: torch.Tensor,Sigma_n_normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    @torch.no_grad()
    def calibrated_posterior(self, 
                x_obs: torch.Tensor, 
                beta: float) -> tuple[torch.Tensor, torch.Tensor]:
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
    def _compute_posterior_precision_conjugate(self, x_n: torch.Tensor, beta: float, w2: torch.Tensor) -> torch.Tensor:
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

    def _compute_posterior_conjugate(self, x_n: torch.Tensor, beta: float, w2: torch.Tensor, grad_w2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _compute_A(self,x_n: torch.Tensor,w2: torch.Tensor) -> torch.Tensor:
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

    def _compute_B(self,x_n: torch.Tensor,w2: torch.Tensor,grad_w2: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
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

    # ------------------------------------------------------------------ #
    # Beta calibration: the two steps driven by                           #
    # :func:`nsm_bayes.inference.calibrate_beta.calibrate_beta`.          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prepare_calibration(self,
                            beta: float,
                            alpha: float,
                            B: int,
                            x_obs: torch.Tensor,
                            theta_samples_base: torch.Tensor | None = None,
                            ess_threshold: float = 0.3) -> None:
        """Set up the state used by :meth:`estimate_coverage`.

        Computes the NSM point estimate ``theta_hat`` -- the closed-form
        minimiser of the score-matching loss on `x_obs` -- that the
        bootstrap credible regions must cover.

        Parameters
        ----------
        beta : float
            Starting value of ``beta`` (unused by the conjugate method;
            kept so both posterior classes share one flat signature).
        alpha : float
            Miscoverage level (unused here; the target coverage is passed
            to :meth:`estimate_coverage` through ``alpha``).
        B : int
            Number of bootstrap resamples per iteration (unused here; it
            is passed to :meth:`estimate_coverage` on every call).
        x_obs : torch.Tensor
            Observed data of shape ``(n, d_x)`` in original units.
        theta_samples_base : torch.Tensor, optional
            Base MCMC pool (unused by the conjugate method; the posterior
            is closed-form).
        ess_threshold : float, default 0.3
            ESS refresh trigger of the general method (unused here).

        Notes
        -----
        Mirrors the pre-refactor behaviour of
        ``calibrate_beta_gpc``, which recomputed ``theta_hat`` on every
        call, by caching it here once instead.
        """
        self.theta_hat = self.compute_theta_hat_true(x_obs)

    def theta_samples_base(self) -> torch.Tensor | None:
        """Base MCMC pool for importance reweighting.

        Always ``None`` for the conjugate method: the posterior is
        closed-form and coverage is estimated by resampling the Gaussian
        directly, so no base pool is drawn or cached. Present so both
        posterior classes share one flat interface.
        """
        return None

    @torch.no_grad()
    def estimate_coverage(self,
                          beta: float,
                          alpha: float,
                          B: int,
                          x_obs: torch.Tensor,
                          theta_samples_base: torch.Tensor | None = None,
                          ess_threshold: float = 0.3) -> float:
        """Empirical bootstrap coverage at ``beta``.

        Draws ``B`` resamples of the rows of `x_obs` with replacement,
        evaluates the closed-form posterior
        :meth:`calibrated_posterior` at ``beta`` for each, and returns the
        fraction of resamples whose ``1 - alpha`` chi-squared ellipsoid
        contains the cached ``theta_hat``.

        Parameters
        ----------
        beta : float
            Current value of ``beta`` to evaluate the coverage at.
        alpha : float
            Miscoverage level; the credible ellipsoid has nominal level
            ``1 - alpha``.
        B : int
            Number of bootstrap resamples.
        x_obs : torch.Tensor
            Observed data of shape ``(n, d_x)`` in original units.
        theta_samples_base : torch.Tensor, optional
            Base MCMC pool (unused by the conjugate method).
        ess_threshold : float, default 0.3
            ESS refresh trigger of the general method (unused here).

        Returns
        -------
        float
            Fraction of resamples whose credible ellipsoid covers
            ``theta_hat``.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare_calibration`.
        """
        if self.theta_hat is None:
            raise RuntimeError(
                "ConjugatePosterior.estimate_coverage requires "
                "prepare_calibration to be called first"
            )

        d_theta = self.prior_mean.shape[0]
        n_obs = x_obs.shape[0]
        device = x_obs.device
        chi2_critical_value = chi2.ppf(1.0 - alpha, df=d_theta)

        covered = 0
        for _ in range(B):
            indices = torch.randint(0, n_obs, (n_obs,), device=device)
            x_boot = x_obs[indices]
            mu_boot, Sigma_boot = self.calibrated_posterior(x_boot, beta)
            if mahalanobis_sq(self.theta_hat, mu_boot, Sigma_boot).item() \
                    <= chi2_critical_value:
                covered += 1
        return covered / B
