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
    jacobian_and_laplacian,
    jacobian
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
        Chooses ``beta`` for :meth:`compute_posterior`.

    Notes
    -----
    The data weights are recomputed from whichever data set is passed to
    :meth:`compute_posterior`, so each bootstrap resample gets its own
    robust centre and scale.
    """

    #: The closed-form posterior makes bootstrap coverage cheap to
    #: re-evaluate, so this posterior supports the array ``initial_beta``
    #: grid search and the ``grid`` / ``bisect`` calibration strategies in
    #: addition to ``gpc`` and ``rm``.
    supports_grid_search = True

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


        #for caching the differentiated values during calibration loops
        self._calib_x_obs= None
        self._calib_x_n = None 
        self._calib_terms = None

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
    def compute_posterior(self, 
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

        #mu_hat, Sigma_inv = estimate_stats(x_n, self.robust)
        
        #w2, grad_w2 = weight_function(self.weight_type, x_n, mu_hat, Sigma_inv)
        mu_out, Sigma_out = self._posterior_from_terms(x_n, self._point_terms(x_n), beta)
        return mu_out, Sigma_out

    def _weights(self, x_n: Tensor) -> tuple[Tensor, Tensor]:
        """Squared weights and their gradients for a standardised data set.

        The weight function is centred and scaled with the statistics of
        `x_n` itself (robust or plain, see ``robust``).

        Parameters
        ----------
        x_n : torch.Tensor
            Standardised data of shape ``(n, d_x)``.

        Returns
        -------
        w2 : torch.Tensor
            ``w(x_i)^2``, shape ``(n,)``.
        grad_w2 : torch.Tensor
            ``grad w(x_i)^2``, shape ``(n, d_x)``.
        """
        mu_hat, Sigma_inv = estimate_stats(x_n, self.robust)
        return weight_function(self.weight_type, x_n, mu_hat, Sigma_inv)

    def _point_terms_single(self,xi, T_phi, b_phi):
        J, L = jacobian_and_laplacian(xi, T_phi)
        grad_b = jacobian(xi, b_phi)
        return J, J @ J.T, J @ grad_b, L          # J, G, c, L

    def _point_terms(self, x_n: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Network terms at every standardised observation.

        Parameters
        ----------
        x_n : torch.Tensor
            Standardised data of shape ``(n, d_x)``.

        Returns
        -------
        J : torch.Tensor
            Jacobians of ``T``, shape ``(n, d_theta, d_x)``.
        G : torch.Tensor
            ``J_i @ J_i.T``, shape ``(n, d_theta, d_theta)``.
        c : torch.Tensor
            ``J_i @ grad b(x_i)``, shape ``(n, d_theta)``.
        L : torch.Tensor
            Laplacians of ``T``, shape ``(n, d_theta)``.
        """
        return vmap(self._point_terms_single, in_dims=(0, None, None))(
            x_n, self.model.T_phi, self.model.b_phi
        )

    @staticmethod
    def _compute_A(w2: Tensor, G: Tensor) -> Tensor:
        """Quadratic coefficient ``A = sum_i w2_i J_T(x_i) J_T(x_i)^T``.

        Parameters
        ----------
        w2 : torch.Tensor
            Squared weights, shape ``(n,)``.
        G : torch.Tensor
            ``J_i @ J_i.T`` from :meth:`_point_terms`, shape
            ``(n, d_theta, d_theta)``.

        Returns
        -------
        torch.Tensor
            Symmetric positive semi-definite matrix of shape
            ``(d_theta, d_theta)``.
        """
        return torch.einsum("i,ijk->jk", w2, G)

    @staticmethod
    def _compute_B(w2: Tensor, grad_w2: Tensor, J: Tensor, c: Tensor, L: Tensor) -> Tensor:
        """Linear coefficient ``B`` of the summed score-matching loss.

        ``B = sum_i [w2_i (J_T(x_i) grad b(x_i) + lap T(x_i))
        + J_T(x_i) grad_w2_i]``; see the module docstring.

        Parameters
        ----------
        w2 : torch.Tensor
            Squared weights, shape ``(n,)``.
        grad_w2 : torch.Tensor
            Gradients of the squared weights, shape ``(n, d_x)``.
        J, c, L : torch.Tensor
            Network terms from :meth:`_point_terms`, shapes
            ``(n, d_theta, d_x)``, ``(n, d_theta)`` and ``(n, d_theta)``.

        Returns
        -------
        torch.Tensor
            Vector of shape ``(d_theta,)``.
        """
        return torch.einsum("i,ij->j", w2, c + L) + torch.einsum("ijk,ik->j", J, grad_w2)


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

        return self._theta_hat_from_terms(x_n, self._point_terms(x_n))

    def _posterior_from_terms(self, x_n, terms, beta):
        """use cached terms to compute posterior"""
        
        J, G, c, L = terms
        w2, grad_w2 = self._weights(x_n)                       # estimate_stats + weight_function
        
        A = torch.einsum("i,ijk->jk", w2, G)
        B = torch.einsum("i,ij->j", w2, c + L) + torch.einsum("ijk,ik->j", J, grad_w2)
        
        Sigma_n_inv = self.prior_cov_inv + 2 * beta * A
        rhs = self.prior_cov_inv @ self.prior_mean_norm - 2 * beta * B
        mu_n = torch.linalg.solve(Sigma_n_inv, rhs)
        
        mu_out, Sigma_out = self._unnormalise(mu_n, sigma_inverse(Sigma_n_inv))
        return mu_out, Sigma_out

    def _theta_hat_from_terms(
        self, x_n: Tensor, terms: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:
        """Score-matching minimiser from precomputed network terms.

        Parameters
        ----------
        x_n : torch.Tensor
            Standardised data of shape ``(n, d_x)``; used for the weights.
        terms : tuple of torch.Tensor
            ``(J, G, c, L)`` for the rows of `x_n`, from
            :meth:`_point_terms`.

        Returns
        -------
        torch.Tensor
            ``theta_hat`` of shape ``(d_theta,)`` in original ``theta``
            units; see :meth:`compute_theta_hat_true`.
        """
        J, G, c, L = terms
        w2, grad_w2 = self._weights(x_n)
        A = self._compute_A(w2, G)
        B = self._compute_B(w2, grad_w2, J, c, L)

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
        ess_threshold : float, default 0.3
            ESS refresh trigger of the general method (unused here).

        Notes
        -----
        Mirrors the pre-refactor behaviour of
        ``calibrate_beta_gpc``, which recomputed ``theta_hat`` on every
        call, by caching it here once instead.

        """
        # prepare_calibration
        x_n = self._normalise_input(x_obs)

        #build jacobian and hessian cache for reuse in the whole 
        terms = self._point_terms(x_n)            # one vmap pass over the observed points

        self._calib_x_obs, self._calib_x_n, self._calib_terms = x_obs, x_n, terms
        
        self.theta_hat = self._theta_hat_from_terms(x_n, terms)

        #self.theta_hat = self.compute_theta_hat_true(x_obs)

    
    @torch.no_grad()
    def estimate_coverage(self,
                          beta: float,
                          alpha: float,
                          B: int,
                          x_obs: torch.Tensor,
                          ess_threshold: float = 0.3) -> tuple[float, float, int]:
        """Empirical bootstrap coverage at ``beta``.

        Draws ``B`` resamples of the rows of `x_obs` with replacement,
        evaluates the closed-form posterior
        :meth:`compute_posterior` at ``beta`` for each, and returns the
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
        ess_threshold : float, default 0.3
            ESS refresh trigger of the general method (unused here).

        Returns
        -------
        cov : float
            Fraction of resamples whose credible ellipsoid covers
            ``theta_hat``.
        ess_mean : float
            Always ``float('inf')`` for the conjugate method: the
            closed-form posterior has no MCMC pool, so there is no
            effective sample size to track.
        pool_size : int
            Always ``0``: no MCMC pool exists.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare_calibration`.
        """
        if self.theta_hat is None or self._calib_terms is None:
            raise RuntimeError(
                "ConjugatePosterior.estimate_coverage requires "
                "prepare_calibration to be called first"
            )

        if x_obs is not self._calib_x_obs and not (
                x_obs.shape == self._calib_x_obs.shape
                and torch.equal(x_obs, self._calib_x_obs)
            ):
                raise ValueError(
                    "x_obs differs from the data passed to prepare_calibration; "
                    "call prepare_calibration again for new data"
                )

        chi2_critical_value = chi2.ppf(1.0 - alpha, df=self.prior_mean.shape[0])

        covered = 0
        for _ in range(B):
            indices = torch.randint(0, x_obs.shape[0], (x_obs.shape[0],), device=x_obs.device)
            x_boot = x_obs[indices]
            terms_boot = tuple(t[indices] for t in self._calib_terms)
            #mu_boot, Sigma_boot = self.compute_posterior(x_boot, beta)
            #replace with computation from cached terms
            mu_boot, Sigma_boot = self._posterior_from_terms(self._calib_x_n[indices], terms_boot, beta)
            
            if mahalanobis_sq(self.theta_hat, mu_boot, Sigma_boot).item() \
                    <= chi2_critical_value:
                covered += 1
        
        cov = covered / B
        
        # Closed-form posterior: no MCMC pool, so ESS is infinite and
        # pool size is zero.  The caller's refresh check
        # ``ess_mean < ess_threshold * pool_size`` is then
        # ``inf < 0`` which is always False, so no refresh is triggered.
        return cov, float("inf"), 0
