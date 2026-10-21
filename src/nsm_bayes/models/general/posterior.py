r"""Generalised (score-matching) posterior for the normalising-flow method.

The posterior is

.. math::

    p_\beta(\theta \mid x_{1:n}) \propto \pi(\theta)
        \exp\{-\beta\, n\, L_{NSM}(\theta; x_{1:n})\},

where :math:`L_{NSM}` is the mean over observations of the weighted
score-matching loss of the trained flow :math:`q(x \mid \theta)`,

.. math::

    \ell(\theta; x) = w^2(x) \lVert s_\theta(x) \rVert^2
        + 2\, \nabla w^2(x)^\top s_\theta(x)
        + 2\, w^2(x)\, \Delta_x \log q(x \mid \theta),
    \qquad s_\theta(x) = \nabla_x \log q(x \mid \theta).

Down-weighting outlying observations through :math:`w` makes the posterior
robust to contamination; :math:`\beta` is chosen by
:func:`~nsm_bayes.inference.general.calibrate.calibrate_beta`.
"""
from __future__ import annotations

import functools
import logging

import numpy as np
import torch
from torch import Tensor
from torch.func import vmap

from nsm_bayes.inference.general.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.models.general.mcmc_settings import MCMCSettings
from nsm_bayes.utils.linalg import jacobian_and_laplacian
from nsm_bayes.utils.statistics import estimate_stats
from nsm_bayes.utils.weights import weight_function

_log = logging.getLogger(__name__)


class GeneralPosterior:
    """Generalised score-matching posterior built on a trained flow.

    Parameters
    ----------
    x_obs : torch.Tensor
        Observed data of shape ``(n, d_x)``, in the same (unstandardised)
        units the flow was trained on.
    flow_model : GeneralModel
        Model with a trained :attr:`~GeneralModel.estimator`.
    prior : torch.distributions.Distribution
        Prior over ``theta``.
    mcmc : MCMCSettings, optional
        Slice-sampler settings for :meth:`sample`. Defaults to
        ``MCMCSettings()``.
    robust : bool, default True
        Centre and scale the weights with the median and MCD covariance of
        `x_obs` (true) or the sample mean and covariance (false).
    weight_type : {"imq", "huber", "none"}, default "imq"
        Weight function; see :func:`~nsm_bayes.utils.weights.weight_function`.
    c : float, default 1.0
        Scale parameter of the weight function.

    Attributes
    ----------
    n_obs : int
        Number of observations ``n``.
    mu_hat, Sigma_inv : torch.Tensor
        Location and inverse scatter of `x_obs` used by the weights.
    w2, grad_w2 : torch.Tensor
        Squared weights ``(n,)`` and their gradients ``(n, d_x)``, computed
        once from `x_obs`.

    Notes
    -----
    The weights are fitted once on the raw `x_obs`, since the flow's score
    is taken with respect to raw ``x``.
    """

    def __init__(self, x_obs,
                    flow_model,  
                    prior,
                    mcmc: MCMCSettings | None = None,
                    robust: bool = True,
                    weight_type: str = "imq",
                    c: float = 1.0):

        self.flow = flow_model
        self.x_obs = x_obs
        self.prior = prior
        self.mcmc = mcmc if mcmc is not None else MCMCSettings()
        self.robust, self.weight_type, self.c = robust, weight_type, c
        self.n_obs = x_obs.shape[0]

        self.mu_hat, self.Sigma_inv = estimate_stats(x_obs, robust)
        self.w2, self.grad_w2 = weight_function(
            weight_type, self.x_obs, self.mu_hat, self.Sigma_inv, c
        )

    def _per_x_sm_loss(self, xi, w2_i, grad_w2_i, theta):
        """Weighted SM loss for one observation `xi` at one `theta`.

        Parameters
        ----------
        xi : torch.Tensor
            A single observation of shape ``(d_x,)``.
        w2_i : torch.Tensor
            Squared weight at `xi`, a scalar.
        grad_w2_i : torch.Tensor
            Weight gradient at `xi`, shape ``(d_x,)``.
        theta : torch.Tensor
            Parameter vector of shape ``(d_theta,)``.

        Returns
        -------
        torch.Tensor
            Scalar score-matching loss contribution for this observation.
        """

        log_q_lambda = lambda xi : self.flow.estimator.log_prob(xi,theta).reshape(1) #self.log_q(xi,theta) 
        score, lap = jacobian_and_laplacian(xi, log_q_lambda)  # (d_x,), scalar
        return w2_i * (score @ score) + 2.0 * (grad_w2_i @ score) + 2.0 * w2_i * lap

    
    def _losses_one_theta(self, theta):
        """Per-observation losses at one parameter value.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector of shape ``(d_theta,)``.

        Returns
        -------
        torch.Tensor
            Losses of shape ``(n,)``; differentiable with respect to
            `theta`.
        """
        return vmap(self._per_x_sm_loss, in_dims=(0, 0, 0, None))(
            self.x_obs, self.w2, self.grad_w2, theta
        )

    def sm_loss(self, theta):
        """Mean score-matching loss ``L_NSM(theta)`` over the observations.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector of shape ``(d_theta,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss; differentiable with respect to `theta`.
        """
        return self._losses_one_theta(theta).mean()

    @torch.no_grad()
    def per_x_losses(self, theta_samples):
        """Per-observation losses for a batch of parameter samples.

        Parameters
        ----------
        theta_samples : torch.Tensor
            Parameter samples of shape ``(M, d_theta)``.

        Returns
        -------
        ell : torch.Tensor
            Losses ``ell[m, i] = l(theta_m; x_i)``, shape ``(M, n)``.
        L_orig : torch.Tensor
            Mean loss for each sample, ``ell.mean(dim=1)``, shape ``(M,)``.

        Notes
        -----
        Evaluated without gradients, vectorised over samples and
        observations with nested ``vmap``. The outputs form the cache used
        by :func:`~nsm_bayes.inference.bootstrap.reweight_for_bootstrap_beta`.
        """
        ell = vmap(lambda th: self._losses_one_theta(th))(theta_samples)
        return ell, ell.mean(dim=1)

    #@torch.no_grad()
    def posterior_log_prob(self, theta_np: np.ndarray, beta: float) -> float:
        """Unnormalised log-density of the generalised posterior.

        Parameters
        ----------
        theta_np : numpy.ndarray
            Parameter vector of shape ``(d_theta,)``.
        beta : float
            Learning rate of the generalised posterior.

        Returns
        -------
        float
            ``log prior(theta) - beta * n * L_NSM(theta)``, or ``-inf`` where
            the prior log-density is ``-inf``.

        Notes
        -----
        Accepts and returns NumPy/Python types so it can be passed directly
        to the slice sampler.
        """
        theta = torch.as_tensor(theta_np, dtype=self.x_obs.dtype, device=self.x_obs.device)
        lp = self.prior.log_prob(theta).sum()
        if torch.isneginf(lp):
            return -np.inf
        return float(lp - beta * self.n_obs * self.sm_loss(theta))

    def sample(self, beta: float) -> Tensor:
        """Draw samples from the generalised posterior.

        Parameters
        ----------
        beta : float
            Learning rate of the generalised posterior.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(S, d_theta)`` with the dtype and device of
            `x_obs`; see
            :func:`~nsm_bayes.inference.general.slice_sampler.run_multivariate_slice_sampler_tuned`
            for ``S``.
        """
        samples = run_multivariate_slice_sampler_tuned(
                            log_posterior_fn=functools.partial(self.posterior_log_prob, beta=beta),
                            prior=self.prior,
                            mcmcsettings=self.mcmc
                        )
        return torch.from_numpy(samples).to(dtype=self.x_obs.dtype, device=self.x_obs.device)

    def compute_theta_hat_true(self, theta_init: Tensor, num_steps: int = 200, lr: float = 1e-2) -> Tensor:
        """Minimise the score-matching loss on the observed data.

        Finds ``theta_hat = argmin_theta L_NSM(theta; x_obs)`` with Adam;
        used as the reference parameter in beta calibration.

        Parameters
        ----------
        theta_init : torch.Tensor
            Starting point of shape ``(d_theta,)``; not modified.
        num_steps : int, default 200
            Number of Adam steps.
        lr : float, default 1e-2
            Adam learning rate.

        Returns
        -------
        torch.Tensor
            The final iterate, detached, shape ``(d_theta,)``. No
            convergence check is made.
        """
        theta = theta_init.detach().clone().to(dtype=self.x_obs.dtype, device=self.x_obs.device)
        theta.requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self.sm_loss(theta)
            loss.backward()
            optimizer.step()
            if step % 20 == 0 or step == num_steps - 1:
                _log.debug("nsm_minimiser step %d | L_NSM = %.4f", step, loss.item())
        return theta.detach()
