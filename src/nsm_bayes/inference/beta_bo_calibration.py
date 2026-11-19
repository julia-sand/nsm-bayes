"""Bayesian-optimisation beta calibration via a GP on a coverage grid.

Companion to :mod:`nsm_bayes.inference.beta_update_rules`.  Instead of
iterating a stochastic-approximation step, :func:`optimize_beta_bo` fits a
Gaussian process to pre-computed ``(beta, coverage)`` pairs and returns the
``beta`` where the GP mean crosses the target coverage.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import gpytorch

__all__ = ["optimize_beta_bo"]


class _CoverageGP(gpytorch.models.ExactGP):
    """Zero-mean exact GP with a unit-amplitude Matern-5/2 kernel.

    The lengthscale is the only free hyperparameter and is optimised in
    log space (``exp`` transform).

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs ``log beta``, shape ``(n, 1)``.
    train_y : torch.Tensor
        Standardised training coverages, shape ``(n,)``.
    likelihood : gpytorch.likelihoods.Likelihood
        Observation likelihood.
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=2.5,
            lengthscale_constraint=gpytorch.constraints.Positive(
                transform=torch.exp, inv_transform=torch.log
            ),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Prior distribution of the latent function at ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Inputs of shape ``(m, 1)``.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Prior over the ``m`` function values.
        """
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _gp_posterior_mean(x: np.ndarray, y: np.ndarray,
                       eval_x: np.ndarray) -> np.ndarray:
    """Fit the coverage GP and return its posterior mean at ``eval_x``.

    The coverages are standardised (zero mean, unit standard deviation;
    a zero standard deviation is replaced by 1), the lengthscale is fitted
    by maximising the exact marginal likelihood with L-BFGS from
    ``_INIT_LENGTHSCALE``, and the mean is mapped back to coverage units.

    Parameters
    ----------
    x : numpy.ndarray
        Training inputs ``log beta``, shape ``(n,)``.
    y : numpy.ndarray
        Training coverages, shape ``(n,)``.
    eval_x : numpy.ndarray
        Evaluation inputs ``log beta``, shape ``(m,)``.

    Returns
    -------
    numpy.ndarray
        GP posterior mean of the coverage at ``eval_x``, shape ``(m,)``.
    """
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std < 10 * np.finfo(np.float64).eps:
        y_std = 1.0

    train_x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
    train_y = torch.as_tensor((y - y_mean) / y_std, dtype=torch.float64)
    test_x = torch.as_tensor(eval_x, dtype=torch.float64).reshape(-1, 1)

    # gpytorch raises fixed noise below min_fixed_noise (1e-6 in double);
    # lower the floor so the GP keeps its intended 1e-10 noise.
    with gpytorch.settings.min_fixed_noise(double_value= 1e-10):
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.full_like(train_y,  1e-10),
            learn_additional_noise=False,
        )
        model = _CoverageGP(train_x, train_y, likelihood).double()
        #model.covar_module.lengthscale = _INIT_LENGTHSCALE

        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=1.0,
            line_search_fn="strong_wolfe",
        )

        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()

        optimizer.step()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mean = model(test_x).mean

    return mean.numpy() * y_std + y_mean


def optimize_beta_bo(
    betas: list[float] | np.ndarray,
    coverages: list[float] | np.ndarray,
    *,
    target_coverage: float,
    beta_min: float = 0.01,
) -> float:
    """Return the beta where a GP fit of coverage on log beta crosses the target.

    Fits a GP to the ``(log beta, coverage)`` pairs and returns the
    ``beta`` where its posterior mean crosses *target_coverage*.

    Parameters
    ----------
    betas : array-like of float
        Positive beta values at which coverage was evaluated.
    coverages : array-like of float
        Coverage values corresponding to each beta (same length as *betas*).
    target_coverage : float
        The desired coverage level (e.g. ``1 - alpha = 0.95``).
    beta_min : float
        Floor applied to the returned beta.

    Returns
    -------
    float
        The beta whose GP-mean coverage is closest to *target_coverage*,
        floored at *beta_min*.

    Raises
    ------
    ValueError
        If *betas* and *coverages* are not 1-D arrays of the same length,
        have fewer than two entries, or *betas* contains a non-positive
        value.

    Notes
    -----
    The GP is a zero-mean GPyTorch ``ExactGP`` on the standardised
    coverages with a unit-amplitude Matern-5/2 kernel and fixed noise
    ``1e-10``.  Its lengthscale is fitted by maximising the exact marginal
    likelihood with L-BFGS, starting from 1.0.
    """
    betas = np.asarray(betas, dtype=np.float64)
    coverages = np.asarray(coverages, dtype=np.float64)

    if betas.ndim != 1 or coverages.ndim != 1:
        raise ValueError("betas and coverages must be 1-D arrays")
    if betas.shape[0] != coverages.shape[0]:
        raise ValueError("betas and coverages must have the same length")
    if betas.shape[0] < 2:
        raise ValueError("need at least two (beta, coverage) pairs")
    if np.any(betas <= 0):
        raise ValueError("all beta values must be positive")

    x = np.log(betas)

    # Dense evaluation over the log-beta range to find the crossing.
    x_min, x_max = float(x.min()), float(x.max())
    # Pad the range slightly so the crossing isn't clipped at the boundary.
    pad = 0.1 * (x_max - x_min) if x_max > x_min else 0.1
    eval_x = np.linspace(x_min - pad, x_max + pad, 2001)
    mean = _gp_posterior_mean(x, coverages, eval_x)

    # Find where the GP mean crosses the target.
    diff = mean - target_coverage
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) == 0:
        # No crossing: return the beta whose GP-mean coverage is closest.
        best_idx = int(np.argmin(np.abs(diff)))
        best_log_beta = eval_x[best_idx]
    else:
        # Use the first crossing; linear interpolation in (log beta, coverage).
        i = int(sign_changes[0])
        x0, x1 = eval_x[i], eval_x[i + 1]
        y0, y1 = mean[i], mean[i + 1]
        frac = (target_coverage - y0) / (y1 - y0)
        best_log_beta = x0 + frac * (x1 - x0)

    beta = math.exp(best_log_beta)
    return max(beta, beta_min)
