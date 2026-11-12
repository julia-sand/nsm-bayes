"""Unified beta calibration for the NSM-Bayes generalised posterior.

General posterior calibration (GPC; Syring and Martin, 2019) picks the
learning rate ``beta`` so that the ``1 - alpha`` credible region of the
generalised posterior covers the NSM point estimate ``theta_hat`` with
frequency ``1 - alpha`` across bootstrap resamples of the observed data.

Both methods drive the *same* stochastic-approximation loop over
``log beta`` (see
:func:`nsm_bayes.inference.beta_calibration_loop.run_beta_calibration_loop`);
everything method-specific lives in the posterior classes, which expose a
flat, shared interface for the two calibration steps:

* :meth:`~nsm_bayes.models.general.posterior.GeneralPosterior.prepare_calibration`
  / :meth:`~nsm_bayes.models.conjugate.conj_posterior.ConjugatePosterior.prepare_calibration`
  compute and cache ``theta_hat`` (and, for the general method, draw and
  cache the base MCMC pool at ``beta_init`` and its per-observation
  losses);
* :meth:`~nsm_bayes.models.general.posterior.GeneralPosterior.estimate_coverage`
  / :meth:`~nsm_bayes.models.conjugate.conj_posterior.ConjugatePosterior.estimate_coverage`
  estimate the bootstrap coverage at one value of ``beta`` -- by
  importance-reweighting the cached pool (with an ESS-gated refresh) for
  the general method, and by resampling the closed-form Gaussian
  posterior for the conjugate one.

Each implementation ignores the arguments its method does not need; the
flat signature keeps this module free of per-method dispatch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from nsm_bayes.inference.beta_calibration_loop import run_beta_calibration_loop

_log = logging.getLogger(__name__)

__all__ = ["calibrate_beta"]


def calibrate_beta(
    posterior,
    beta_init: float,
    alpha: float = 0.05,
    B: int = 200,
    T: int = 10,
    x_obs: torch.Tensor | None = None,
    step_schedule: Callable[[int], float] = lambda t: 10.0 / (t + 10),
    max_log_step: float = 0.25,
    beta_min: float = 0.01,
    ess_threshold: float = 0.3,
) -> tuple[float, dict]:
    """Calibrate ``beta`` for a generalised posterior by bootstrap coverage matching.

    Works for both the normalising-flow posterior
    (:class:`~nsm_bayes.models.general.posterior.GeneralPosterior`) and the
    closed-form conjugate posterior
    (:class:`~nsm_bayes.models.conjugate.conj_posterior.ConjugatePosterior`);
    the method-specific work (point estimate, bootstrap coverage, and for
    the general method the ESS-gated sample refresh) is delegated to the
    ``prepare_calibration`` / ``estimate_coverage`` methods of
    ``posterior``.

    The base MCMC pool is *not* passed in: ``prepare_calibration`` draws
    it at `beta_init` when the method needs one (the general method) and
    caches it on the posterior; :meth:`~nsm_bayes.models.general.posterior.GeneralPosterior.theta_samples_base`
    returns it on each call. :class:`ConjugatePosterior` is closed-form,
    so its ``theta_samples_base`` is always ``None``.

    Parameters
    ----------
    posterior : GeneralPosterior or ConjugatePosterior
        Posterior whose ``prepare_calibration``, ``estimate_coverage`` and
        ``theta_samples_base`` methods implement the coverage matching for
        its method.
    beta_init : float
        Starting value of ``beta``; for the general method it is also the
        value at which the base MCMC pool is drawn.
    alpha : float, default 0.05
        Miscoverage level; ``1 - alpha`` is the target coverage.
    B : int, default 200
        Number of bootstrap resamples per iteration.
    T : int, default 10
        Number of stochastic-approximation iterations.
    x_obs : torch.Tensor, optional
        Observed data of shape ``(n, d_x)`` in original units. Required
        for :class:`ConjugatePosterior`; must be ``None`` (or arbitrary)
        for :class:`GeneralPosterior`, which holds its own copy.
    step_schedule : callable, default ``lambda t: 10.0 / (t + 10)``
        Step-size schedule ``k_t`` for the update of ``log(beta)``; called
        with the zero-based iteration index.
    max_log_step : float, default 0.25
        Absolute cap on the per-iteration change in ``log(beta)``.
    beta_min : float, default 0.01
        Lower bound on ``beta``, to stop it collapsing towards zero.
    ess_threshold : float, default 0.3
        General method only: if the mean effective sample size of the
        importance weights falls below ``ess_threshold * M``, the sample
        pool is redrawn inside ``estimate_coverage``.

    Returns
    -------
    beta : float
        The calibrated value of ``beta`` after the final iteration.
    history : dict
        ``{"betas": list[float], "coverages": list[float]}``: the value of
        ``beta`` after each update and the empirical coverage that drove it.

    Notes
    -----
    References
    ----------
    .. [1] N. Syring and R. Martin, "Calibrating general posterior credible
       regions", *Biometrika*, 106(2), 479-486, 2019.
    """
    kwargs = dict(
        alpha=alpha,
        B=B,
        x_obs=x_obs,
        ess_threshold=ess_threshold,
    )

    # One-time setup: theta_hat (and, for the general method, the base pool
    # -- drawn here at beta_init -- and its cached losses).
    posterior.prepare_calibration(beta_init, **kwargs)

    beta, history = run_beta_calibration_loop(
        coverage_fn=lambda beta: posterior.estimate_coverage(beta, **kwargs),
        n_iterations=T,
        beta_init=beta_init,
        beta_min=beta_min,
        target_coverage=1.0 - alpha,
        step_schedule=step_schedule,
        max_log_step=max_log_step,
    )
    return beta, history
