"""Unified robust weighting for the neural score-matching Bayes method.

The general method and the conjugate method both build a robust
generalised posterior by weighting each observation ``x_i`` with a
function of its distance to a robust centre ``mu_hat`` under a robust
covariance ``Sigma_hat``.  Historically these ingredients (``mu_hat``,
``Sigma_hat``/``Sigma_inv`` and the scale ``c``) were recomputed and
shuffled around as four loose values in several places (posterior
building, ``beta`` calibration and the score-matching loss), which made
the weighting statistically inconsistent between the loss that is
*optimised* and the posterior that is *sampled*, and duplicated the
weight-function plumbing in every call signature.

This module centralises that.  :class:`Weighting` is a frozen,
space-agnostic container for a weight function together with its fitted
parameters; it knows how to evaluate the squared weight ``w(x)^2`` and
its gradient ``grad_w2`` for any coordinates ``x`` supplied to it.  The
*space* in which the weights are fitted is a caller concern:

* the **general** (normalising-flow) method fits the weights on the raw
  observed data ``x_obs`` (the flow's score is taken with respect to
  raw ``x``);
* the **conjugate** method fits them on the standardised residuals
  ``z = (x - mu) / sigma`` (the energy is taken with respect to
  ``z``).

Both go through the same :meth:`Weighting.fit` entry point, differing
only in the ``standardise`` flag, so the fitting logic lives in exactly
one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nsm_bayes.utils.statistics import (
    robust_mean_cov,
    sample_mean_and_covariance,
)
from nsm_bayes.utils.weightfuns import (
    cauchy_weight,
    gpc_weight,
    huber_weight,
    imq_weight,
)

__all__ = ["Weighting", "FittedWeighting"]


@dataclass(frozen=True)
class Weighting:
    """A robust weighting function together with its fitted parameters.

    A ``Weighting`` is *space-agnostic*: it stores which weight function
    to use and the fitted centre ``mu_hat``, inverse robust covariance
    ``Sigma_inv`` and scale ``c``, but it does **not** store the data.
    It evaluates the weight for whatever coordinates it is given, so the
    same object can be built on standardised residuals (conjugate method)
    or on raw observations (general method) and queried in that space.

    The object is immutable once fitted, and its parameters are plain
    tensors that can be carried across devices (see :meth:`to`).

    Parameters
    ----------
    weight_type : str
        Name of the weight function. One of :data:`WEIGHT_NAMES`
        (``"imq"``, ``"huber"`` or ``"none"``).
    mu_hat : torch.Tensor
        Fitted centre of shape ``(d_x,)`` in the weighting space.
    Sigma_inv : torch.Tensor
        Fitted inverse covariance of shape ``(d_x, d_x)`` in the
        weighting space.
    c : float, default 1.0
        Scale parameter passed to the weight function.
    zeta : float, default 1.0
        Sharpness parameter (only used by the inverse-multiquadric
        weight).
    """

    weight_type: str
    mu_hat: torch.Tensor
    Sigma_inv: torch.Tensor
    c: float = 1.0
    zeta: float = 1.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def none(cls, d_x: int, *, dtype=None, device=None) -> "Weighting":
        """Return a constant (unweighted) weighting in ``d_x`` dimensions.

        Parameters
        ----------
        d_x : int
            Dimension of the coordinate space the weights act on.
        dtype : torch.dtype, optional
            Dtype for the (unused) parameter tensors.
        device : torch.device, optional
            Device for the (unused) parameter tensors.

        Returns
        -------
        Weighting
            A weighting with ``weight_type="none"``.
        """
        eye = torch.eye(d_x, dtype=dtype, device=device)
        mu = torch.zeros(d_x, dtype=dtype, device=device)
        return cls(weight_type="none", mu_hat=mu, Sigma_inv=eye, c=1.0)

    @classmethod
    def fit(
        cls,
        x_obs: torch.Tensor,
        weight_type: str = "imq",
        *,
        robust: bool = True,
        standardise: bool = False,
        c: float = 1.0,
        zeta: float = 1.0,
    ) -> "FittedWeighting":
        """Fit a weighting on the supplied observations.

        Parameters
        ----------
        x_obs : torch.Tensor
            Observations of shape ``(n, d_x)``.
        weight_type : str, default "imq"
            Name of the weight function to use.
        robust : bool, default True
            If ``True``, the centre and covariance are robust estimates
            (``sample_robust_mean_cov``); if ``False``, the ordinary
            sample mean and covariance are used.
        standardise : bool, default False
            If ``True``, the observations are standardised
            (``(x - mu) / sigma``) *before* the robust centre and
            covariance are fitted, and the returned weights are to be
            evaluated on standardised coordinates. If ``False``, the
            weights are fitted on the raw observations and are to be
            evaluated on raw coordinates. This is the single switch that
            distinguishes the conjugate (standardised) and general (raw)
            methods.
        c : float, default 1.0
            Scale parameter.
        zeta : float, default 1.0
            Sharpness parameter (IMQ only).

        Returns
        -------
        FittedWeighting
            The fitted weighting plus the (optionally standardised)
            observations it was fit on.

        Raises
        ------
        ValueError
            If ``x_obs`` is not 2-D.
        """
        if x_obs.ndim != 2:
            raise ValueError(f"x_obs must be (n, d_x); got {tuple(x_obs.shape)}")

        if standardise:
            mu_s = x_obs.mean(dim=0)
            # Standard deviations are the sqrt of the per-column variance.
            sigma_s = (x_obs.var(dim=0, unbiased=False) + 1e-12).sqrt()
            x_w = (x_obs - mu_s) / sigma_s
        else:
            x_w = x_obs

        if robust:
            mu_hat, Sigma_hat = robust_mean_cov(x_w)
        else:
            mu_hat, Sigma_hat = sample_mean_and_covariance(x_w)

        d_x = x_w.shape[-1]
        # Symmetrise the robust covariance (the M-estimator is not
        # guaranteed to return an exactly symmetric matrix) before
        # inverting, with a small ridge for numerical safety.
        Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)
        ridge = 1e-6 * torch.eye(d_x, dtype=x_w.dtype, device=x_w.device)
        Sigma_inv = torch.linalg.inv(Sigma_hat + ridge)

        return FittedWeighting(
            weight_type=weight_type,
            mu_hat=mu_hat,
            Sigma_inv=Sigma_inv,
            c=float(c),
            zeta=float(zeta),
            x_obs=x_w,
            standardised=standardise,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(w2, grad_w2)`` evaluated at coordinates ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape ``(d_x,)`` or ``(n, d_x)`` **in the same
            space the weights were fitted in** (see :meth:`fit`).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The squared weight ``w2`` and its gradient ``grad_w2``, with
            shapes matching the convention of ``x`` (see the individual
            weight functions).

        Raises
        ------
        ValueError
            If ``weight_type`` is not recognised.
        """
        wt = self.weight_type
        if wt == "imq":
            return imq_weight(x, self.mu_hat, self.Sigma_inv, self.c, self.zeta)
        if wt == "huber":
            return huber_weight(x, self.mu_hat, self.Sigma_inv, self.c, self.zeta)
        if wt == "cauchy":
            return cauchy_weight(x, self.mu_hat, self.Sigma_inv, self.c, self.zeta)
        if wt == "gpc":
            return gpc_weight(x, self.mu_hat, self.Sigma_inv, self.c, self.zeta)
        raise ValueError(f"Unknown weight_type: {wt!r}")

    def to(self, device: torch.device | None = None, dtype=None) -> "Weighting":
        """Return a copy moved to ``device`` / cast to ``dtype``.

        Parameters
        ----------
        device : torch.device, optional
            Target device.
        dtype : torch.dtype, optional
            Target dtype for the parameter tensors.

        Returns
        -------
        Weighting
            A new ``Weighting`` with the parameter tensors moved/cast.
        """
        mu = self.mu_hat
        sig = self.Sigma_inv
        if device is not None:
            mu = mu.to(device=device)
            sig = sig.to(device=device)
        if dtype is not None:
            mu = mu.to(dtype=dtype)
            sig = sig.to(dtype=dtype)
        return Weighting(
            weight_type=self.weight_type,
            mu_hat=mu,
            Sigma_inv=sig,
            c=self.c,
            zeta=self.zeta,
        )

class FittedWeighting(Weighting):
    """A :class:`Weighting` that also stores the data it was fit on.

    Attributes
    ----------
    x_obs : torch.Tensor
        The observations the weighting was fit on, of shape
        ``(n, d_x)``, **in the space the weights evaluate in** (i.e.
        standardised if ``standardised`` is true, raw otherwise). Callers
        that need the observations for their own loss terms (the GPC
        residual, the density evaluation point, ...) use this directly.
    standardised : bool
        Whether ``x_obs`` (and therefore every coordinate passed to the
        callable) is standardised. Mirrors the ``standardise`` flag used
        in :meth:`Weighting.fit`.
    """

    x_obs: torch.Tensor
    standardised: bool = False

    def to(self, device: torch.device | None = None, dtype=None) -> "FittedWeighting":
        """Return a copy (with the fitted data) moved/cast.

        Parameters
        ----------
        device : torch.device, optional
            Target device.
        dtype : torch.dtype, optional
            Target dtype.

        Returns
        -------
        FittedWeighting
            A new ``FittedWeighting`` with all tensors moved/cast.
        """
        base = super().to(device=device, dtype=dtype)
        x_obs = self.x_obs
        if device is not None:
            x_obs = x_obs.to(device=device)
        if dtype is not None:
            x_obs = x_obs.to(dtype=dtype)
        return FittedWeighting(
            weight_type=base.weight_type,
            mu_hat=base.mu_hat,
            Sigma_inv=base.Sigma_inv,
            c=base.c,
            zeta=base.zeta,
            x_obs=x_obs,
            standardised=self.standardised,
        )
