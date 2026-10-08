"""Normalising-flow model for the general NSM-Bayes method.

    The general method uses a normalising flow as its likelihood estimator:
    an sbi ``SNLE`` trainer with a masked autoregressive flow (``"maf"``)
    density estimator. :func:`train_normflow` trains the flow on simulated
    ``(theta, x)`` pairs and :func:`make_nle_logprob` wraps a trained flow
    into the ``f(x, theta) -> (1,)`` log-probability callable expected by the
    shared calibration and score-matching utilities.

    The flow itself is a neural network in ``sbi`` (``NFlowsFlow``), so this
    module only contains the training entry point and the shape-conversion
    helper; no flow parameters live in this repository.
"""
from __future__ import annotations

from typing import Callable

import torch
from sbi.inference import SNLE
from torch.distributions import MultivariateNormal

__all__ = ["train_normflow", "make_nle_logprob"]


def train_normflow(
    theta_sim: torch.Tensor,
    x_sim: torch.Tensor,
    prior: MultivariateNormal,
    density_estimator: str = "maf",
) -> "SNLE":
    """Train a normalising-flow likelihood estimator on simulated data.

    Parameters
    ----------
    theta_sim : torch.Tensor
        Simulated parameters with shape ``(m, d_theta)``, drawn from `prior`.
    x_sim : torch.Tensor
        Simulated data with shape ``(m, d_x)`` corresponding to `theta_sim`.
    prior : torch.distributions.MultivariateNormal
        Prior distribution over the parameters.
    density_estimator : str, default "maf"
        Name of the sbi density estimator, e.g. ``"maf"`` or ``"mdn"``.

    Returns
    -------
    sbi.inference.SNLE
        Trained SBLE instance whose ``log_prob(x, theta)`` evaluates
        ``log q_phi(x | theta)`` for the normalising flow.
    """
    inference = SNLE(prior, density_estimator=density_estimator)
    return inference.append_simulations(theta_sim, x_sim).train()


def make_nle_logprob(estimator) -> Callable:
    """Wrap a trained flow into an ``f(x, theta) -> (1,)`` log-probability callable.

    The returned callable is compatible with the shared utilities
    (``calibrate_beta`` in ``gpc.py`` and the score-matching loss in
    ``method.py``):

    - accepts `x` with shape ``(d_x,)`` or ``(1, d_x)``
    - accepts `theta` with shape ``(d_theta,)`` or ``(1, d_theta)``
    - handles device/dtype conversion
    - reshapes to the ``(sample_dim, batch_dim, d_x)`` convention used by
      SBI's ``NFlowsFlow``

    Parameters
    ----------
    estimator : sbi.inference.SNLE
        Trained likelihood estimator whose ``log_prob`` method is wrapped.

    Returns
    -------
    Callable
        ``f(x, theta)`` returning a tensor of shape ``(1,)`` with
        ``log q_phi(x | theta)``.
    """
    # pick device/dtype from the estimator
    p = next(estimator.parameters())
    dev, dt = p.device, p.dtype

    def f(x, theta):
        # ensure 1D event shapes
        x_row = x.reshape(-1)
        th_row = theta.reshape(-1)

        # reshape to (sample_dim=1, batch_dim=1, d_x)
        x_b = x_row.to(device=dev, dtype=dt).reshape(1, 1, -1).contiguous()
        th_b = th_row.to(device=dev, dtype=dt).reshape(1, -1).contiguous()

        # NFlowsFlow.log_prob returns shape (sample_dim, batch_dim) = (1,1)
        out = estimator.log_prob(x_b, th_b).reshape(-1)  # -> (1,)
        return out

    return f
