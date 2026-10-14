"""Unit tests for the tuning-constant refactor.

Covers:
* ``sigma_inverse`` matches ``torch.linalg.inv(S + jitter * I)`` (d=1 and d=3)
  and the ``jitter=0`` case matches a plain inverse;
* the beta step size in both calibration functions is ``10 / (t + 10)``;
* ``set_seed`` makes random, numpy and torch draws reproducible.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from nsm_bayes.gpc import calibrate_beta, calibrate_beta_gpc
from nsm_bayes.method import sigma_inverse
from nsm_bayes.utils import set_seed


# --------------------------------------------------------------------------- #
# sigma_inverse                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [1, 3])
def test_sigma_inverse_matches_inv_with_jitter(d: int):
    """``sigma_inverse(S)`` equals ``inv(S + 1e-6 * I)`` in both cases."""
    torch.manual_seed(0)
    # build a small PD matrix
    A = torch.randn(d, d)
    S = A @ A.T + d * torch.eye(d)
    ref = torch.linalg.inv(S + 1e-6 * torch.eye(d))
    out = sigma_inverse(S)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("d", [1, 3])
def test_sigma_inverse_zero_jitter(d: int):
    """``sigma_inverse(S, jitter=0)`` equals ``inv(S)``."""
    torch.manual_seed(1)
    A = torch.randn(d, d)
    S = A @ A.T + d * torch.eye(d)
    ref = torch.linalg.inv(S)
    out = sigma_inverse(S, jitter=0.0)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_sigma_inverse_scalar_shape():
    """A scalar variance input returns a 1-element tensor."""
    out = sigma_inverse(torch.tensor(4.0))
    assert out.numel() == 1
    assert torch.allclose(out, torch.tensor(1.0 / (4.0 + 1e-6)))


# --------------------------------------------------------------------------- #
# beta step size                                                               #
# --------------------------------------------------------------------------- #
def _default_step(fn) -> list[float]:
    """Return the step size used at t = 0..20 for a calibration function.

    Both functions currently accept their schedule as a callable parameter
    with a default; we evaluate that default. (If a future refactor switches
    to ``lr_beta_scale``/``lr_beta_offset`` floats, evaluate ``scale /
    (t + offset)`` instead.)
    """
    params = inspect.signature(fn).parameters
    if "step_schedule" in params:
        sched = params["step_schedule"].default
        return [sched(t) for t in range(21)]
    if "learning_rate_fn" in params:
        sched = params["learning_rate_fn"].default
        return [sched(t) for t in range(21)]
    if "lr_beta_scale" in params and "lr_beta_offset" in params:
        a = params["lr_beta_scale"].default
        b = params["lr_beta_offset"].default
        return [a / (t + b) for t in range(21)]
    raise AssertionError("no step-size parameter found")


@pytest.mark.parametrize("fn", [calibrate_beta, calibrate_beta_gpc])
def test_beta_step_size_10_over_t_plus_10(fn):
    """The step size at t = 0..20 equals 10 / (t + 10)."""
    got = _default_step(fn)
    expected = [10.0 / (t + 10.0) for t in range(21)]
    assert got == expected


# --------------------------------------------------------------------------- #
# set_seed                                                                     #
# --------------------------------------------------------------------------- #
def test_set_seed_reproducible():
    """``set_seed(5)`` twice gives identical draws from all three RNGs."""
    import random

    set_seed(5)
    a_t = torch.rand(8)
    a_n = np.random.rand(8)
    a_p = [random.random() for _ in range(8)]

    set_seed(5)
    b_t = torch.rand(8)
    b_n = np.random.rand(8)
    b_p = [random.random() for _ in range(8)]

    assert torch.equal(a_t, b_t)
    assert np.array_equal(a_n, b_n)
    assert a_p == b_p
