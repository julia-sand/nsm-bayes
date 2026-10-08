"""Regression tests for ``scripts/run_turin.py``.

``run_turin._run`` currently loads its training data from a hard-coded
``rca_sbi/data_turin/*.pt`` path and references an undefined loop index, so
the full run is not yet executable. The :class:`TurinModel` simulator,
however, is self-contained and deterministic, and is the piece that must not
regress, so it is tested directly here.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from nsm_bayes.simulators.benchmark_simulators.simulators import TurinModel


def _theta_true() -> torch.Tensor:
    """Return the log-space ``theta_true`` from ``configs/turin.yaml``."""
    cfg = OmegaConf.load("configs/turin.yaml")
    return torch.log(torch.tensor(cfg.theta_true, dtype=torch.float32))


def test_turin_model_deterministic():
    """The same seed gives bit-identical moment summaries."""
    t = _theta_true()
    torch.manual_seed(0)
    a = TurinModel(t, N=30, Ns=801, output="moments", epsilon=0.1, device="cpu")
    torch.manual_seed(0)
    b = TurinModel(t, N=30, Ns=801, output="moments", epsilon=0.1, device="cpu")
    assert a.shape == (30, 3), f"TurinModel moments shape {tuple(a.shape)}"
    assert torch.isfinite(a).all()
    assert torch.equal(a, b), "TurinModel is not deterministic under a fixed seed"


def test_turin_model_epsilon_changes_output():
    """Replacing half the receivers with noise must change the summaries."""
    t = _theta_true()
    torch.manual_seed(1)
    clean = TurinModel(t, N=60, Ns=801, output="moments", epsilon=0.0, device="cpu")
    torch.manual_seed(1)
    noisy = TurinModel(t, N=60, Ns=801, output="moments", epsilon=0.5, device="cpu")
    assert clean.shape == noisy.shape == (60, 3)
    assert not torch.equal(clean, noisy), "epsilon=0.5 had no effect on the output"
