"""Regression tests for ``scripts/run_turin.py``.

The full pipeline is exercised end to end through ``run_turin._run`` with a
tiny config (see :mod:`tests.conftest` for the fast NLE/
score-matching stand-ins).  The :class:`TurinModel` simulator is also tested
directly: it must stay self-contained and deterministic under a fixed seed.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from nsm_bayes.simulators.benchmark_simulators.simulators import TurinModel
from scripts import run_turin
from tests.conftest import assert_posterior_artifacts, load_tensor


def _theta_true() -> torch.Tensor:
    """Return the log-space ``theta_true`` from ``configs/turin.yaml``."""
    cfg = OmegaConf.load("configs/turin.yaml")
    return torch.log(torch.tensor(cfg.theta_true, dtype=torch.float32))


@pytest.fixture
def turin_run(turin_cfg, tmp_path):
    """Run one tiny Turin experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "turin"
    run_turin._run(cfg=turin_cfg, save_dir=save_dir)
    return save_dir


def test_turin_model_deterministic():
    """The same seed gives bit-identical moment summaries."""
    t = _theta_true()
    sim = TurinModel(N=30, Ns=801, device="cpu")
    torch.manual_seed(0)
    a = sim.simulate(t, output="moments", epsilon=0.1)
    torch.manual_seed(0)
    b = sim.simulate(t, output="moments", epsilon=0.1)
    assert a.shape == (30, 3), f"TurinModel moments shape {tuple(a.shape)}"
    assert torch.isfinite(a).all()
    assert torch.equal(a, b), "TurinModel is not deterministic under a fixed seed"


def test_turin_model_epsilon_changes_output():
    """Replacing half the receivers with noise must change the summaries."""
    t = _theta_true()
    n = 60
    torch.manual_seed(1)
    sim = TurinModel(N=n, Ns=801, device="cpu")
    clean = sim.simulate(t, output="moments", epsilon=0.0)
    torch.manual_seed(1)
    noisy = sim.simulate(t, output="moments", epsilon=0.5)
    assert clean.shape == noisy.shape == (n, 3)
    assert not torch.equal(clean, noisy), "epsilon=0.5 had no effect on the output"


def test_turin_case1_posterior(turin_run, turin_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        turin_run, d_theta=4, beta_min=float(turin_cfg.get("beta_min_case1", 0.0))
    )


def test_turin_simulated_data_shapes(turin_run, turin_cfg):
    """Simulated training and observed data have the expected shapes.

    Each training row is one receiver simulated under its own theta
    (``num_samples`` receivers total).  The observed dataset is one
    simulation of ``n_obs`` receivers under ``theta_true`` with
    ``epsilon`` outliers.
    """
    theta = load_tensor(turin_run, "theta_0.pkl")
    x_sim = load_tensor(turin_run, "x_sim_0.pkl")
    x_obs_mis = load_tensor(turin_run, "x_obs_mis_0.pkl")

    assert theta.shape == (turin_cfg.num_samples, 4)
    assert x_sim.shape == (turin_cfg.num_samples, turin_cfg.d_x)
    assert x_obs_mis.shape == (turin_cfg.n_obs, turin_cfg.d_x)
    for arr, name in ((theta, "theta"), (x_sim, "x_sim"), (x_obs_mis, "x_obs_mis")):
        assert torch.isfinite(arr).all(), f"{name} has non-finite entries"


def test_turin_prior():
    """The built-in prior matches the hyperparameters in ``configs/turin.yaml``."""
    cfg = OmegaConf.load("configs/turin.yaml")
    sim = TurinModel()
    prior = sim.prior()
    assert torch.allclose(prior.loc, torch.tensor(cfg.prior_mean, dtype=torch.float32))
    assert torch.allclose(
        prior.covariance_matrix, torch.tensor(cfg.prior_cov, dtype=torch.float32)
    )
    assert prior.sample((5,)).shape == (5, 4)
    assert torch.isfinite(prior.sample((10,))).all()


def test_turin_generate_training_data_shapes():
    """Training data from the prior has the expected shapes."""
    sim = TurinModel(N=6, Ns=33, device="cpu")
    theta, x_sim = sim.generate_training_data(sim.prior(), 6)
    assert theta.shape == (6, 4)
    assert x_sim.shape == (6, 3)
    assert torch.isfinite(x_sim).all()
