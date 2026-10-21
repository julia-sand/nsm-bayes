"""Regression tests for the Turin channel-model experiment (normflow variant).

The full pipeline is exercised end to end with a tiny config (see
:mod:`tests.conftest` for the fast stand-ins).  The data files must be
bit-identical to the conjugate variant's for the same seed.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from nsm_bayes.experiments import common, turin
from nsm_bayes.simulators.benchmark_simulators.simulators import TurinSimulator
from scripts.conjugate import run_turin as run_turin_conjugate
from scripts.normflow import run_turin
from tests.conftest import assert_posterior_artifacts, load_tensor


def _theta_true() -> torch.Tensor:
    """Return the log-space ``theta_true`` from ``configs/turin.yaml``."""
    cfg = OmegaConf.load("configs/turin.yaml")
    return torch.log(torch.tensor(cfg.theta_true, dtype=torch.float32))


@pytest.fixture
def turin_run(turin_cfg, tmp_path):
    """Run one tiny Turin experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "turin"
    common.run_experiment(turin_cfg, turin, common.run_normflow_methods, save_dir)
    return save_dir


def test_turin_model_deterministic():
    """The same seed gives bit-identical results."""
    t = _theta_true()
    sim = TurinSimulator(N=30, Ns=801, device="cpu")
    torch.manual_seed(0)
    a = sim.simulate(t, N=30)
    torch.manual_seed(0)
    b = sim.simulate(t, N=30)
    assert a.shape == (30, 3), f"TurinSimulator moments shape {tuple(a.shape)}"
    assert torch.isfinite(a).all()
    assert torch.equal(a, b), "TurinSimulator is not deterministic under a fixed seed"


def test_turin_model_changes_with_seed():
    """Different seeds give different data."""
    t = _theta_true()
    n = 60
    torch.manual_seed(1)
    sim = TurinSimulator(N=n, Ns=801, device="cpu")
    clean = sim.simulate(t, N=n)
    torch.manual_seed(2)
    noisy = sim.simulate(t, N=n)
    assert clean.shape == noisy.shape == (n, 3)
    assert not torch.equal(clean, noisy), "Different seeds had no effect on the output"


def test_turin_case1_posterior(turin_run, turin_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        turin_run, d_theta=4, beta_min=float(turin_cfg.get("beta_min_general", 0.0))
    )


def test_turin_simulated_data_shapes(turin_run, turin_cfg):
    """Simulated training and observed data have the expected shapes."""
    theta = load_tensor(turin_run, "theta_0.pkl")
    x_sim = load_tensor(turin_run, "x_sim_0.pkl")
    x_obs_mis = load_tensor(turin_run, "x_obs_mis_0.pkl")

    assert theta.shape == (turin_cfg.num_samples, 4)
    assert x_sim.shape == (turin_cfg.num_samples, turin_cfg.d_x)
    assert x_obs_mis.shape == (turin_cfg.n_obs, turin_cfg.d_x)
    for arr, name in ((theta, "theta"), (x_sim, "x_sim"), (x_obs_mis, "x_obs_mis")):
        assert torch.isfinite(arr).all(), f"{name} has non-finite entries"


def test_turin_data_identical_across_methods(turin_cfg, tmp_path):
    """Both method variants produce identical data for the same seed."""
    nf_dir = tmp_path / "turin_nf"
    cj_dir = tmp_path / "turin_cj"
    common.run_experiment(turin_cfg, turin, common.run_normflow_methods, nf_dir)
    common.run_experiment(turin_cfg, turin, common.run_conjugate_method, cj_dir)
    for name in ("theta", "x_sim", "x_obs_mis"):
        assert torch.equal(load_tensor(nf_dir, f"{name}_0.pkl"),
                           load_tensor(cj_dir, f"{name}_0.pkl")), f"{name} differs"


def test_turin_prior():
    """The built-in prior matches the hyperparameters in ``configs/turin.yaml``."""
    cfg = OmegaConf.load("configs/turin.yaml")
    sim = TurinSimulator()
    #prior = sim.prior
    assert torch.allclose(sim.prior.loc, torch.tensor(cfg.prior_mean, dtype=torch.float32))
    assert torch.allclose(
        sim.prior.covariance_matrix, torch.tensor(cfg.prior_cov, dtype=torch.float32)
    )
    assert sim.prior.sample((5,)).shape == (5, 4)
    assert torch.isfinite(sim.prior.sample((10,))).all()


def test_turin_generate_training_data_shapes():
    """Training data from the prior has the expected shapes."""
    sim = TurinSimulator(N=6, Ns=33, device="cpu")
    theta, x_sim = sim.generate_training_data(6)
    assert theta.shape == (6, 4)
    assert x_sim.shape == (6, 3)
    assert torch.isfinite(x_sim).all()
