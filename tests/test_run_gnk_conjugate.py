"""Regression tests for the g-and-k experiment (conjugate variant).

The full pipeline is exercised end to end through
``scripts.conjugate.run_gnk._run`` with a tiny config (see
:mod:`tests.conftest` for the fast NLE/score-matching stand-ins).
The data files must also be bit-identical to the normflow variant's.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from nsm_bayes.experiments import common, gnk
from nsm_bayes.simulators.benchmark_simulators.simulators import GNKSimulator
from scripts.conjugate import run_gnk
from scripts.normflow import run_gnk as run_gnk_normflow
from tests.conftest import assert_posterior_artifacts, load_tensor


def _theta_true() -> torch.Tensor:
    """Return the ``theta_true`` from ``configs/gnk.yaml``."""
    cfg = OmegaConf.load("configs/gnk.yaml")
    return torch.tensor(cfg.theta_true, dtype=torch.float32)


@pytest.fixture
def gnk_run(gnk_cfg, tmp_path):
    """Run one tiny GNK experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "gnk_conjugate"
    common.run_experiment(gnk_cfg, gnk, common.run_conjugate_method, save_dir)
    return save_dir


def test_gnk_model_deterministic():
    """The same seed gives bit-identical data."""
    t = _theta_true()
    sim = GNKSimulator()
    torch.manual_seed(0)
    a = sim.simulate(t, n=4)
    torch.manual_seed(0)
    b = sim.simulate(t, n=4)
    assert a.shape == (4,)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b), "GNKSimulator is not deterministic under a fixed seed"


def test_gnk_model_data_changes_with_seed():
    """Different seeds give different data."""
    t = _theta_true()
    sim = GNKSimulator()
    torch.manual_seed(1)
    clean = sim.simulate(t, n=4)
    torch.manual_seed(2)
    noisy = sim.simulate(t, n=4)
    assert clean.shape == noisy.shape == (4,)
    assert not torch.equal(clean, noisy), "Different seeds had no effect on the output"


def test_gnk_case1_posterior(gnk_run, gnk_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        gnk_run, d_theta=4, beta_min=float(gnk_cfg.get("beta_min_case1", 0.0))
    )


def test_gnk_simulated_data_shapes(gnk_run, gnk_cfg):
    """Simulated training and observed data have the expected shapes."""
    theta = load_tensor(gnk_run, "theta_0.pkl")
    x_sim = load_tensor(gnk_run, "x_sim_0.pkl")
    x_obs_mis = load_tensor(gnk_run, "x_obs_mis_0.pkl")

    assert theta.shape == (gnk_cfg.num_samples, 4)
    assert x_sim.shape == (gnk_cfg.num_samples, gnk_cfg.d_x)
    assert x_obs_mis.shape == (gnk_cfg.n_obs, gnk_cfg.d_x)
    for arr, name in ((theta, "theta"), (x_sim, "x_sim"), (x_obs_mis, "x_obs_mis")):
        assert torch.isfinite(arr).all(), f"{name} has non-finite entries"


def test_gnk_nets_saved(gnk_run):
    """The Case-1 nets and standardizers are saved with the expected keys."""
    nets = torch.load(gnk_run / "conjugate_nets_0.pt", map_location="cpu", weights_only=False)
    for key in (
        "T_phi_state_dict",
        "B_phi_state_dict",
        "standardizer_x_state_dict",
        "standardizer_theta_state_dict",
        "config",
    ):
        assert key in nets, f"conjugate_nets missing {key!r}"


def test_gnk_data_identical_across_methods(gnk_cfg, tmp_path):
    """Both method variants produce identical data for the same seed."""
    nf_dir = tmp_path / "gnk_nf"
    common.run_experiment(gnk_cfg, gnk, common.run_normflow_methods, nf_dir)
    cj_dir = tmp_path / "gnk_cj"
    common.run_experiment(gnk_cfg, gnk, common.run_conjugate_method, cj_dir)
    for name in ("theta", "x_sim", "x_obs", "x_obs_mis", "outlier_indices"):
        assert torch.equal(load_tensor(nf_dir, f"{name}_0.pkl"),
                           load_tensor(cj_dir, f"{name}_0.pkl")), f"{name} differs"


def test_gnk_prior():
    """The built-in prior matches the hyperparameters in ``configs/gnk.yaml``."""
    cfg = OmegaConf.load("configs/gnk.yaml")
    sim = GNKSimulator()
    prior = sim.prior()
    assert torch.allclose(prior.loc, torch.tensor(cfg.prior_mean, dtype=torch.float32))
    assert torch.allclose(
        prior.covariance_matrix, torch.tensor(cfg.prior_cov, dtype=torch.float32)
    )
    assert prior.sample((5,)).shape == (5, 4)
    assert torch.isfinite(prior.sample((10,))).all()
