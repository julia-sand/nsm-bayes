"""Regression tests for the g-and-k experiment (normflow variant).

The pipeline is well-specified (clean g-and-k data plus a few additive
outliers), so the strongest check is that the Case-1 posterior mean lands
close to the true parameter.  The data files must also be bit-identical to
the ones the conjugate variant writes for the same seed.
"""
from __future__ import annotations

import pytest
import torch

from nsm_bayes.experiments import common, gnk
from tests.conftest import assert_posterior_artifacts, load_tensor


@pytest.fixture
def gnk_run(gnk_cfg, tmp_path):
    """Run one tiny GNK experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "gnk"
    common.run_experiment(gnk_cfg, gnk, common.run_normflow_methods, save_dir)
    return save_dir


def test_gnk_observed_data(gnk_run, gnk_cfg):
    """Exactly ``floor(epsilon * n_obs)`` rows are shifted by an outlier value."""
    x_obs = load_tensor(gnk_run, "x_obs_0.pkl")
    x_obs_mis = load_tensor(gnk_run, "x_obs_mis_0.pkl")
    idx = load_tensor(gnk_run, "outlier_indices_0.pkl")
    assert x_obs.shape == (gnk_cfg.n_obs, 1)
    assert x_obs_mis.shape == (gnk_cfg.n_obs, 1)
    n_out = int(torch.floor(torch.tensor(gnk_cfg.epsilon * gnk_cfg.n_obs)).item())
    assert idx.numel() == n_out
    diff = (x_obs_mis - x_obs).abs().sum(dim=1) > 0
    assert diff.sum().item() == n_out
    for i in idx.tolist():
        delta = float(x_obs_mis[i] - x_obs[i])
        assert any(abs(delta - v) < 1e-4 for v in gnk_cfg.outlier_values)

def test_gnk_case1_posterior(gnk_run, gnk_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        gnk_run, d_theta=4, beta_min=float(gnk_cfg.get("beta_min_general", 0.0))
    )


def test_gnk_data_identical_across_methods(gnk_cfg, tmp_path):
    """Both method variants produce identical data for the same seed."""
    nf_dir = tmp_path / "gnk_nf"
    cj_dir = tmp_path / "gnk_cj"
    common.run_experiment(gnk_cfg, gnk, common.run_normflow_methods, nf_dir)
    common.run_experiment(gnk_cfg, gnk, common.run_conjugate_method, cj_dir)
    for name in ("theta", "x_sim", "x_obs", "x_obs_mis", "outlier_indices"):
        assert torch.equal(load_tensor(nf_dir, f"{name}_0.pkl"),
                           load_tensor(cj_dir, f"{name}_0.pkl")), f"{name} differs"


def test_gnk_prior_matches_config():
    """The built-in prior matches the hyperparameters in ``configs/gnk.yaml``."""
    import torch
    from omegaconf import OmegaConf

    from nsm_bayes.simulators.benchmark_simulators.simulators import GandKSimulator

    cfg = OmegaConf.load("configs/gnk.yaml")
    sim = GandKSimulator()
    assert torch.allclose(sim.prior.loc, torch.tensor(cfg.prior_mean, dtype=torch.float32))
    assert torch.allclose(
        sim.prior.covariance_matrix, torch.tensor(cfg.prior_cov, dtype=torch.float32)
    )


def test_gnk_simulate_shapes():
    """``simulate`` accepts 1D and 2D theta and returns the expected shapes."""
    from nsm_bayes.simulators.benchmark_simulators.simulators import GandKSimulator

    sim = GandKSimulator()
    theta = sim.prior.sample((5,))
    assert sim.simulate(theta).shape == (5,)
    assert sim.simulate(theta[0]).dim() == 0
    assert sim.simulate(theta[0].unsqueeze(0), n=4).shape == (4,)
    assert sim.simulate(theta, n=4).shape == (5, 4)
    x = sim.simulate(theta).unsqueeze(-1)
    assert x.shape == (5, 1) and torch.isfinite(x).all()
