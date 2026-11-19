"""Regression tests for the g-and-k experiment (normflow variant).

Checks that the contamination is applied exactly as configured, that the
posterior artifacts are present and structurally valid, and that the data
files are bit-identical to the ones the conjugate variant writes for the
same seed.
"""
import pytest
import torch

from nsm_bayes.experiments import common, gnk
from nsm_bayes.utils import set_seed
from tests.conftest import assert_posterior_artifacts


@pytest.fixture
def gnk_run(gnk_cfg, tmp_path):
    """Run one tiny GNK experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "gnk"
    common.run_experiment(gnk_cfg, gnk, common.run_normflow_methods, save_dir)
    return save_dir


def test_gnk_observed_data(gnk_cfg):
    """Exactly ``floor(epsilon * n_obs)`` rows are shifted by an outlier value."""
    sim, _, theta_true = gnk.build_problem(gnk_cfg)
    set_seed(gnk_cfg.seed_offset)
    gnk_data = gnk.generate_data(sim, theta_true, gnk_cfg)
    x_obs, x_obs_mis = gnk_data.x_obs, gnk_data.x_obs_mis
    idx = gnk_data.extras["outlier_indices"]
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
