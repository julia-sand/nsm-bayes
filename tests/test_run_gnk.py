"""Regression tests for ``scripts/run_gnk.py``.

The pipeline is well-specified (clean g-and-k data plus a few additive
outliers), so the strongest check is that the Case-1 posterior mean lands
close to the true parameter. Artifacts are written into ``tmp_path`` via the
``_run`` seam; nothing touches the repository's ``data/`` directory.
"""
from __future__ import annotations

import pytest
import torch

from scripts import run_gnk
from tests.conftest import assert_posterior_artifacts, load_tensor


@pytest.fixture
def gnk_run(gnk_cfg, tmp_path):
    """Run one tiny GNK experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "gnk"
    run_gnk._run(cfg=gnk_cfg, save_dir=save_dir)
    return save_dir


def test_gnk_training_data_shapes(gnk_run, gnk_cfg):
    """Training data artifacts have the expected shapes and are finite."""
    theta = load_tensor(gnk_run, "theta_0.pkl")
    x_sim = load_tensor(gnk_run, "x_sim_0.pkl")
    assert theta.shape == (gnk_cfg.num_samples, 4)
    assert x_sim.shape[0] == gnk_cfg.num_samples
    assert torch.isfinite(theta).all() and torch.isfinite(x_sim).all()


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


def test_gnk_nets_saved(gnk_run):
    """The Case-1 nets and standardizers are saved with the expected keys."""
    nets = torch.load(gnk_run / "case1_nets_0.pt", map_location="cpu", weights_only=False)
    for key in (
        "T_phi_state_dict",
        "B_phi_state_dict",
        "standardizer_x_state_dict",
        "standardizer_theta_state_dict",
        "config",
    ):
        assert key in nets, f"case1_nets missing {key!r}"


def test_gnk_case1_posterior(gnk_run, gnk_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        gnk_run, d_theta=4, beta_min=float(gnk_cfg.get("beta_min_case1", 0.0))
    )


def test_gnk_posterior_near_truth(gnk_run, gnk_cfg):
    """The Case-1 posterior mean is within 3 posterior stds of the truth."""
    mu = load_tensor(gnk_run, "posterior_mean_case1_0.pkl")
    cov = load_tensor(gnk_run, "posterior_covariance_case1_0.pkl")
    theta_true = torch.tensor(gnk_cfg.theta_true, dtype=torch.float32)
    z = (mu - theta_true).abs() / torch.sqrt(torch.diagonal(cov)).clamp_min(1e-2)
    assert (z < 3.0).all(), f"posterior mean too far from truth (max z {z.max()})"
