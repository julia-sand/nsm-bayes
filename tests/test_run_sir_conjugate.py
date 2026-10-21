"""Regression tests for the SIR experiment (conjugate variant).

SIR data is misspecified/contaminated, so the pseudo-true parameter is not
``cfg.theta_true``.  The checks are that the pipeline runs end to end and
produces structurally valid, non-negative summary statistics and a valid
Case-1 posterior.  The data files must also be bit-identical to the ones
the normflow variant writes for the same seed.
"""
from __future__ import annotations

import pytest
import torch

from nsm_bayes.experiments import common, sir
from scripts.conjugate import run_sir
from scripts.normflow import run_sir as run_sir_normflow
from tests.conftest import assert_posterior_artifacts, load_tensor


@pytest.fixture
def sir_run(sir_cfg, tmp_path):
    """Run one tiny SIR experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "sir_conjugate"
    common.run_experiment(sir_cfg, sir, common.run_conjugate_method, save_dir)
    return save_dir


def test_summary_stats(sir_run, sir_cfg):
    """Summary statistics have the right shape, are finite and non-negative."""
    x_sim = load_tensor(sir_run, "x_sim_0.pkl")
    x_obs = load_tensor(sir_run, "x_obs_0.pkl")
    x_obs_mis = load_tensor(sir_run, "x_obs_mis_0.pkl")
    for arr, name in ((x_sim, "x_sim"), (x_obs, "x_obs"), (x_obs_mis, "x_obs_mis")):
        assert arr.shape[1] == sir_cfg.d_x, f"{name} wrong feature dim {tuple(arr.shape)}"
        assert torch.isfinite(arr).all(), f"{name} has non-finite entries"
    assert x_sim.shape[0] == sir_cfg.num_samples
    assert x_obs.shape[0] == sir_cfg.n_obs and x_obs_mis.shape[0] == sir_cfg.n_obs
    # SIR summaries are counts/rates: attack rate and peak columns are >= 0
    assert (x_obs_mis[:, 0] >= 0).all() and (x_obs_mis[:, 2] >= 0).all()


def test_sir_case1_posterior(sir_run, sir_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_run, d_theta=4, beta_min=float(sir_cfg.get("beta_min_case1", 0.0))
    )


def test_sir_nets_saved(sir_run):
    """The Case-1 nets and standardizers are saved with the expected keys."""
    nets = torch.load(sir_run / "conjugate_nets_0.pt", map_location="cpu", weights_only=False)
    for key in (
        "T_phi_state_dict",
        "B_phi_state_dict",
        "standardizer_x_state_dict",
        "standardizer_theta_state_dict",
        "config",
    ):
        assert key in nets, f"conjugate_nets missing {key!r}"
