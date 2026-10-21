"""Regression tests for the Turin channel-model experiment (conjugate variant).

The full pipeline is exercised end to end with a tiny config (see
:mod:`tests.conftest` for the fast stand-ins).  The data files must be
bit-identical to the normflow variant's for the same seed.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from nsm_bayes.experiments import common, turin
from tests.conftest import assert_posterior_artifacts, load_tensor


@pytest.fixture
def turin_run(turin_cfg, tmp_path):
    """Run one tiny Turin experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "turin_conjugate"
    common.run_experiment(turin_cfg, turin, common.run_conjugate_method, save_dir)
    return save_dir


def test_turin_case1_posterior(turin_run, turin_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        turin_run, d_theta=4, beta_min=float(turin_cfg.get("beta_min_case1", 0.0))
    )


def test_turin_nets_saved(turin_run):
    """The Case-1 nets and standardizers are saved with the expected keys."""
    nets = torch.load(turin_run / "conjugate_nets_0.pt", map_location="cpu", weights_only=False)
    for key in (
        "T_phi_state_dict",
        "B_phi_state_dict",
        "standardizer_x_state_dict",
        "standardizer_theta_state_dict",
        "config",
    ):
        assert key in nets, f"conjugate_nets missing {key!r}"


