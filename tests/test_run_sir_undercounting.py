"""Regression tests for ``scripts/run_sir_undercounting.py``.

Undercounting is one-sided (binomial thinning), so the corrupted incidence
trajectories must never exceed the clean ones. The tests check that invariant
directly and that the Case-1 posterior is structurally valid.
"""
from __future__ import annotations

import pytest
import torch

from nsm_bayes.simulators.add_noise import apply_undercounting_trajectory
from scripts import run_sir_undercounting
from tests.conftest import assert_posterior_artifacts, load_tensor


@pytest.fixture
def sir_uc_run(sir_uc_cfg, tmp_path):
    """Run one tiny SIR-undercounting experiment into ``tmp_path``."""
    save_dir = tmp_path / "sir_uc"
    run_sir_undercounting._run(cfg=sir_uc_cfg, save_dir=save_dir)
    return save_dir


def test_sir_uc_clean_trajectory_saved(sir_uc_run, sir_uc_cfg):
    """The clean observed trajectories are saved with the right shape."""
    y_obs = load_tensor(sir_uc_run, "y_obs_0.pkl")
    assert y_obs.shape == (sir_uc_cfg.n_obs, sir_uc_cfg.T_sir)
    assert torch.isfinite(y_obs).all()


def test_sir_uc_undercounting_is_one_sided(sir_uc_cfg, sir_uc_run):
    """Thinning only removes counts; un-contaminated rows are untouched."""
    y_obs = load_tensor(sir_uc_run, "y_obs_0.pkl")
    torch.manual_seed(123)  # same seed the script uses for ind=0
    y_cor, is_contam = apply_undercounting_trajectory(
        y_obs, epsilon=sir_uc_cfg.epsilon, q=sir_uc_cfg.q, per_time=False
    )
    if is_contam.any():
        # contaminated trajectories lose counts, never gain them
        assert (y_cor[is_contam] <= y_obs[is_contam]).all()
        # with q < 1 at least one contaminated trajectory shrinks
        assert (y_cor[is_contam] < y_obs[is_contam]).any()
    if (~is_contam).any():
        assert torch.equal(y_cor[~is_contam], y_obs[~is_contam])


def test_sir_uc_case1_posterior(sir_uc_run, sir_uc_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_uc_run, d_theta=4, beta_min=float(sir_uc_cfg.get("beta_min_case1", 0.0))
    )
