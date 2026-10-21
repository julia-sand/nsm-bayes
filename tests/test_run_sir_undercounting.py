"""Regression tests for the SIR undercounting experiment (normflow variant).

Undercounting is one-sided (binomial thinning), so the corrupted incidence
trajectories must never exceed the clean ones.  The tests check that
invariant directly and that the Case-1 posterior is structurally valid.
"""
from __future__ import annotations

import pytest
import torch

from nsm_bayes.experiments import common, sir_undercounting
from nsm_bayes.utils import set_seed
from tests.conftest import assert_posterior_artifacts, load_tensor


@pytest.fixture
def sir_uc_run(sir_uc_cfg, tmp_path):
    """Run one tiny SIR-undercounting experiment into ``tmp_path``."""
    save_dir = tmp_path / "sir_uc"
    common.run_experiment(sir_uc_cfg, sir_undercounting, common.run_normflow_methods, save_dir)
    return save_dir


def test_sir_uc_clean_trajectory_saved(sir_uc_run, sir_uc_cfg):
    """The clean observed trajectories are saved with the right shape."""
    y_obs = load_tensor(sir_uc_run, "y_obs_0.pkl")
    assert y_obs.shape == (sir_uc_cfg.n_obs, sir_uc_cfg.T_sir)
    assert torch.isfinite(y_obs).all()


@pytest.fixture
def sir_uc_data(sir_uc_cfg):
    """One repetition of the experiment's data, generated exactly as
    ``run_experiment`` does for rep 0, without running any method."""
    sim, prior, theta_true, d_x = sir_undercounting.build_problem(sir_uc_cfg)
    set_seed(sir_uc_cfg.seed_offset)
    return sir_undercounting.generate_data(sim, prior, theta_true, d_x, sir_uc_cfg)


def test_sir_uc_undercounting_is_one_sided(sir_uc_data):
    """Checks the corruption the experiment actually applied
    (``SIRSimulator.apply_undercounting_trajectory``), via the summaries it
    produced: thinning only removes counts, so attack rate and peak can only
    drop, and un-contaminated rows are untouched."""
    x_clean, x_cor = sir_uc_data.x_obs, sir_uc_data.x_obs_mis
    is_contam = sir_uc_data.extras["is_contam"]
    assert is_contam.any() and (~is_contam).any(), "seed gives a vacuous test; change the seed"

    # untouched trajectories give identical summaries
    assert torch.equal(x_cor[~is_contam], x_clean[~is_contam])
    # attack rate (col 0) and peak (col 2) never increase under thinning ...
    for col in (0, 2):
        assert (x_cor[is_contam, col] <= x_clean[is_contam, col]).all()
    # ... and with q < 1 the attack rate of some contaminated trajectory drops
    assert (x_cor[is_contam, 0] < x_clean[is_contam, 0]).any()


def test_sir_uc_case1_posterior(sir_uc_run, sir_uc_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_uc_run, d_theta=4, beta_min=float(sir_uc_cfg.get("beta_min_general", 0.0))
    )


def test_sir_uc_data_identical_across_methods(sir_uc_cfg, tmp_path):
    """Both method variants produce identical data for the same seed."""
    nf_dir = tmp_path / "sir_uc_nf"
    cj_dir = tmp_path / "sir_uc_cj"
    common.run_experiment(sir_uc_cfg, sir_undercounting, common.run_normflow_methods, nf_dir)
    common.run_experiment(sir_uc_cfg, sir_undercounting, common.run_conjugate_method, cj_dir)
    for name in ("theta", "x_sim", "x_obs_mis", "y_obs", "is_contam"):
        assert torch.equal(load_tensor(nf_dir, f"{name}_0.pkl"),
                           load_tensor(cj_dir, f"{name}_0.pkl")), f"{name} differs"
