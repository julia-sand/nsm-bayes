"""Regression tests for the g-and-k experiment (conjugate variant).

The full pipeline is exercised end to end through
``scripts.conjugate.run_gnk._run`` with a tiny config (see
:mod:`tests.conftest` for the fast NLE/score-matching stand-ins).
Cross-method data identity with the normflow variant is checked in
:mod:`tests.test_run_gnk`.
"""
import pytest
import torch

from nsm_bayes.experiments import common, gnk

from tests.conftest import assert_posterior_artifacts


@pytest.fixture
def gnk_run(gnk_cfg, tmp_path):
    """Run one tiny GNK experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "gnk_conjugate"
    common.run_experiment(gnk_cfg, gnk, common.run_conjugate_method, save_dir)
    return save_dir


def test_gnk_case1_posterior(gnk_run, gnk_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        gnk_run, d_theta=4, beta_min=float(gnk_cfg.get("beta_min_case1", 0.0))
    )


