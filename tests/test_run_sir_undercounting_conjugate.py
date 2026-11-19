"""Regression tests for the SIR undercounting experiment (conjugate variant).

The full pipeline is exercised end to end through
``scripts.conjugate.run_sir_undercounting._run`` with a tiny config (see
:mod:`tests.conftest` for the fast stand-ins).
"""
import pytest

from nsm_bayes.experiments import common, sir_undercounting
from tests.conftest import assert_posterior_artifacts


@pytest.fixture
def sir_un_run(sir_uc_cfg, tmp_path):
    """Run one tiny SIR-un experiment into ``tmp_path``."""
    save_dir = tmp_path / "sir_undercounting_conjugate"
    common.run_experiment(sir_uc_cfg, sir_undercounting, common.run_conjugate_method, save_dir)
    return save_dir


def test_sir_un_case1_posterior(sir_un_run, sir_uc_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_un_run, d_theta=4, beta_min=float(sir_uc_cfg.get("beta_min_case1", 0.01))
    )

