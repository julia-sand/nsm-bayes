"""Regression tests for the SIR experiment (conjugate variant).

SIR data is misspecified/contaminated, so the pseudo-true parameter is not
``cfg.theta_true``.  The checks are that the pipeline runs end to end and
produces structurally valid, non-negative summary statistics and a valid
Case-1 posterior.
"""
import pytest

from nsm_bayes.experiments import common, sir
from tests.conftest import assert_posterior_artifacts

@pytest.fixture
def sir_run(sir_cfg, tmp_path):
    """Run one tiny SIR experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "sir_conjugate"
    common.run_experiment(sir_cfg, sir, common.run_conjugate_method, save_dir)
    return save_dir


def test_sir_case1_posterior(sir_run, sir_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_run, d_theta=4, beta_min=float(sir_cfg.get("beta_min_case1", 0.0))
    )
