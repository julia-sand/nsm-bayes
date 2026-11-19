"""Regression tests for the Turin channel-model experiment (normflow variant).

The full pipeline is exercised end to end with a tiny config (see
:mod:`tests.conftest` for the fast stand-ins).
"""
import pytest
import torch

from nsm_bayes.experiments import common, turin
from nsm_bayes.simulators.benchmark_simulators.simulators import TurinSimulator
from tests.conftest import assert_posterior_artifacts


@pytest.fixture
def turin_run(turin_cfg, tmp_path):
    """Run one tiny Turin experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "turin"
    common.run_experiment(turin_cfg, turin, common.run_normflow_methods, save_dir)
    return save_dir


def test_turin_case1_posterior(turin_run, turin_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        turin_run, d_theta=4, beta_min=float(turin_cfg.get("beta_min_general", 0.0))
    )


def test_turin_generate_training_data_shapes():
    """Training data from the prior has the expected shapes."""
    sim = TurinSimulator(N=6, Ns=33, device="cpu")
    theta, x_sim = sim.generate_training_data(6)
    assert theta.shape == (6, 4)
    assert x_sim.shape == (6, 3)
    assert torch.isfinite(x_sim).all()
