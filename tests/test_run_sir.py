"""Regression tests for the SIR experiment (normflow variant).

SIR data is misspecified/contaminated, so the pseudo-true parameter is not
``cfg.theta_true``.  The checks are that the pipeline runs end to end and
produces structurally valid, non-negative summary statistics and a valid
Case-1 posterior.  The data files must also be bit-identical to the ones
the conjugate variant writes for the same seed.
"""
from __future__ import annotations

import pytest
import torch

from nsm_bayes.experiments import common, sir
from tests.conftest import assert_posterior_artifacts, load_tensor


@pytest.fixture
def sir_run(sir_cfg, tmp_path):
    """Run one tiny SIR experiment into ``tmp_path`` and return the save dir."""
    save_dir = tmp_path / "sir"
    common.run_experiment(sir_cfg, sir, common.run_normflow_methods, save_dir)
    return save_dir



def test_sir_case1_posterior(sir_run, sir_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_run, d_theta=4, beta_min=float(sir_cfg.get("beta_min_general", 0.0))
    )


def test_sir_data_identical_across_methods(sir_cfg, tmp_path):
    """Both method variants produce identical data for the same seed."""
    nf_dir = tmp_path / "sir_nf"
    cj_dir = tmp_path / "sir_cj"
    common.run_experiment(sir_cfg, sir, common.run_normflow_methods, nf_dir)
    common.run_experiment(sir_cfg, sir, common.run_conjugate_method, cj_dir)
    for name in ("theta", "x_sim", "x_obs", "x_obs_mis", "is_contam"):
        assert torch.equal(load_tensor(nf_dir, f"{name}_0.pkl"),
                           load_tensor(cj_dir, f"{name}_0.pkl")), f"{name} differs"


def test_sir_prior_matches_config():
    """The built-in prior matches the hyperparameters in ``configs/sir.yaml``."""
    from omegaconf import OmegaConf

    from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator

    cfg = OmegaConf.load("configs/sir.yaml")
    sim = SIRSimulator()
    mean = torch.tensor(cfg.prior_mean, dtype=torch.float32)
    cov = torch.diag(torch.tensor(cfg.prior_std, dtype=torch.float32) ** 2)
    assert torch.allclose(sim.prior.loc, mean)
    assert torch.allclose(sim.prior.covariance_matrix, cov)
    assert sim.prior.sample((5,)).shape == (5, 4)
