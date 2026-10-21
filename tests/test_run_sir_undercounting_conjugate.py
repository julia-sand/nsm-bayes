"""Regression tests for the SIR undercounting experiment (conjugate variant).

The full pipeline is exercised end to end through
``scripts.conjugate.run_sir_undercounting._run`` with a tiny config (see
:mod:`tests.conftest` for the fast stand-ins).  The data files must be
bit-identical to the normflow variant's for the same seed.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from nsm_bayes.experiments import common, sir_undercounting
from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator
from tests.conftest import assert_posterior_artifacts, load_tensor


def _theta_true() -> torch.Tensor:
    """Return the true ``theta_true`` from ``configs/sir_undercounting.yaml``."""
    cfg = OmegaConf.load("configs/sir_undercounting.yaml")
    theta = torch.tensor(cfg.theta_true, dtype=torch.float32)
    # The simulator consumes the unconstrained (log/logit) parameterization,
    # mirroring `nsm_bayes.experiments.sir.build_problem`.
    return SIRSimulator().to_unconstrained(theta.unsqueeze(0)).squeeze(0)


@pytest.fixture
def sir_un_run(sir_uc_cfg, tmp_path):
    """Run one tiny SIR-un experiment into ``tmp_path``."""
    save_dir = tmp_path / "sir_undercounting_conjugate"
    common.run_experiment(sir_uc_cfg, sir_undercounting, common.run_conjugate_method, save_dir)
    return save_dir


def test_sir_un_model_changes_with_seed():
    """Different seeds give different trajectories."""
    sim = SIRSimulator(obs_model="poisson", T=40, N=1000)
    torch.manual_seed(1)
    clean = sim.simulate(_theta_true().repeat(4, 1))
    torch.manual_seed(2)
    noisy = sim.simulate(_theta_true().repeat(4, 1))
    assert clean.shape == noisy.shape == (4, 40)
    assert not torch.equal(clean, noisy), "Different seeds had no effect on the output"


def test_sir_un_case1_posterior(sir_un_run, sir_uc_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_un_run, d_theta=4, beta_min=float(sir_uc_cfg.get("beta_min_case1", 0.01))
    )

