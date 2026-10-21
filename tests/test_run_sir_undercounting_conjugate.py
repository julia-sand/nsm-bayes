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
from scripts.conjugate import run_sir_undercounting
from scripts.normflow import run_sir_undercounting as run_sir_uc_normflow
from tests.conftest import assert_posterior_artifacts, load_tensor


def _theta_true() -> torch.Tensor:
    """Return the true ``theta_true`` from ``configs/sir_undercounting.yaml``."""
    cfg = OmegaConf.load("configs/sir_undercounting.yaml")
    return torch.tensor(cfg.theta_true, dtype=torch.float32)


@pytest.fixture
def sir_un_run(sir_uc_cfg, tmp_path):
    """Run one tiny SIR-un experiment into ``tmp_path``."""
    save_dir = tmp_path / "sir_undercounting_conjugate"
    common.run_experiment(sir_uc_cfg, sir_undercounting, common.run_conjugate_method, save_dir)
    return save_dir


def test_sir_un_model_deterministic():
    """The same seed gives bit-identical trajectories."""
    sim = SIRSimulator(obs_model="poisson")
    torch.manual_seed(0)
    a = sim.simulate(_theta_true().repeat(4, 1), T=40, N=1000)
    torch.manual_seed(0)
    b = sim.simulate(_theta_true().repeat(4, 1), T=40, N=1000)
    assert a.shape == (4, 40), f"SIR trajectories shape {tuple(a.shape)}"
    assert torch.isfinite(a).all()
    assert torch.equal(a, b), "SIRSimulator is not deterministic under a fixed seed"


def test_sir_un_model_changes_with_seed():
    """Different seeds give different trajectories."""
    sim = SIRSimulator(obs_model="poisson")
    torch.manual_seed(1)
    clean = sim.simulate(_theta_true().repeat(4, 1), T=40, N=1000)
    torch.manual_seed(2)
    noisy = sim.simulate(_theta_true().repeat(4, 1), T=40, N=1000)
    assert clean.shape == noisy.shape == (4, 40)
    assert not torch.equal(clean, noisy), "Different seeds had no effect on the output"


def test_sir_un_case1_posterior(sir_un_run, sir_uc_cfg):
    """The Case-1 posterior artifacts exist and are structurally valid."""
    assert_posterior_artifacts(
        sir_un_run, d_theta=4, beta_min=float(sir_uc_cfg.get("beta_min_case1", 0.0))
    )


def test_sir_un_simulated_data_shapes(sir_un_run, sir_uc_cfg):
    """Simulated training and observed data have the expected shapes."""
    theta = load_tensor(sir_un_run, "theta_0.pkl")
    x_sim = load_tensor(sir_un_run, "x_sim_0.pkl")
    x_obs_mis = load_tensor(sir_un_run, "x_obs_mis_0.pkl")

    assert theta.shape == (sir_uc_cfg.num_samples, 4)
    assert x_sim.shape == (sir_uc_cfg.num_samples, sir_uc_cfg.d_x)
    assert x_obs_mis.shape == (sir_uc_cfg.n_obs, sir_uc_cfg.d_x)
    for arr, name in ((theta, "theta"), (x_sim, "x_sim"), (x_obs_mis, "x_obs_mis")):
        assert torch.isfinite(arr).all(), f"{name} has non-finite entries"
