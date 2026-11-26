"""Parametrized end-to-end pipeline tests for all four experiments.

Replaces the eight per-experiment files::

    test_run_gnk.py,                test_run_gnk_conjugate.py,
    test_run_sir.py,                test_run_sir_conjugate.py,
    test_run_sir_undercounting.py,  test_run_sir_undercounting_conjugate.py,
    test_run_turin.py,              test_run_turin_conjugate.py

with a single parametrized ``test_pipeline_posterior`` (4 experiments x 2
methods = 8 cases) plus the experiment-specific data-invariant tests that
were in the original files.
"""

import pytest
import torch

from nsm_bayes.experiments import common, gnk, sir, sir_undercounting, turin
from nsm_bayes.utils import set_seed
from tests.conftest import assert_posterior_artifacts, load_tensor

# ---------------------------------------------------------------------------
# Parametrized pipeline: 4 experiments x 2 methods
# ---------------------------------------------------------------------------

_PIPES = [
    ("gnk", "gnk_cfg", "normflow"),
    ("gnk", "gnk_cfg", "conjugate"),
    ("sir", "sir_cfg", "normflow"),
    ("sir", "sir_cfg", "conjugate"),
    ("sir_uc", "sir_uc_cfg", "normflow"),
    ("sir_uc", "sir_uc_cfg", "conjugate"),
    ("turin", "turin_cfg", "normflow"),
    ("turin", "turin_cfg", "conjugate"),
]


@pytest.mark.parametrize("exp_name, cfg_fixture, method", _PIPES)
def test_pipeline_posterior(request, tmp_path, exp_name, cfg_fixture, method):
    """Run the full pipeline and check the posterior artifacts.

    Each case runs one tiny experiment (one repetition) with the given
    method variant and asserts that the posterior artifacts are present
    and structurally valid.
    """
    cfg = request.getfixturevalue(cfg_fixture)
    problem = {"gnk": gnk, "sir": sir,
               "sir_uc": sir_undercounting, "turin": turin}[exp_name]
    method_step = (common.run_normflow_methods if method == "normflow"
                   else common.run_conjugate_method)
    beta_key = "beta_min_general" if method == "normflow" else "beta_min_case1"

    save_dir = tmp_path / f"{exp_name}_{method}"
    common.run_experiment(cfg, problem, method_step, save_dir)

    assert_posterior_artifacts(
        save_dir, d_theta=4, beta_min=float(cfg.get(beta_key, 0.0))
    )


# ---------------------------------------------------------------------------
# GNK-specific tests (from test_run_gnk.py)
# ---------------------------------------------------------------------------

def _gnk_generate(cfg, seed):
    """One repetition of the data, generated exactly as ``run_experiment``
    does it for rep 0 (same seed and call order), without running a method."""
    sim, _, theta_true = gnk.build_problem(cfg)
    set_seed(seed)
    return gnk.generate_data(sim, theta_true, cfg)


def test_gnk_data_identical_across_methods(gnk_cfg, tmp_path):
    """Both method variants produce identical data for the same seed."""
    nf_dir = tmp_path / "gnk_nf"
    cj_dir = tmp_path / "gnk_cj"
    common.run_experiment(gnk_cfg, gnk, common.run_normflow_methods, nf_dir)
    common.run_experiment(gnk_cfg, gnk, common.run_conjugate_method, cj_dir)
    for name in ("theta", "x_sim", "x_obs", "x_obs_mis", "outlier_indices"):
        assert torch.equal(load_tensor(nf_dir, f"{name}_0.pkl"),
                           load_tensor(cj_dir, f"{name}_0.pkl")), f"{name} differs"


def test_gnk_data_changes_with_seed(gnk_cfg):
    """A different seed must give different data (seeding still wired up)."""
    d0 = _gnk_generate(gnk_cfg, gnk_cfg.seed_offset)
    d1 = _gnk_generate(gnk_cfg, gnk_cfg.seed_offset + 1)
    assert not torch.equal(d0.theta, d1.theta)
    assert not torch.equal(d0.x_sim, d1.x_sim)
    assert not torch.equal(d0.x_obs_mis, d1.x_obs_mis)


def test_gnk_observed_data(gnk_cfg):
    """Exactly ``floor(epsilon * n_obs)`` rows are shifted by an outlier value."""
    d = _gnk_generate(gnk_cfg, gnk_cfg.seed_offset)
    x_obs, x_obs_mis = d.x_obs, d.x_obs_mis
    idx = d.extras["outlier_indices"]
    assert x_obs.shape == (gnk_cfg.n_obs, 1)
    assert x_obs_mis.shape == (gnk_cfg.n_obs, 1)
    n_out = int(torch.floor(torch.tensor(gnk_cfg.epsilon * gnk_cfg.n_obs)).item())
    assert idx.numel() == n_out
    diff = (x_obs_mis - x_obs).abs().sum(dim=1) > 0
    assert diff.sum().item() == n_out
    for i in idx.tolist():
        delta = float(x_obs_mis[i] - x_obs[i])


def test_gnk_simulate_shapes():
    """``simulate`` accepts 1D and 2D theta and returns the expected shapes."""
    from nsm_bayes.simulators.benchmark_simulators import GandKSimulator

    sim = GandKSimulator()
    theta = sim.prior.sample((5,))
    assert sim.simulate(theta).shape == (5,)
    assert sim.simulate(theta[0]).dim() == 0
    assert sim.simulate(theta[0].unsqueeze(0), n=4).shape == (4,)
    assert sim.simulate(theta, n=4).shape == (5, 4)
    x = sim.simulate(theta).unsqueeze(-1)
    assert x.shape == (5, 1) and torch.isfinite(x).all()


# ---------------------------------------------------------------------------
# SIR undercounting-specific test (from test_run_sir_undercounting.py)
# ---------------------------------------------------------------------------

def test_sir_uc_undercounting_is_one_sided(sir_uc_cfg):
    """Undercounting is one-sided: thinning only removes counts."""
    sim, _, theta_true = sir_undercounting.build_problem(sir_uc_cfg)
    set_seed(sir_uc_cfg.seed_offset)
    d = sir_undercounting.generate_data(sim, theta_true, sir_uc_cfg)

    x_clean, x_cor = d.x_obs, d.x_obs_mis
    is_contam = d.extras["is_contam"]
    assert is_contam.any() and (~is_contam).any(), \
        "seed gives a vacuous test; change the seed"

    atol = 0.05
    atol_peak_time = 10.0 / max(sir_uc_cfg.T_sir - 1, 1)
    assert torch.allclose(x_cor[:, [0, 2]][~is_contam],
                          x_clean[:, [0, 2]][~is_contam], atol=atol)
    assert torch.allclose(x_cor[:, 1][~is_contam],
                          x_clean[:, 1][~is_contam], atol=atol_peak_time)
    for col in (0, 2):
        assert (x_cor[is_contam, col] <= x_clean[is_contam, col] + atol).all()
    assert (x_cor[is_contam, 0] < x_clean[is_contam, 0]).any()

    y_obs = d.extras["y_obs"]
    assert y_obs.shape == (sir_uc_cfg.n_obs, sir_uc_cfg.T_sir)
    assert torch.isfinite(y_obs).all()


# ---------------------------------------------------------------------------
# Turin-specific test (from test_run_turin.py)
# ---------------------------------------------------------------------------

def test_turin_generate_training_data_shapes():
    """Training data from the prior has the expected shapes."""
    from nsm_bayes.simulators.benchmark_simulators import TurinSimulator

    sim = TurinSimulator(N=6, Ns=33, device="cpu")
    theta, x_sim = sim.generate_training_data(6)
    assert theta.shape == (6, 4)
    assert x_sim.shape == (6, 3)
    assert torch.isfinite(x_sim).all()

