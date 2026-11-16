"""Build the self-contained SIR pickle bundle for the beta-stepping notebooks.

Shared helpers (paths, config loading, data generation, bundle assembly and
saving) live in :mod:`nsm_bayes.experiments.test_data.common`.

This script writes one bundle so the per-experiment notebook
(:file:`compare_beta_sir.ipynb`) can compare gradient-stepping rules
(GPC / GD / Adam) without touching ``data/`` or re-training:

* ``examples/experimental/data/test_sir.pkl`` + ``test_sir.pt``

The bundle is generated *from scratch* at a fixed (capped) sample size so the
notebook runs quickly for testing.  The build is deterministic for a fixed rep
(``seed = seed_offset + rep``).

The bundle is a dict with the keys::

    experiment, rep, seed, theta_true, prior_mean, prior_cov,
    theta, x_sim, x_obs_mis, x_obs, x_obs_under, d_x, d_theta,
    config  (relevant scalar config values for the notebook),
    meta    (sizes, timestamps, ...),

plus a model checkpoint ``<experiment>.pt`` next to the ``.pkl`` (nets and
standardisers, written by ``EnergyBasedModel.save_checkpoint``).

Run from the repo root (with the package installed, e.g. ``pip install -e .``)::

    python -m nsm_bayes.experiments.test_data.make_test_data_sir            # build the bundle
    python -m nsm_bayes.experiments.test_data.make_test_data_sir --force    # overwrite
"""
from pathlib import Path

import torch

from nsm_bayes.experiments import sir as sir_exp
from nsm_bayes.experiments.test_data.common import (
    DATA_DIR,
    cli_main,
    generate_experiment_data,
    load_config,
    make_bundle,
    save_bundle,
)
from nsm_bayes.models.conjugate import EnergyBasedModel

# One repetition per bundle (seed = seed_offset + rep).
SIR_REP = 0

# Fixed (capped) bundle sizes so the notebook runs quickly for testing.
SIR_NUM_SAMPLES = 20000
SIR_NUM_OBS = 100
SIR_N = 100
SIR_T = 40

# std of the small Gaussian noise added to the SIR training summaries (x_sim)
# before they are saved into the bundle, so the surrogate is not trained on
# point masses.  See Vincent 2011 (arXiv:2001.11235) or Hyvärinen 2005.
SIR_X_SIM_NOISE = 1 / SIR_N

# Retention probability for the undercounted observation set, as in
# configs/sir_undercounting.yaml.
Q_UNDER = 0.5


def make_sir_bundle(force: bool = False) -> Path:
    """Generate one SIR bundle from scratch (data + trained surrogate)."""
    out_path = DATA_DIR / "test_sir.pkl"
    ckpt_path = DATA_DIR / "test_sir.pt"
    if out_path.exists() and ckpt_path.exists() and not force:
        print(f"sir bundle already exists ({out_path}, {ckpt_path}); use --force to rebuild")
        return out_path

    cfg = load_config("sir", num_samples=SIR_NUM_SAMPLES, n_obs=SIR_NUM_OBS,
                      N_sir=SIR_N, T_sir=SIR_T, robust_flag=True)
    sim, prior, theta_true, d_x, seed, data = generate_experiment_data(
        sir_exp, cfg, SIR_REP)

    # Add small Gaussian noise to the training summaries so the surrogate
    # model is not trained on point masses.
    data.x_sim = data.x_sim + torch.randn_like(data.x_sim) * SIR_X_SIM_NOISE

    # Generate a second set of contaminated observations via SIR undercounting
    # (binomial thinning of incidence trajectories) so the notebooks can
    # compare the two misspecification mechanisms side-by-side.
    y_obs_under, _ = sim.apply_undercounting_trajectory(
        data.extras["y_obs"], epsilon=cfg.epsilon, q=Q_UNDER,
    )
    x_obs_under = sim.summary(y_obs_under)

    model = EnergyBasedModel(
        d_x=d_x, d_theta=theta_true.shape[0], hidden_dim=cfg.hidden_dim
    )
    model.fit(x_sim=data.x_sim, theta_sim=data.theta)
    model.save_checkpoint(ckpt_path)
    print(f"wrote {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")

    bundle = make_bundle(
        "sir", SIR_REP, seed, cfg, prior, theta_true, d_x, data,
        config_keys=[
            "hidden_dim", "robust_flag", "beta_base_case1",
            "beta_min_case1", "alpha", "B", "T", "n_obs",
        ],
        meta_extra={"N_sir": int(cfg.N_sir), "T_sir": int(cfg.T_sir)},
        x_obs_under=x_obs_under,
    )
    save_bundle(out_path, bundle)
    return out_path


if __name__ == "__main__":
    cli_main(make_sir_bundle)
