"""Build the self-contained GNK pickle bundle for the beta-stepping notebooks.

This script writes one bundle so the per-experiment notebook
(:file:`compare_beta_gnk.ipynb`) can compare gradient-stepping rules
(GPC / GD / Adam) without touching ``data/`` or re-training:

* ``examples/experimental/data/test_gnk.pkl`` + ``test_gnk.pt``

The bundle is generated *from scratch* at a fixed (capped) sample size so the
notebook runs quickly for testing.  The build is deterministic for a fixed rep
(``seed = seed_offset + rep``).

The bundle is a dict with the keys::

    experiment, rep, seed, theta_true, prior_mean, prior_cov,
    theta, x_sim, x_obs_mis, x_obs, d_x, d_theta,
    config  (relevant scalar config values for the notebook),
    meta    (sizes, timestamps, ...),

plus a model checkpoint ``<experiment>.pt`` next to the ``.pkl`` (nets and
standardisers, written by ``EnergyBasedModel.save_checkpoint``).

Shared helpers (paths, config loading, data generation, bundle assembly and
saving) live in :mod:`nsm_bayes.experiments.test_data.common`.

Run from the repo root (with the package installed, e.g. ``pip install -e .``)::

    python -m nsm_bayes.experiments.test_data.make_test_data_gnk            # build the bundle
    python -m nsm_bayes.experiments.test_data.make_test_data_gnk --force    # overwrite
"""
from pathlib import Path

from nsm_bayes.experiments import gnk as gnk_exp
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
GNK_REP = 0

# number of samples + observations
GNK_NUM_SAMPLES = 8000
GNK_NUM_OBS = 60


def make_gnk_bundle(force: bool = False) -> Path:
    """Generate one GNK bundle from scratch (data + trained surrogate)."""
    out_path = DATA_DIR / "test_gnk.pkl"
    ckpt_path = DATA_DIR / "test_gnk.pt"
    if out_path.exists() and ckpt_path.exists() and not force:
        print(f"gnk bundle already exists ({out_path}, {ckpt_path}); use --force to rebuild")
        return out_path

    cfg = load_config("gnk", num_samples=GNK_NUM_SAMPLES, n_obs=GNK_NUM_OBS,
                      robust_flag=True)
    _, prior, theta_true, d_x, seed, data = generate_experiment_data(
        gnk_exp, cfg, GNK_REP)

    model = EnergyBasedModel(
        d_x=d_x, d_theta=theta_true.shape[0], hidden_dim=cfg.hidden_dim
    )
    model.fit(x_sim=data.x_sim, theta_sim=data.theta)
    model.save_checkpoint(ckpt_path)
    print(f"wrote {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")

    bundle = make_bundle(
        "gnk", GNK_REP, seed, cfg, prior, theta_true, d_x, data,
        config_keys=[
            "hidden_dim", "robust_flag", "beta_base_case1",
            "beta_min_case1", "alpha", "B", "T", "n_obs",
        ],
    )
    save_bundle(out_path, bundle)
    return out_path


if __name__ == "__main__":
    cli_main(make_gnk_bundle)
