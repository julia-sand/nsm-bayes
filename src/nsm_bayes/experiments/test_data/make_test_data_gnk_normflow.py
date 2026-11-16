"""Build the self-contained GNK pickle bundle for the normflow notebook.

Writes one data bundle + one model checkpoint so the notebook
(:file:`examples/gnk_general.ipynb`) can run the full general (normalising
flow) workflow — flow training skipped, loaded from checkpoint — without
re-training:

* ``examples/experimental/data/test_gnk_normflow.pkl`` (data bundle)
* ``examples/experimental/data/test_gnk_normflow.pt`` (the trained SNLE flow)

The bundle is generated *from scratch* at a fixed (capped) sample size so the
notebook runs quickly for testing.  The build is deterministic for a fixed rep
(``seed = seed_offset + rep``).

The ``.pkl`` bundle is a dict with the keys::

    experiment, rep, seed, theta_true, prior_mean, prior_cov,
    theta, x_sim, x_obs_mis, x_obs, d_x, d_theta,
    config  (relevant scalar config values for the notebook),
    meta    (sizes, timestamps, ...),

The ``.pt`` checkpoint is the **whole trained estimator module**
(``GeneralModel.estimator``, an SBI normalising flow) pickled in ``.eval()``
mode.  ``GeneralModel.load_checkpoint`` does
``torch.load(path, weights_only=False)`` and assigns the result straight to
``self.estimator``, so the checkpoint is *not* a state dict — it is the full
module, loaded the way the notebook loads it::

    bundle  = torch.load(".../test_gnk_normflow.pkl", weights_only=False)
    model   = GeneralModel(bundle["prior"], MCMCSettings()).load_checkpoint(
                  ".../test_gnk_normflow.pt")
    posterior = GeneralPosterior(
        bundle["x_obs_mis"], model, bundle["prior"], MCMCSettings(),
        robust=False, weight_type="imq", c=1.0,
    )

``GeneralModel.fit`` uses sbi's own validation split + early stopping, which
needs a decent training set; the sample size below is chosen so the split is
still meaningful.

Shared helpers (paths, config loading, data generation, bundle assembly and
saving) live in :mod:`nsm_bayes.experiments.test_data.common`.

Run from the repo root (with the package installed, e.g. ``pip install -e .``)::

    python -m nsm_bayes.experiments.test_data.make_test_data_gnk_normflow            # build the bundle
    python -m nsm_bayes.experiments.test_data.make_test_data_gnk_normflow --force    # overwrite
"""
from pathlib import Path

import torch

from nsm_bayes.experiments import gnk as gnk_exp
from nsm_bayes.experiments.test_data.common import (
    DATA_DIR,
    cli_main,
    generate_experiment_data,
    load_config,
    make_bundle,
    save_bundle,
)
from nsm_bayes.models.general import GeneralModel
from nsm_bayes.models.general.mcmc_settings import MCMCSettings

# One repetition per bundle (seed = seed_offset + rep).
GNK_REP = 0

#number of samples and observations
GNK_NUM_SAMPLES = 2000
GNK_NUM_OBS = 60


def make_gnk_normflow_bundle(force: bool = False) -> Path:
    """Generate one GNK normflow bundle from scratch (data + trained flow)."""
    out_path = DATA_DIR / "test_gnk_normflow.pkl"
    ckpt_path = DATA_DIR / "test_gnk_normflow.pt"
    if out_path.exists() and ckpt_path.exists() and not force:
        print(f"gnk normflow bundle already exists ({out_path}, {ckpt_path}); use --force to rebuild")
        return out_path

    cfg = load_config("gnk", num_samples=GNK_NUM_SAMPLES, n_obs=GNK_NUM_OBS,
                      robust_flag=True)
    _, prior, theta_true, d_x, seed, data = generate_experiment_data(
        gnk_exp, cfg, GNK_REP)

    model = GeneralModel(prior, MCMCSettings(), density_estimator="maf")
    model.fit(x_sim=data.x_sim, theta_sim=data.theta)
    
    # Save the trained estimator module itself (not a state dict):
    # GeneralModel.load_checkpoint torch.loads the file straight into
    # self.estimator.  Mirror the notebook's construction of `model`.
    torch.save(model.estimator.eval(), ckpt_path)
    print(f"wrote {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")

    bundle = make_bundle(
        "gnk", GNK_REP, seed, cfg, prior, theta_true, d_x, data,
        config_keys=[
            "density_estimator", "num_transforms", "num_blocks",
            "hidden_dim", "robust_flag", "weight_function", "imq_c",
            "beta_base_general", "beta_min_general",
            "alpha", "B", "T", "n_obs",
            "num_posterior_samples", "num_chains", "warmup_steps", "thin",
        ],
    )
    save_bundle(out_path, bundle)
    return out_path


if __name__ == "__main__":
    cli_main(make_gnk_normflow_bundle)
