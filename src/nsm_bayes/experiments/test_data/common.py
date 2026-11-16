"""Shared helpers for the ``make_test_data_*`` bundle builders.

Each builder loads a capped-size config, generates one repetition of one
experiment (seeding exactly like
:func:`nsm_bayes.experiments.common.run_experiment`), trains one surrogate
and writes a self-contained ``.pkl`` bundle plus ``.pt`` checkpoint to
:data:`DATA_DIR`, next to the notebooks that load them.
"""
import argparse
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch
from omegaconf import DictConfig, OmegaConf

from nsm_bayes.utils import set_seed

__all__ = [
    "DATA_DIR",
    "cli_main",
    "config_subset",
    "generate_experiment_data",
    "load_config",
    "make_bundle",
    "save_bundle",
]

REPO_ROOT = Path(__file__).resolve().parents[4]
# Bundles stay next to the notebooks that load them (not inside the package).
DATA_DIR = REPO_ROOT / "examples" / "experimental" / "data"
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str, **overrides) -> DictConfig:
    """Load ``configs/<name>.yaml`` with the capped bundle-size overrides."""
    cfg = OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def generate_experiment_data(exp_mod: ModuleType, cfg: DictConfig, rep: int) -> tuple:
    """Build the problem, seed with ``cfg.seed_offset + rep``, generate data.

    Same seeding and call order as ``run_experiment``, so the bundle data is
    what the experiment would generate for this seed.

    Returns
    -------
    (sim, prior, theta_true, d_x, seed, data)
        The ``build_problem`` outputs, the seed, and the ``ExperimentData``.
    """
    sim, prior, theta_true, d_x = exp_mod.build_problem(cfg)
    seed = int(cfg.seed_offset + rep)
    set_seed(seed)
    data = exp_mod.generate_data(sim, prior, theta_true, d_x, cfg)
    return sim, prior, theta_true, d_x, seed, data


def config_subset(cfg: DictConfig, keys: list[str]) -> dict:
    """Pick the given config keys into a plain (picklable) dict; missing keys are skipped."""
    return {key: cfg[key] for key in keys if key in cfg}


def make_bundle(
    experiment: str,
    rep: int,
    seed: int,
    cfg: DictConfig,
    prior,
    theta_true: torch.Tensor,
    d_x: int,
    data,
    *,
    config_keys: list[str],
    meta_extra: dict | None = None,
    **extra,
) -> dict:
    """Assemble the standard bundle dict from one repetition's data.

    ``extra`` adds further top-level entries (e.g. SIR's ``x_obs_under``);
    ``meta_extra`` adds keys to ``meta``.
    """
    bundle = {
        "experiment": experiment,
        "rep": rep,
        "seed": seed,
        "theta_true": theta_true,
        "prior_mean": prior.loc,
        "prior_cov": prior.covariance_matrix,
        "theta": data.theta,
        "x_sim": data.x_sim,
        "x_obs_mis": data.x_obs_mis,
        "x_obs": data.x_obs,
        "d_x": d_x,
        "d_theta": theta_true.shape[0],
        "config": config_subset(cfg, config_keys),
    }
    bundle.update(extra)
    meta = {
        "source": f"generated from scratch (configs/{experiment}.yaml)",
        "num_samples": int(cfg.num_samples),
        "n_obs": int(cfg.n_obs),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if meta_extra:
        meta.update(meta_extra)
    bundle["meta"] = meta
    return bundle


def save_bundle(path: Path, bundle: dict) -> None:
    """Pickle ``bundle`` to ``path`` (creating parent dirs) and print its size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def cli_main(make_bundle_fn: Callable[..., Path]) -> None:
    """Shared ``__main__`` body: parse ``--force`` and build the bundle."""
    parser = argparse.ArgumentParser(
        description=f"Build the test-data bundle: {make_bundle_fn.__name__}")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the existing bundle")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    make_bundle_fn(force=args.force)
    print(f"done in {time.time() - t0:.1f}s")
