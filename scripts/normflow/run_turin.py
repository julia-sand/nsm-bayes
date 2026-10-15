"""Normalising-flow method for the Turin channel-model experiment.

Hydra entry point; see :mod:`nsm_bayes.experiments.turin` for the
experiment-specific data generation.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from nsm_bayes.experiments import common, turin


@hydra.main(version_base=None, config_path="../../configs", config_name="turin")
def run_turin(cfg: DictConfig):
    """Hydra entry point for the Turin channel-model experiment."""
    save_dir = (
        Path(get_original_cwd()) / "data" / cfg.experiment_name
        / f"eps={cfg.epsilon}" / f"de={cfg.density_estimator}"
    )
    common.run_experiment(cfg, turin, common.run_normflow_methods, save_dir)


if __name__ == "__main__":
    run_turin()
