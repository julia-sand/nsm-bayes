"""Normalising-flow method for the g-and-k experiment (Hydra entry point)."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from nsm_bayes.experiments import common, gnk


@hydra.main(version_base=None, config_path="../../configs", config_name="gnk")
def run_gnk(cfg: DictConfig):
    """Hydra entry point for the g-and-k experiment."""
    save_dir = Path(get_original_cwd()) / "data" / cfg.experiment_name / "general"
    common.run_experiment(cfg, gnk, common.run_normflow_methods, save_dir)


if __name__ == "__main__":
    run_gnk()
