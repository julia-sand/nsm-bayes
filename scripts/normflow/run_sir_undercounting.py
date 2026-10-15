"""Normalising-flow method for the SIR undercounting experiment.

Hydra entry point; see :mod:`nsm_bayes.experiments.sir_undercounting` for
the experiment-specific data generation.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from nsm_bayes.experiments import common, sir_undercounting


@hydra.main(version_base=None, config_path="../../configs", config_name="sir_undercounting")
def run_sir_undercounting(cfg: DictConfig):
    """Hydra entry point for the SIR undercounting experiment."""
    save_dir = Path(get_original_cwd()) / "data" / cfg.experiment_name
    common.run_experiment(cfg, sir_undercounting, common.run_normflow_methods, save_dir)


if __name__ == "__main__":
    run_sir_undercounting()
