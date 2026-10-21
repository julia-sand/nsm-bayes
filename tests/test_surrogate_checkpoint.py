"""Reusing a trained surrogate checkpoint found in ``save_dir``.

A second run into the same ``save_dir`` must load the saved surrogate
instead of retraining it, leave the original training cost untouched, and
still produce the downstream posterior artifacts.
"""
from __future__ import annotations

from unittest import mock

import pytest
from omegaconf import OmegaConf

from nsm_bayes.experiments import common, gnk, sir
from nsm_bayes.models.conjugate import ConjugateModel
from nsm_bayes.models.general.normflow import GeneralModel
from tests.conftest import load_tensor

# Each method runs on a problem whose pipeline works for that method.
METHODS = [
    pytest.param(gnk, "gnk_cfg", common.run_normflow_methods, GeneralModel,
                 "likelihood_estimator_full_0.pt", "cost_nle_training_0.pkl",
                 "samples_sm_0.pkl", id="normflow"),
    pytest.param(sir, "sir_cfg", common.run_conjugate_method, ConjugateModel,
                 "conjugate_nets_0.pt", "cost_sm_train_0.pkl",
                 "posterior_mean_0.pkl", id="conjugate"),
]


@pytest.mark.parametrize("problem, cfg_fixture, method, model_cls, ckpt, cost, output",
                         METHODS)
def test_rerun_reuses_checkpoint(request, tmp_path, problem, cfg_fixture, method,
                                 model_cls, ckpt, cost, output):
    cfg = OmegaConf.merge(request.getfixturevalue(cfg_fixture), {"num_repeat": 1})
    save_dir = tmp_path / "run"

    common.run_experiment(cfg, problem, method, save_dir)
    assert (save_dir / ckpt).is_file()
    first_cost = load_tensor(save_dir, cost)
    (save_dir / output).unlink()

    with mock.patch.object(model_cls, "fit", autospec=True) as fit:
        common.run_experiment(cfg, problem, method, save_dir)
    fit.assert_not_called()
    assert load_tensor(save_dir, cost) == first_cost
    assert (save_dir / output).is_file()
