"""Central dispatch from a resolved config to the experiment/method runners.

A single Hydra entry point (``scripts/run.py``) resolves the top-level
``method`` and ``experiment`` config keys with :func:`resolve` and calls the
returned run function.

The ``method`` key selects the inference method and maps to the per-method
output folder used by :func:`nsm_bayes.experiments.common.run_experiment`:

- ``normflow`` -> ``general`` (normalising-flow likelihood surrogate)
- ``conjugate`` -> ``conjugate`` (neural exponential family, closed form)

The ``experiment`` key selects the benchmark simulator
(``gnk``, ``sir``, ``sir_undercounting`` or ``turin``); the experiment module
itself is read from the config (``experiment_module``), so the run function
is experiment-agnostic except for the Turin save-directory layout, which adds
``eps=<epsilon>/de=<density_estimator>`` subfolders under
``<experiment_name>/<method>/`` to separate the multiple ``turin``
configurations that share one ``experiment_name``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nsm_bayes.experiments.common import run_conjugate_method, run_normflow_methods

__all__ = ["MethodExperiment", "resolve"]

_METHODS = ("normflow", "conjugate")
_EXPERIMENTS = ("gnk", "sir", "sir_undercounting", "turin")
_METHOD_FOLDERS = {"normflow": "general", "conjugate": "conjugate"}


@dataclass(frozen=True)
class MethodExperiment:
    """A resolved ``(method, experiment)`` pair.

    Attributes:
        method: the inference method name, either ``"normflow"`` (writes to
            the ``general`` folder) or ``"conjugate"``.
        experiment: the benchmark experiment name, one of ``"gnk"``,
            ``"sir"``, ``"sir_undercounting"``, ``"turin"``.
        run: the experiment run function
            ``(cfg, experiment_module) -> None`` that ``method`` maps to.
        method_folder: the per-method output subfolder
            (``"general"`` / ``"conjugate"``) used to build ``save_dir``.
    """

    method: str
    experiment: str
    run: Callable
    method_folder: str


def _run_normflow(cfg, experiment_module) -> None:
    """Run the normflow (general) method step for the resolved experiment."""
    run_normflow_methods(cfg, experiment_module)


def _run_conjugate(cfg, experiment_module) -> None:
    """Run the conjugate method step for the resolved experiment."""
    run_conjugate_method(cfg, experiment_module)


_RUNS = {"normflow": _run_normflow, "conjugate": _run_conjugate}


def _make_turin_run(folder: str, run_method) -> Callable:
    """Build the run function for the Turin experiments.

    Turin is the only experiment whose save directory is not simply
    ``<experiment_name>/<method>``: because its single config
    (``turin.yaml``) covers several ``epsilon`` values and density
    estimators that share the same ``experiment_name``, the results are kept
    in ``<experiment_name>/<method>/eps=<epsilon>/de=<density_estimator>``
    subfolders instead.
    """

    def _run_turin(cfg, experiment_module) -> None:
        base_dir = cfg.get("save_dir")
        save_dir = (
            f"{base_dir}/eps={cfg['epsilon']}/de={cfg['density_estimator']}"
            if base_dir is not None
            else None
        )
        run_method(cfg, experiment_module, save_dir=save_dir)

    _run_turin.__name__ = f"run_{folder}_turin"
    return _run_turin


def _run_standard(folder: str, run_method) -> Callable:
    """Build the run function for the experiments with a flat save directory."""

    def _run_standard(cfg, experiment_module) -> None:
        run_method(cfg, experiment_module)

    _run_standard.__name__ = f"run_{folder}"
    return _run_standard


def resolve(cfg) -> MethodExperiment:
    """Resolve the config's ``method`` and ``experiment`` to a run function.

    Args:
        cfg: a resolved OmegaConf config (or mapping) carrying at least
            the top-level dispatch keys ``method`` and ``experiment``.

    Returns:
        A :class:`MethodExperiment` whose ``run(cfg, experiment_module)``
        performs the full ``run_experiment`` pipeline (seed, data generation
        and save, then the method step) with the per-method save-directory
        logic for the resolved experiment.

    Raises:
        KeyError: if ``method`` or ``experiment`` is missing from ``cfg``.
        ValueError: if ``method`` is not one of
            :data:`_METHODS` or ``experiment`` is not one of
            :data:`_EXPERIMENTS`.
    """
    method = cfg["method"]
    experiment = cfg["experiment"]

    if method not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {_METHODS}."
        )
    if experiment not in _EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}; expected one of {_EXPERIMENTS}."
        )

    folder = _METHOD_FOLDERS[method]
    run_method = _RUNS[method]
    if experiment == "turin":
        run = _make_turin_run(folder, run_method)
    else:
        run = _run_standard(folder, run_method)
    return MethodExperiment(
        method=method,
        experiment=experiment,
        run=run,
        method_folder=folder,
    )
