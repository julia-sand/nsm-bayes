"""Experiment drivers shared by the ``scripts/`` entry points.

Each experiment module (``gnk``, ``sir``, ``sir_undercounting``, ``turin``)
exposes exactly two functions, ``build_problem`` and ``generate_data``;
:mod:`nsm_bayes.experiments.common` holds the repetition loop, the two
method steps (conjugate and normalising-flow) and the small saving helpers.
"""
from nsm_bayes.experiments.common import ExperimentData, run_experiment

__all__ = ["ExperimentData", "run_experiment"]
