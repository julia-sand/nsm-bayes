"""Test-data bundle builders for the ``examples/experimental/`` notebooks.

Each ``make_test_data_*`` module generates, from scratch, one capped-size
data bundle (``.pkl``) plus a pre-trained surrogate checkpoint (``.pt``) so
the notebooks run quickly without re-training or touching ``data/``.  The
bundles are written to ``examples/experimental/data/`` next to the notebooks
that load them.  Run them as modules from the repo root, e.g.
``python -m nsm_bayes.experiments.test_data.make_test_data_gnk``.
"""
