"""Shared utilities: linear algebra, autodiff helpers, statistics and weights.

Modules:

* ``linalg``: Stable covariance inversion and Jacobian/Hessian/Laplacian helpers.
* ``statistics``: Robust and sample moments of the observed data.
* ``weights``: Robust weight functions for the score-matching loss.
* ``misc``: Seeding of the global random number generators.
"""
from nsm_bayes.utils.misc import set_seed

__all__ = [
    "set_seed",
]
