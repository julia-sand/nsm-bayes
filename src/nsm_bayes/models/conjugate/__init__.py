"""Conjugate NSM-Bayes: neural exponential-family surrogate and its posterior.

Modules:

* ``conj_model``: :class:`ConjugateModel`, the surrogate ``theta^T T_phi(x) + b_phi(x)``.
* ``conj_train``: Score-matching training loop for the surrogate networks.
* ``conj_posterior``: Closed-form Gaussian generalised posterior.
* ``tphi_net``, ``bphi_net``: The two networks.
"""
from nsm_bayes.models.conjugate.conj_model import ConjugateModel

__all__ = ["ConjugateModel"]
