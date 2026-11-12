"""Posterior calibration and sampling routines.

* :mod:`nsm_bayes.inference.calibrate_beta` is the shared beta
  calibration driver for both methods; the method-specific steps live in
  the posterior classes' ``prepare_calibration`` / ``estimate_coverage``.
* :mod:`nsm_bayes.inference.beta_calibration_loop` holds the
  stochastic-approximation loop over ``log beta``.
* :mod:`nsm_bayes.inference.bootstrap` holds the importance-reweighted
  bootstrap helpers used by the general calibration.

Subpackages:

* ``general``: slice-sampling for the normalising-flow posterior.
"""
