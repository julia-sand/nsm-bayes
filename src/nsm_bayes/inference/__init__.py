"""Posterior calibration and sampling routines.

* :mod:`nsm_bayes.inference.beta_calibration_loop` holds the shared
  stochastic-approximation loop over ``log beta``
  (:func:`~nsm_bayes.inference.beta_calibration_loop.calibrate_beta`);
  the method-specific steps live in the posterior classes'
  ``prepare_calibration`` / ``estimate_coverage`` methods.
* :mod:`nsm_bayes.inference.bootstrap` holds the importance-reweighted
  bootstrap helpers used by the general calibration.
* :mod:`nsm_bayes.inference.slice_sampler` holds slice-sampling for the
  normalising-flow posterior.
"""
