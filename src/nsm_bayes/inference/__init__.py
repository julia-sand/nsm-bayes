"""Posterior calibration and sampling routines.

* :mod:`nsm_bayes.inference.beta_calibration_loop` holds the shared
  stochastic-approximation loop over ``log beta``
  (:func:`~nsm_bayes.inference.beta_calibration_loop.calibrate_beta`);
  the method-specific steps live in the posterior classes'
  ``prepare_calibration`` / ``estimate_coverage`` methods.
* :mod:`nsm_bayes.inference.beta_update_rules` holds the pluggable update
  rules for the calibration loop (GPC, bisection, grid search).
* :mod:`nsm_bayes.inference.beta_bo_calibration` holds the Bayesian-optimisation
  beta calibration via a GP on a coverage grid
  (:func:`~nsm_bayes.inference.beta_bo_calibration.optimize_beta_bo`).
* :mod:`nsm_bayes.inference.bootstrap` holds the importance-reweighted
  bootstrap helpers used by the general calibration.
"""
