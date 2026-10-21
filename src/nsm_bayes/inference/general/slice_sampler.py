"""Multivariate slice sampler with adaptive width tuning for NSM-Bayes.

Each step picks a uniformly random direction on the unit sphere and performs
one univariate slice-sampling update (Neal, 2003) along that line, using the
stepping-out and shrinkage procedures.  During warm-up the initial bracket
width is adapted towards the widths of the final brackets with an
exponential moving average.

References
----------
.. [1] R. M. Neal, "Slice sampling", *The Annals of Statistics*, 31(3),
   705-767, 2003.
"""
import numpy as np
from joblib import Parallel, delayed


def run_multivariate_slice_sampler_tuned(
    log_posterior_fn,
    prior,
    mcmcsettings,
    initial_w=1.0,
    n_jobs=1,
    seed=None,
):
    """Run several random-direction slice-sampling chains.

    Parameters
    ----------
    log_posterior_fn : callable
        Maps a parameter vector (``numpy.ndarray`` of shape ``(d,)``) to its
        unnormalised log-posterior as a float. May return ``-inf`` outside
        the support; must not return NaN.
    prior : torch.distributions.Distribution
        Used only to draw one initial state per chain.
    mcmcsettings : MCMCSettings
        Number of chains, retained samples, warm-up steps and thinning.
    initial_w : float, default 1.0
        Initial bracket width for the univariate slice steps.
    n_jobs : int, default 1
        Number of joblib workers (threads) that run chains in parallel.
    seed : int, optional
        If given, seeds NumPy's global RNG before sampling. The initial
        states are drawn with torch's RNG and are not affected.

    Returns
    -------
    numpy.ndarray
        Post-warm-up samples from all chains stacked along the first axis,
        shape ``(num_chains * S, d)``.

    Raises
    ------
    FloatingPointError
        If `log_posterior_fn` returns NaN, or a non-finite value at a
        chain's current state.

    Notes
    -----
    Each chain runs ``(warmup_steps + num_samples) * thin`` steps. The chain
    is thinned by ``thin`` and the first ``warmup_steps // thin`` thinned
    states are discarded, so each chain contributes
    ``S = warmup_steps + num_samples - warmup_steps // thin`` samples. This
    equals ``num_samples`` only when ``thin == 1``.
    """
    if seed is not None:
        np.random.seed(seed)

    total_steps_per_chain = (mcmcsettings.warmup_steps + mcmcsettings.num_samples) * mcmcsettings.thin
    initial_thetas = prior.sample((mcmcsettings.num_chains,)).numpy()

    print(f"Running {mcmcsettings.num_chains} slice-sampler chains for {total_steps_per_chain} steps each...")

    all_chains_samples_raw = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_single_chain_multivariate_tuned)(
            i,
            log_posterior_fn,
            initial_thetas[i],
            total_steps_per_chain,
            mcmcsettings.warmup_steps,
            initial_w,
        )
        for i in range(mcmcsettings.num_chains)
    )

    print("All chains complete.")

    final_samples = []
    for chain_samples_raw in all_chains_samples_raw:
        thinned_chain = chain_samples_raw[::mcmcsettings.thin]
        # Discard the warmup samples from the *thinned* chain
        processed_chain = thinned_chain[int(mcmcsettings.warmup_steps / mcmcsettings.thin) :]
        final_samples.append(processed_chain)

    final_samples = np.vstack(final_samples)
    print(f"Collected {final_samples.shape[0]} total samples from {mcmcsettings.num_chains} chains.")
    return final_samples

def log_posterior_wrapper(a,theta_current,direction,log_posterior_fn):
    """Evaluate the log-posterior along a line through the current state.

    Parameters
    ----------
    a : float
        Offset along `direction`.
    theta_current : numpy.ndarray
        Current state of shape ``(d,)``.
    direction : numpy.ndarray
        Unit direction of shape ``(d,)``.
    log_posterior_fn : callable
        Log-posterior on the full parameter space.

    Returns
    -------
    float
        ``log_posterior_fn(theta_current + a * direction)``.

    Raises
    ------
    FloatingPointError
        If the log-posterior is NaN at that point.
    """

    v = float(log_posterior_fn(theta_current + a * direction))
    if np.isnan(v):
        raise FloatingPointError(f"log-posterior is NaN at offset a={a}")

    return v


def _run_single_chain_multivariate_tuned(
    chain_id,
    log_posterior_fn,
    initial_theta,
    total_steps,
    warmup_steps,
    initial_w,
    max_step_out=1000,
):
    """Run one random-direction slice-sampling chain.

    Parameters
    ----------
    chain_id : int
        Index used only in progress messages.
    log_posterior_fn : callable
        Log-posterior on the full parameter space.
    initial_theta : numpy.ndarray
        Starting state of shape ``(d,)``; copied, not modified.
    total_steps : int
        Total number of steps, warm-up included.
    warmup_steps : int
        Number of initial steps during which the bracket width is adapted.
    initial_w : float
        Initial bracket width.
    max_step_out : int, default 1000
        Maximum number of stepping-out expansions on each side of the
        bracket; the shrinkage phase may take up to ``10 * max_step_out``
        proposals.

    Returns
    -------
    numpy.ndarray
        Every state of the chain (no thinning or warm-up removal), shape
        ``(total_steps, d)``.

    Notes
    -----
    During warm-up the width is updated as
    ``w <- 0.95 * w + 0.05 * final_bracket_width`` after every step and is
    then frozen.
    """
    d_theta = len(initial_theta)
    theta_current = initial_theta.copy()
    samples = np.zeros((total_steps, d_theta))
    w_tuned = float(initial_w)

    print(f"  Chain {chain_id}: Starting warmup ({warmup_steps} steps)...")

    for i in range(total_steps):
        # Pick a random direction on the unit sphere
        direction = np.random.randn(d_theta)
        direction /= np.linalg.norm(direction)

        log_posterior_1d = lambda a: log_posterior_wrapper(a,theta_current,direction,log_posterior_fn)

        # One 1-D slice step along that direction
        a_new, final_bracket_width = _slice_sampler_1d_adaptive(
            log_posterior_1d, 0.0, w_tuned, max_step_out=max_step_out
        )

        theta_current += a_new * direction

        # Update width during warmup using exponential moving average
        if i < warmup_steps:
            alpha = 0.05  # smoothing factor
            w_tuned = (1 - alpha) * w_tuned + alpha * final_bracket_width

        samples[i] = theta_current

        if (i + 1) % 500 == 0 and i >= warmup_steps:
            print(f"  Chain {chain_id}: {i + 1}/{total_steps} steps complete")

    print(f"  Chain {chain_id}: tuning done, final width ≈ {w_tuned:.4f}")
    return samples


def _slice_sampler_1d_adaptive(log_p_1d, current_val, w=1.0, max_step_out=1000):
    """Perform one univariate slice-sampling step.

    Parameters
    ----------
    log_p_1d : callable
        Unnormalised log-density of a scalar argument.
    current_val : float
        Current position.
    w : float, default 1.0
        Initial bracket width, placed uniformly at random around
        `current_val`.
    max_step_out : int, default 1000
        Maximum number of stepping-out expansions on each side.

    Returns
    -------
    new_val : float
        The accepted position, or `current_val` if no proposal was accepted
        within ``10 * max_step_out`` shrinkage proposals (a warning is
        printed in that case).
    bracket_width : float
        Width of the bracket from which `new_val` was accepted, used for
        width adaptation; `w` on fallback.

    Raises
    ------
    FloatingPointError
        If the log-density at `current_val` is not finite.
    """

    log_p_current = log_p_1d(current_val)
    
    if not np.isfinite(log_p_current):
        #stops silent failures here 
        raise FloatingPointError(f"log-posterior is {log_p_current} at the current point")

    slice_height = log_p_current - np.random.exponential(1.0)

    # Random initial bracket
    rand = np.random.rand()
    left_bound = current_val - rand * w
    right_bound = left_bound + w

    log_p_left = log_p_1d(left_bound)
    log_p_right = log_p_1d(right_bound)

    # --- Stepping-out phase with limits ---
    for _ in range(max_step_out):
        if log_p_left <= slice_height or np.isneginf(log_p_left):
            break
        left_bound -= w
        log_p_left = log_p_1d(left_bound) #recompute the bound
 
    for _ in range(max_step_out):
        if log_p_right <= slice_height or np.isneginf(log_p_right):
            break
        right_bound += w
        log_p_right = log_p_1d(right_bound)

    # --- Shrinking phase ---
    for _ in range(max_step_out * 10):  # plenty of chances but bounded
        proposal_val = np.random.uniform(left_bound, right_bound)
        val = log_p_1d(proposal_val)
        if val > slice_height:
            return proposal_val, (right_bound - left_bound)
        if proposal_val < current_val:
            left_bound = proposal_val
        else:
            right_bound = proposal_val

    # Fallback if nothing accepted
    print("Warning: slice sampler failed to find valid proposal; returning current value.")
    return current_val, w

