import numpy as np
from joblib import Parallel, delayed


def run_multivariate_slice_sampler_tuned(
    log_posterior_fn,
    prior,
    num_samples,
    num_chains=4,
    warmup_steps=1000,
    thin=10,
    initial_w=1.0,
    n_jobs=1,
    seed=None,
):
    """
    Parallel multivariate slice sampler with adaptive width tuning.
    """
    if seed is not None:
        np.random.seed(seed)

    d_theta = prior.event_shape[0]
    total_steps_per_chain = (warmup_steps + num_samples) * thin
    initial_thetas = prior.sample((num_chains,)).numpy()

    print(f"Running {num_chains} slice-sampler chains for {total_steps_per_chain} steps each...")

    all_chains_samples_raw = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_single_chain_multivariate_tuned)(
            i,
            log_posterior_fn,
            initial_thetas[i],
            total_steps_per_chain,
            warmup_steps,
            initial_w,
        )
        for i in range(num_chains)
    )

    print("All chains complete.")

    final_samples = []
    for chain_samples_raw in all_chains_samples_raw:
        thinned_chain = chain_samples_raw[::thin]
        # Discard the warmup samples from the *thinned* chain
        processed_chain = thinned_chain[int(warmup_steps / thin) :]
        final_samples.append(processed_chain)

    final_samples = np.vstack(final_samples)
    print(f"Collected {final_samples.shape[0]} total samples from {num_chains} chains.")
    return final_samples


def _run_single_chain_multivariate_tuned(
    chain_id,
    log_posterior_fn,
    initial_theta,
    total_steps,
    warmup_steps,
    initial_w,
    max_step_out=1000,
):
    """
    Runs one MCMC chain with adaptive width tuning.
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

        log_posterior_1d = lambda a: float(log_posterior_fn(theta_current + a * direction))

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
    """
    1-D slice sampling step with adaptive width return.
    """
    log_p_current = log_p_1d(current_val)
    if np.isneginf(log_p_current):
        print("Warning: log-posterior = −inf at current point.")
        return current_val, w

    slice_height = log_p_current - np.random.exponential(1.0)

    # Random initial bracket
    rand = np.random.rand()
    left_bound = current_val - rand * w
    right_bound = left_bound + w

    # --- Stepping-out phase with limits ---
    for _ in range(max_step_out):
        if log_p_1d(left_bound) <= slice_height or np.isneginf(log_p_1d(left_bound)):
            break
        left_bound -= w
    for _ in range(max_step_out):
        if log_p_1d(right_bound) <= slice_height or np.isneginf(log_p_1d(right_bound)):
            break
        right_bound += w

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

