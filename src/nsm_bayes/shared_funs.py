
def make_nle_logprob(estimator):
    """
    Returns a callable f(x, theta) -> Tensor of shape (1,)
    that is compatible with cache_perx_sm_losses:
      - accepts x with shape (d_x,) or (1, d_x)
      - accepts theta with shape (d_theta,) or (1, d_theta)
      - handles device/dtype
      - reshapes to the (sample_dim, batch_dim, d_x) convention used by SBI's NFlowsFlow
    """
    # pick device/dtype from the estimator
    p = next(estimator.parameters())
    dev, dt = p.device, p.dtype

    def f(x, theta):
        # ensure 1D event shapes
        if x.ndim == 2:            # (1, d_x)
            x_row = x.reshape(-1)
        else:                       # (d_x,)
            x_row = x.reshape(-1)
        if theta.ndim == 2:         # (1, d_theta)
            th_row = theta.reshape(-1)
        else:                       # (d_theta,)
            th_row = theta.reshape(-1)

        # reshape to (sample_dim=1, batch_dim=1, d_x)
        x_b  = x_row.to(device=dev, dtype=dt).reshape(1, 1, -1).contiguous()
        th_b = th_row.to(device=dev, dtype=dt).reshape(1, -1).contiguous()

        # NFlowsFlow.log_prob returns shape (sample_dim, batch_dim) = (1,1)
        out = estimator.log_prob(x_b, th_b).reshape(-1)  # -> (1,)
        return out

    return f
