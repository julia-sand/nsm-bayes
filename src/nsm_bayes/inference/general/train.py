import torch

def compute_theta_hat_true_general(
    x_obs: torch.Tensor,                       # (n, d_x)
    q_phi_log_prob,                            # fn (x_batch, theta_batch) -> log prob
    w2: torch.Tensor,                          # (n,)
    grad_w2: torch.Tensor,                     # (n, d_x)
    theta_init: torch.Tensor,                  # (d_theta,), starting point
    num_steps: int = 200,
    lr: float = 1e-2,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute the NSM-only minimiser theta_hat_true for NSM-Bayes:

        theta_hat_true = argmin_theta L_NSM(theta; x_obs)

    using gradient-based optimisation.

    Args
    ----
    x_obs      : (n, d_x) observed data.
    q_phi_log_prob : function (x, theta) -> log q_phi(x | theta), batch-safe.
    w2, grad_w2   : outputs of weight_function_factory_batched (n,), (n, d_x).
    theta_init    : initial guess for theta (e.g. posterior mean at beta_base).
    num_steps     : number of optimisation steps.
    lr            : learning rate for Adam.
    verbose       : if True, prints loss occasionally.

    Returns
    -------
    theta_hat_true : (d_theta,) tensor (same space/scale as theta_init).
    """

    device = x_obs.device
    dtype  = x_obs.dtype
    n, d_x = x_obs.shape
    d_theta = theta_init.shape[0]

    theta = theta_init.clone().detach().to(device=device, dtype=dtype)
    theta.requires_grad_(True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    def nsm_loss_current_theta():
        # per-x NSM loss, same structure as in cache_perx_sm_losses
        def per_x_loss_for_theta(xi, w2_i, grad_w2_i):
            # log q(x | theta) for single xi, current theta
            log_q_fn = lambda x_vec: q_phi_log_prob(
                x_vec.unsqueeze(0),
                theta.unsqueeze(0)
            ).sum()

            score = jacrev(log_q_fn)(xi)      # (d_x,)
            hess  = hessian(log_q_fn)(xi)     # (d_x,d_x) or scalar

            term1 = w2_i * (score @ score)
            term2 = 2.0 * grad_w2_i @ score
            if hess.ndim == 0:
                term3 = 2.0 * w2_i * hess
            else:
                term3 = 2.0 * w2_i * torch.trace(hess)
            return term1 + term2 + term3

        # vectorise over x
        losses = vmap(
            per_x_loss_for_theta,
            in_dims=(0, 0, 0)
        )(x_obs, w2, grad_w2)  # (n,)

        return losses.mean()

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = nsm_loss_current_theta()
        loss.backward()
        optimizer.step()

        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(f"[theta_hat optimisation] step {step:4d} | L_NSM = {loss.item():.4f}")
    print("Estimated theta true: ", theta)
    return theta.detach()