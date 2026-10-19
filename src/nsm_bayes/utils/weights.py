"""Robust weighting functions for neural score-matching Bayes.

All public functions accept data of shape ``(d_x,)`` (single
observation) or ``(n, d_x)`` (batch of observations) and return
matching output shapes.
"""

import torch

__all__ = [
    "imq_weight",
    "huber_weight",
    "unweighted",
    "weight_function",
]


def _squeeze(x, w2, grad_w2):
    """Squeeze batched outputs back to row shape if the input was 1-D."""
    if x.ndim == 1:
        return w2.squeeze(0), grad_w2.squeeze(0)
    return w2, grad_w2

def _as_batch(x):
    """Promote a (d_x,) row to (1, d_x)."""
    return x.unsqueeze(0) if x.ndim == 1 else x


def imq_weight(x, mu_hat, Sigma_inv, c=1, zeta=1):
    """Compute the squared inverse-multiquadric weight ``w_IMQ(x)^2``.

    Parameters
    ----------
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.
    mu_hat : torch.Tensor
        Robust location estimate of shape ``(d_x,)``.
    Sigma_inv : torch.Tensor
        Inverse robust covariance of shape ``(d_x, d_x)``.
    c : float, default 1
        Scale parameter.
    zeta : float, default 1
        Sharpness parameter.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``w2`` of shape ``(n,)`` (scalar for a ``(d_x,)`` input) and
        the analytic gradient ``grad_w2`` of shape ``(n, d_x)``
        (``(d_x,)`` for a row input).
    """
    x = _as_batch(x)

    d_x = x.shape[-1]
    diff = x - mu_hat.reshape(1, d_x)
    Sd = diff @ Sigma_inv.reshape(d_x, d_x)
    Sd = Sd.reshape(x.shape)
    norm_sq = torch.sum(Sd * diff, dim=-1)

    denom = 1 + norm_sq / c 
    w2 = 1.0 / (denom ** (2 / zeta))

    coef = (-(4.0 / (zeta * c)) * denom ** (-(2 / zeta) - 1))
    grad_w2 = coef.unsqueeze(-1) * Sd

    return _squeeze(x, w2, grad_w2)


def unweighted(x, *args, **kwargs):
    """Constant weight ``w(x) = 1`` (``w2 = 1``, ``grad_w2 = 0``).

    Parameters
    ----------
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.
    *args, **kwargs : any
        Additional arguments (ignored).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``w2`` and ``grad_w2`` matching the shape convention of ``x``.
    """
    x = _as_batch(x)
    return torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device), torch.zeros_like(x)


def huber_weight(x, mu_hat, Sigma_inv, c=1.0, eps=1e-12):
    """Compute the squared Huber-type weight.

        w(x)    = min(1, c / r),            r = ||x - mu||_{Sigma_inv}
        w2      = 1                         if r <= c
                  c^2 / r^2                 otherwise
        grad_w2 = 0                         if r <= c
                  -2 c^2 Sigma_inv (x - mu) / r^4   otherwise

    Parameters
    ----------
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.
    mu_hat : torch.Tensor
        Robust location estimate of shape ``(d_x,)``.
    Sigma_inv : torch.Tensor
        Inverse robust covariance of shape ``(d_x, d_x)``.
    c : float, default 1.0
        Huber scale.
    c : float, default 1e-12
        small regularisation to prevent errors near 0.
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``w2`` and ``grad_w2`` matching the shape convention of ``x``.
    """
    x = _as_batch(x)

    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)

    diff = x - mu_hat
    Sd = diff @ Sigma_inv
    r2 = (Sd * diff).sum(-1)

    inside = r2 <= c ** 2                   # equivalent to r <= c

    w2 = torch.where(inside, torch.ones_like(r2), c ** 2 / r2)
    grad_outside = -2.0 * c ** 2 / r2 
    grad_w2 = torch.where(
        inside.unsqueeze(-1), torch.zeros_like(Sd), grad_outside.unsqueeze(-1) * Sd
    )

    return _squeeze(x, w2, grad_w2)


def weight_function(name, x, *args, **kwargs):
    """Dispatch to a named weight function.

    Parameters
    ----------
    name : str
        Name of the weight function ('imq', 'huber', 'none').
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.
    *args, **kwargs : any
        Additional arguments (``mu_hat``, ``Sigma_inv``, ``c``, ...)
        passed to the weight function.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(w2, grad_w2)`` matching the shape convention of ``x``.
    """
    WEIGHT_FUNCTIONS = {
        "imq": imq_weight,
        "huber": huber_weight,
        "none": unweighted,
    }

    try:
        fn = WEIGHT_FUNCTIONS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown weight '{name}'. Choose from {list(WEIGHT_FUNCTIONS)}."
        ) from None

    return fn(x, *args, **kwargs)
