"""Robust weighting functions for neural score-matching Bayes.

The weighted score-matching loss uses a weight ``w(x)`` that decays for
observations far from the bulk of the data, measured by the Mahalanobis
distance ``r(x)^2 = (x - mu_hat)^T Sigma_inv (x - mu_hat)``. Every function
returns the *squared* weight ``w2 = w(x)^2`` together with its gradient
``grad_w2`` with respect to ``x``.

All public functions accept data of shape ``(n, d_x)`` and return ``w2``
of shape ``(n,)`` and ``grad_w2`` of shape ``(n, d_x)``. A single
observation of shape ``(d_x,)`` is treated as a batch of one, so the
outputs have shapes ``(1,)`` and ``(1, d_x)``.
"""

import torch

__all__ = [
    "imq_weight",
    "huber_weight",
    "unweighted",
    "weight_function",
]


def _squeeze(x, w2, grad_w2):
    """Squeeze batched outputs back to single-row shape.

    Parameters
    ----------
    x : torch.Tensor
        Data after :func:`_as_batch`; always 2-D in practice, so the
        outputs are returned unchanged.
    w2, grad_w2 : torch.Tensor
        Batched outputs.

    Returns
    -------
    w2, grad_w2 : torch.Tensor
        The outputs, squeezed along the first axis if `x` is 1-D.
    """
    if x.ndim == 1:
        return w2.squeeze(0), grad_w2.squeeze(0)
    return w2, grad_w2

def _as_batch(x):
    """Promote a single observation to a batch of one.

    Parameters
    ----------
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.

    Returns
    -------
    torch.Tensor
        ``(1, d_x)`` for 1-D input, otherwise `x` unchanged.
    """
    return x.unsqueeze(0) if x.ndim == 1 else x


def imq_weight(x, mu_hat, Sigma_inv, c=1, zeta=1):
    r"""Squared inverse-multiquadric (IMQ) weight and its gradient.

    .. math::

        w^2(x) = \left(1 + r(x)^2 / c\right)^{-2/\zeta}

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
    w2 : torch.Tensor
        Squared weights of shape ``(n,)``.
    grad_w2 : torch.Tensor
        Analytic gradient of `w2` with respect to `x`, shape ``(n, d_x)``.

    Notes
    -----
    A single observation of shape ``(d_x,)`` is promoted to ``(1, d_x)``
    and the outputs keep that leading axis (``(1,)`` and ``(1, d_x)``).
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
    """Constant weight ``w(x) = 1``, giving the standard score-matching loss.

    Parameters
    ----------
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.
    *args, **kwargs
        Ignored; accepted so all weight functions share one signature.

    Returns
    -------
    w2 : torch.Tensor
        Ones of shape ``(n,)`` (``(1,)`` for a single observation).
    grad_w2 : torch.Tensor
        Zeros of shape ``(n, d_x)`` (``(1, d_x)`` for a single
        observation).
    """
    x = _as_batch(x)
    return torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device), torch.zeros_like(x)


def huber_weight(x, mu_hat, Sigma_inv, c=1.0, eps=1e-12):
    """Squared Huber-type weight and its gradient.

    The weight is ``w(x) = min(1, c / r(x))``, so

    * ``w2 = 1`` and ``grad_w2 = 0`` if ``r <= c``;
    * ``w2 = c^2 / r^2`` otherwise.

    Parameters
    ----------
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.
    mu_hat : torch.Tensor
        Robust location estimate of shape ``(d_x,)``.
    Sigma_inv : torch.Tensor
        Inverse robust covariance of shape ``(d_x, d_x)``.
    c : float, default 1.0
        Radius (in Mahalanobis units) beyond which observations are
        down-weighted.
    eps : float, default 1e-12
        Currently unused.

    Returns
    -------
    w2 : torch.Tensor
        Squared weights of shape ``(n,)`` (``(1,)`` for a single
        observation).
    grad_w2 : torch.Tensor
        Gradient term of shape ``(n, d_x)`` (``(1, d_x)`` for a single
        observation); see Notes.

    Notes
    -----
    Outside the radius the returned gradient is
    ``-2 c^2 Sigma_inv (x - mu_hat) / r^2``. The analytic gradient of
    ``c^2 / r^2`` is ``-2 c^2 Sigma_inv (x - mu_hat) / r^4``, so the two
    differ by a factor ``r^2``.
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
    """Evaluate a weight function selected by name.

    Parameters
    ----------
    name : {"imq", "huber", "none"}
        Weight function (case-insensitive): :func:`imq_weight`,
        :func:`huber_weight` or :func:`unweighted`.
    x : torch.Tensor
        Data of shape ``(d_x,)`` or ``(n, d_x)``.
    *args, **kwargs
        Passed on to the weight function, typically ``mu_hat``,
        ``Sigma_inv`` and optionally ``c``.

    Returns
    -------
    w2 : torch.Tensor
        Squared weights.
    grad_w2 : torch.Tensor
        Gradients of the squared weights with respect to `x`.

    Raises
    ------
    ValueError
        If `name` is not a known weight function.
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
