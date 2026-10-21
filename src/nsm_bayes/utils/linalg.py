"""Linear-algebra and automatic-differentiation helpers.

The derivative helpers take a single observation ``xi`` of shape ``(d_x,)``
and a network ``fn`` that maps a batch ``(1, d_x)`` to ``(1, d_out)`` (or to
``(1,)`` for scalar networks such as ``b_phi``). They are written for use
inside :func:`torch.func.vmap` over observations.
"""
from __future__ import annotations

import torch
from torch.func import jacfwd, jacrev


def safe_cholesky(Sigma: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
    """Cholesky factor of a covariance matrix, adding jitter only if needed.

    Parameters
    ----------
    Sigma : torch.Tensor
        Symmetric matrix of shape ``(d, d)``.
    max_tries : int, default 6
        Number of jittered attempts after the first, unjittered one. The
        jitter starts at ``1e-8`` and grows tenfold on each attempt.

    Returns
    -------
    torch.Tensor
        Lower-triangular ``L`` with ``L @ L.T == Sigma + jitter * I``, where
        ``jitter`` is ``0`` if `Sigma` is already positive definite.

    Raises
    ------
    RuntimeError
        If the matrix is not positive definite even with the largest jitter.
    """

    L, info = torch.linalg.cholesky_ex(Sigma)
    if info == 0:
        return L
    
    eye = torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype)

    jitter = 1e-8 #* scale
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(Sigma + jitter * eye)
        if info == 0:
            return L
        jitter *= 10
    raise RuntimeError(f"Covariance not positive definite even with jitter {jitter:.2e}")

def sigma_inverse(Sigma_hat: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Invert a variance or covariance matrix.

    Parameters
    ----------
    Sigma_hat : torch.Tensor
        A single variance (any tensor with one element) or a covariance
        matrix of shape ``(d, d)``.
    jitter : float, default 1e-6
        Currently unused (see Notes); only appears in the error message.

    Returns
    -------
    torch.Tensor
        ``1 / Sigma_hat`` with shape ``(1, 1)`` for a single variance;
        otherwise the ``(d, d)`` inverse computed from a Cholesky factor.

    Raises
    ------
    ValueError
        If a single variance is not finite or is non-positive.
    RuntimeError
        From :func:`safe_cholesky`, if a matrix is not positive definite
        even after jittering.

    Notes
    -----
    No ``jitter * I`` is added unconditionally. A matrix input is factorised
    with :func:`safe_cholesky`, which adds a diagonal jitter (from ``1e-8``
    upwards) only if the unjittered factorisation fails. A scalar input is
    inverted exactly.
    """
    if Sigma_hat.numel() == 1:   # scalar variance (d_x = 1)
        var = Sigma_hat.reshape(1, 1)# + jitter
        if not torch.isfinite(var).all():
            raise ValueError(
                f"Variance estimate is not finite ({var.item()}); "
                "check the input data."
            )
        if var.item() <= 0:
            raise ValueError(
                f"Variance estimate {Sigma_hat.item():.3e} plus jitter "
                f"{jitter:.3e} is non-positive. "
                "Data may be constant or the outlier proportion too high."
            )
        return 1.0 / var

    # otherwise sigma is a matrix, find the inverse
    L = safe_cholesky(Sigma_hat) #adds jitter here
    return torch.cholesky_inverse(L)

def _single(fn):
    """Turn a batched function into a single-observation function.

    Parameters
    ----------
    fn : callable
        Maps ``(1, d_x) -> (1, d_out)`` (or ``(1,)``).

    Returns
    -------
    callable
        Maps ``(d_x,) -> (d_out,)`` (or a scalar).
    """
    return lambda x: fn(x.unsqueeze(0)).squeeze(0)

def jacobian_with_spare(xi, fn):
    """Jacobian of `fn` at `xi`, returned twice.

    Used with ``jacfwd(..., has_aux=True)``: the first output is
    differentiated again (giving the Hessian) and the second is passed
    through as the auxiliary value (giving the Jacobian), so the Jacobian
    is computed only once.

    Parameters
    ----------
    xi : torch.Tensor
        Single observation of shape ``(d_x,)``.
    fn : callable
        Batched network, e.g. ``model.T_phi`` or ``model.b_phi``.

    Returns
    -------
    J, J : tuple of torch.Tensor
        Two references to the Jacobian, of shape ``(d_out, d_x)``, or
        ``(d_x,)`` for a scalar-valued `fn`.
    """
    J = jacrev(_single(fn))(xi)                              # (d_out, d_x)
    return J, J

def jacobian(xi, fn):
    """Jacobian of `fn` at a single observation (reverse mode).

    Parameters
    ----------
    xi : torch.Tensor
        Single observation of shape ``(d_x,)``.
    fn : callable
        Batched network, ``(1, d_x) -> (1, d_out)`` or ``(1,)``.

    Returns
    -------
    torch.Tensor
        Jacobian of shape ``(d_out, d_x)``, or the gradient ``(d_x,)`` for a
        scalar-valued `fn`.
    """
    J = jacrev(_single(fn))(xi)
    return J


def jacobian_and_hessian(xi, fn):
    """Jacobian and Hessian of `fn` at a single observation.

    Computed with one forward-over-reverse pass.

    Parameters
    ----------
    xi : torch.Tensor
        Single observation of shape ``(d_x,)``.
    fn : callable
        Batched network, ``(1, d_x) -> (1, d_out)`` or ``(1,)``.

    Returns
    -------
    J : torch.Tensor
        Jacobian of shape ``(d_out, d_x)`` (``(d_x,)`` for scalar `fn`).
    H : torch.Tensor
        Hessian of shape ``(d_out, d_x, d_x)`` (``(d_x, d_x)`` for scalar
        `fn`).
    """
    H, J = jacfwd(jacobian_with_spare, has_aux=True)(xi, fn)
    return J, H

def jacobian_and_laplacian(xi, fn):
    """Jacobian and Laplacian of `fn` at a single observation.

    Parameters
    ----------
    xi : torch.Tensor
        Single observation of shape ``(d_x,)``.
    fn : callable
        Batched network, ``(1, d_x) -> (1, d_out)`` or ``(1,)``.

    Returns
    -------
    J : torch.Tensor
        Jacobian of shape ``(d_out, d_x)`` (``(d_x,)`` for scalar `fn`).
    lap : torch.Tensor
        Laplacian (trace of the Hessian over ``x``) of each output, shape
        ``(d_out,)`` (a scalar for scalar `fn`).
    """
    J, H = jacobian_and_hessian(xi, fn)
    return J, H.diagonal(dim1=-2, dim2=-1).sum(-1)

def get_cov_term_single(xi, fn, Jac=None):
    """Outer product ``J J^T`` of the Jacobian of `fn` at one observation.

    Parameters
    ----------
    xi : torch.Tensor
        Single observation of shape ``(d_x,)``.
    fn : callable
        Batched network, typically ``model.T_phi``.
    Jac : torch.Tensor, optional
        Precomputed Jacobian of shape ``(d_out, d_x)``; computed if omitted.

    Returns
    -------
    torch.Tensor
        Matrix of shape ``(d_out, d_out)``; the per-observation term of
        ``A`` in the conjugate posterior.
    """
    #returns a copy of the jacobian for reuse
    if Jac is None:
        Jac = jacobian(xi, fn)

    return Jac @ Jac.T

# --- Non-Divergence Term (h_n) ---
def get_non_div_term_single(xi, fn1, fn2, Jac1=None, Jac2=None):
    """Product of two Jacobians at one observation, ``J_1 @ J_2``.

    With ``fn1 = T_phi`` and ``fn2 = b_phi`` this is ``J_T(x) grad b(x)``,
    the unweighted non-divergence term of ``B`` in the conjugate posterior.

    Parameters
    ----------
    xi : torch.Tensor
        Single observation of shape ``(d_x,)``.
    fn1 : callable
        Batched network with Jacobian of shape ``(d_theta, d_x)``.
    fn2 : callable
        Batched scalar network with gradient of shape ``(d_x,)``.
    Jac1, Jac2 : torch.Tensor, optional
        Precomputed Jacobians; computed if omitted.

    Returns
    -------
    torch.Tensor
        Vector of shape ``(d_theta,)``.
    """

    if Jac1 is None:
        Jac1 = jacobian(xi,fn1) # shape: (d_theta, d_x)
    if Jac2 is None:
        Jac2 = jacobian(xi,fn2) # shape: (d_x,)
    
    return Jac1 @ Jac2

#Divergence term
def get_div_term_single(xi,fn,w2,grad_w2):
    """Divergence term ``J_T(x) grad_w2 + w2 * lap T(x)`` at one observation.

    This is the per-observation contribution of the integration-by-parts
    term of the weighted score-matching loss to ``B`` in the conjugate
    posterior (the weight is applied here, not by the caller).

    Parameters
    ----------
    xi : torch.Tensor
        Single observation of shape ``(d_x,)``.
    fn : callable
        Batched network, typically ``model.T_phi``.
    w2 : torch.Tensor
        Squared weight at `xi` (scalar).
    grad_w2 : torch.Tensor
        Gradient of the squared weight at `xi`, shape ``(d_x,)``.

    Returns
    -------
    torch.Tensor
        Vector of shape ``(d_theta,)``.
    """

    # 1. Compute Laplacian of T_phi: Tr(Hess(T_phi))
    jac_T,lap_T  = jacobian_and_laplacian(xi,fn)

    #grad_w2 = grad_w2.squeeze(0)
    
    divergence = grad_w2 @ jac_T.T + w2 * lap_T
    
    return divergence