"""Importance-reweighted bootstrap helpers for beta calibration.

These functions support :func:`nsm_bayes.inference.general.calibrate.calibrate_beta`.
Instead of re-running MCMC for every bootstrap resample and every candidate
``beta``, posterior samples drawn once at a base value ``beta_base`` are
importance-reweighted to approximate the generalised posterior at a new
``beta`` on a bootstrap resample of the data.  A bootstrap resample is
represented by a vector of multinomial counts over the ``n`` observations.
"""
import torch

from nsm_bayes.utils.statistics import mahalanobis_sq


def multinomial_bootstrap_counts(n: int, B: int, device=None, dtype=None):
    """Draw multinomial bootstrap count vectors.

    Each row is a draw from ``Multinomial(n, p)`` with ``p_i = 1 / n``, i.e.
    the number of times each of the ``n`` observations appears in one
    bootstrap resample of size ``n``.

    Parameters
    ----------
    n : int
        Number of observations (and size of each resample).
    B : int
        Number of bootstrap resamples.
    device : torch.device or str, optional
        Device on which the probabilities, and hence the counts, are created.
    dtype : torch.dtype, optional
        Dtype of the probability vector. Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Counts of shape ``(B, n)`` with dtype ``torch.float32``; each row
        sums to ``n``.
    """
    # Draw B multinomial(n, p_i=1/n) vectors of counts (B, n)
    p =torch.full((n,), 1.0/n, device=device, dtype=dtype or torch.float32)
    counts = torch.distributions.Multinomial(total_count=n, probs=p).sample((B,))
    return counts.to(dtype=torch.float32)


def reweight_for_bootstrap_beta(
    ell: torch.Tensor,        # (M, n)
    L_orig: torch.Tensor,     # (M,)
    counts: torch.Tensor,     # (B, n)
    beta_new: float,
    beta_base: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Self-normalised importance weights for bootstrap resamples at a new beta.

    The samples ``theta_m`` were drawn from the generalised posterior at
    ``beta_base`` on the original data.  For bootstrap resample ``b`` the
    target is the generalised posterior at ``beta_new`` on that resample, so
    the prior cancels and the unnormalised log-weight is

    .. math::

        \log \tilde W_{bm} = -\beta_{new}\, n\, L_b(\theta_m)
                             + \beta_{base}\, n\, L(\theta_m),

    where :math:`L_b(\theta_m) = n^{-1} \sum_i c_{bi}\, \ell_{mi}` is the
    mean loss on the resample and :math:`L(\theta_m)` is ``L_orig[m]``.

    Parameters
    ----------
    ell : torch.Tensor
        Per-observation losses ``ell[m, i] = l(theta_m; x_i)`` of shape
        ``(M, n)``.
    L_orig : torch.Tensor
        Mean loss on the original data for each sample, shape ``(M,)``.
    counts : torch.Tensor
        Bootstrap counts of shape ``(B, n)``, e.g. from
        :func:`multinomial_bootstrap_counts`. Moved to the device and dtype
        of `ell`.
    beta_new : float
        Target value of ``beta``.
    beta_base : float
        Value of ``beta`` the samples were drawn at.
    eps : float, default 1e-12
        Added to the normaliser to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Weights ``W`` of shape ``(B, M)``; each row is non-negative and sums
        to (approximately) one.

    Notes
    -----
    The maximum log-weight of each row is subtracted before exponentiating,
    so the computation is stable for large ``n`` or large losses.
    """
    M, n = ell.shape
    B = counts.shape[0]

    # >>> ensure same device & dtype as ell
    counts = counts.to(device=ell.device, dtype=ell.dtype)

    # (B, M) = (B, n) @ (n, M) / n
    L_b = (counts @ ell.T) / n

    expo = - (beta_new * n) * L_b + (beta_base * n) * L_orig.unsqueeze(0)
    expo = expo - expo.max(dim=1, keepdim=True).values
    W = torch.exp(expo)
    W = W / (W.sum(dim=1, keepdim=True) + eps)
    return W


def weighted_mean_cov(
    theta_samples: torch.Tensor,
    w: torch.Tensor,
    *,
    eps: float = 1e-12,
):
    """Weighted mean and covariance of samples.

    Parameters
    ----------
    theta_samples : torch.Tensor
        Samples of shape ``(M, d)``.
    w : torch.Tensor
        Non-negative weights of shape ``(M,)``. They are assumed to sum to
        one: the mean is ``sum_m w_m theta_m`` without renormalisation.
    eps : float, default 1e-12
        Added to the covariance denominator for numerical stability.

    Returns
    -------
    mu : torch.Tensor
        Weighted mean of shape ``(d,)``.
    Sigma : torch.Tensor
        Weighted covariance of shape ``(d, d)``, normalised by ``sum(w)``
        (no Bessel correction).
    """
    w = w.view(-1, 1)
    mu = (w * theta_samples).sum(dim=0)
    xc = theta_samples - mu
    # weighted covariance (posterior convention: no Bessel correction)
    Sigma = (w * xc).T @ xc / (w.sum() + eps)

    return mu, Sigma


def weighted_quantile(values: torch.Tensor, w: torch.Tensor, q: float, *, eps: float = 1e-12):
    """Weighted quantile of a 1-D tensor.

    Parameters
    ----------
    values : torch.Tensor
        Values of shape ``(M,)``.
    w : torch.Tensor
        Non-negative weights of shape ``(M,)``; need not be normalised.
    q : float
        Quantile level in ``[0, 1]``.
    eps : float, default 1e-12
        Added to the total weight when normalising the cumulative sum.

    Returns
    -------
    torch.Tensor
        Scalar tensor: the smallest value whose cumulative normalised weight
        is at least `q`.
    """
    v, idx = torch.sort(values)
    w_sorted = w[idx]
    cdf = torch.cumsum(w_sorted, dim=0)
    cdf = cdf / (cdf[-1] + eps)
    return v[torch.searchsorted(cdf, torch.tensor(q, device=v.device))]


def bootstrap_coverage_indicator(
        theta_samples: torch.Tensor,   # (M, d)
        weights_row: torch.Tensor,     # (M,)
        theta_hat: torch.Tensor,       # (d,)
        alpha: float,
        ) -> int:
    """Check whether a point lies in a weighted ``1 - alpha`` credible ellipsoid.

    The ellipsoid is centred at the weighted mean of `theta_samples` with
    shape given by their weighted covariance; its radius is the weighted
    ``1 - alpha`` quantile of the samples' own squared Mahalanobis distances
    (an empirical threshold rather than a chi-squared one).

    Parameters
    ----------
    theta_samples : torch.Tensor
        Posterior samples of shape ``(M, d)``.
    weights_row : torch.Tensor
        Importance weights of shape ``(M,)`` for one bootstrap resample,
        e.g. one row of :func:`reweight_for_bootstrap_beta`.
    theta_hat : torch.Tensor
        Point to test, shape ``(d,)``.
    alpha : float
        Miscoverage level; the region has nominal level ``1 - alpha``.

    Returns
    -------
    int
        ``1`` if `theta_hat` lies inside the region, ``0`` otherwise.
    """
    #w = weights_row / (weights_row.sum() + eps)
    mu, Sigma = weighted_mean_cov(theta_samples,weights_row)# w)
    d2 = mahalanobis_sq(theta_samples, mu, Sigma)
    thresh = weighted_quantile(d2, weights_row, 1.0 - alpha)
    d2_hat = mahalanobis_sq(theta_hat, mu, Sigma)
    return int(d2_hat.item() <= thresh.item())
