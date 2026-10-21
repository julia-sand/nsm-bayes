import torch

from nsm_bayes.utils.linalg import sigma_inverse
from nsm_bayes.utils.statistics import mahalanobis_sq

def mahalanobis_sq(theta_samples: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor):
    xc = theta_samples - mu
    return torch.sum(xc @ Sigma_inv * xc, dim=1)  # (M,)


def multinomial_bootstrap_counts(n: int, B: int, device=None, dtype=None):
    # Draw B multinomial(n, p_i=1/n) vectors of counts (B, n)
    p = torch.full((n,), 1.0/n, device=device, dtype=dtype or torch.float32)
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
    """Returns weights W of shape (B, M) for each bootstrap b and sample m."""
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
        Non-negative weights of shape ``(M,)``.
    eps : float, optional
        Small constant added to denominators for numerical stability.
        Default is 1e-12.

    Returns
    -------
    mu : torch.Tensor
        Weighted mean of shape ``(d,)``.
    Sigma : torch.Tensor
        Weighted covariance of shape ``(d, d)``.
    """
    w = w.view(-1, 1)
    mu = (w * theta_samples).sum(dim=0)
    xc = theta_samples - mu
    # weighted covariance (posterior convention: no Bessel correction)
    Sigma = (w * xc).T @ xc / (w.sum() + eps)

    return mu, Sigma


def weighted_quantile(values: torch.Tensor, w: torch.Tensor, q: float, *, eps: float = 1e-12):
    """Weighted quantile in [0,1] for 1D tensor 'values' with nonnegative weights 'w'."""
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
    
    #w = weights_row / (weights_row.sum() + eps)
    mu, Sigma = weighted_mean_cov(theta_samples,weights_row)# w)
    #Sigma_inv = sigma_inverse(Sigma)
    d2 = mahalanobis_sq(theta_samples, mu, Sigma)
    thresh = weighted_quantile(d2, weights_row, 1.0 - alpha)
    d2_hat = mahalanobis_sq(theta_hat, mu, Sigma)
    #(theta_hat - mu).unsqueeze(0) @ Sigma_inv @ (theta_hat - mu).unsqueeze(1)
    return int(d2_hat.item() <= thresh.item())
