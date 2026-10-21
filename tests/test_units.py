"""Fast unit tests with known answers for the numerical building blocks.

Each expensive result (a fitted posterior, a large simulation, an MCMC run, a
calibration history, robust estimates) is computed once in a module-scoped
fixture and shared by every test that checks a property of it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.distributions import MultivariateNormal

import nsm_bayes.inference.general.calibrate as general_calibrate
from nsm_bayes.inference.bootstrap import (
    bootstrap_coverage_indicator,
    multinomial_bootstrap_counts,
    reweight_for_bootstrap_beta,
    weighted_mean_cov,
    weighted_quantile,
)
from nsm_bayes.inference.conjugate.calibrate_beta import calibrate_beta_gpc
from nsm_bayes.inference.general.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.models.conjugate.conj_model import ConjugateModel
from nsm_bayes.models.conjugate.conj_posterior import ConjugatePosterior
from nsm_bayes.models.general.mcmc_settings import MCMCSettings
from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator
from nsm_bayes.simulators.benchmark_simulators.simulators import GandKSimulator
from nsm_bayes.utils.linalg import jacobian_and_laplacian
from nsm_bayes.utils.statistics import estimate_stats


# --------------------------------------------------------------------------- #
# 6. Conjugate posterior: beta = 0 returns the prior; larger beta shrinks it   #
# --------------------------------------------------------------------------- #
BETAS = (0.0, 0.5, 5.0)


@pytest.fixture(scope="module")
def conjugate_posteriors():
    """Posterior (mean, cov) at each beta in BETAS, plus the prior.

    The nets are untrained: at beta = 0 they drop out of the formulas, and
    for beta > 0 the shrinkage property holds for any T_phi.
    """
    torch.manual_seed(0)
    d_x, d_theta = 3, 4
    prior_mean = torch.tensor([0.5, -1.0, 2.0, 0.0])
    L = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.3, 0.8, 0.0, 0.0],
                      [0.0, 0.2, 1.5, 0.0],
                      [0.1, 0.0, 0.4, 0.6]])
    prior = MultivariateNormal(prior_mean, scale_tril=L)

    model = ConjugateModel(d_x=d_x, d_theta=d_theta, hidden_dim=16)
    model.standardizer_theta.fit(prior.sample((500,)))
    model.standardizer_x.fit(torch.randn(500, d_x) * 2.0 + 1.0)

    posterior = ConjugatePosterior(model, prior.loc, prior.covariance_matrix,
                                   robust=False, weight_type="imq")
    x_obs = torch.randn(30, d_x) * 2.0 + 1.0
    with torch.no_grad():
        results = {b: posterior.calibrated_posterior(x_obs, b) for b in BETAS}
    return prior, results


def test_conjugate_beta_zero_returns_prior_mean(conjugate_posteriors):
    prior, results = conjugate_posteriors
    mu, _ = results[0.0]
    assert torch.allclose(mu, prior.loc, atol=1e-4)


def test_conjugate_beta_zero_returns_prior_covariance(conjugate_posteriors):
    prior, results = conjugate_posteriors
    _, Sigma = results[0.0]
    assert torch.allclose(Sigma, prior.covariance_matrix, atol=1e-4)


def test_conjugate_covariance_shrinks_as_beta_grows(conjugate_posteriors):
    """Precision is prior + 2*beta*A with A PSD, so covariance only shrinks."""
    _, results = conjugate_posteriors
    for b_small, b_large in zip(BETAS, BETAS[1:]):
        gap = results[b_small][1] - results[b_large][1]
        gap = 0.5 * (gap + gap.T)
        assert torch.linalg.eigvalsh(gap).min() > -1e-5, (b_small, b_large)


# --------------------------------------------------------------------------- #
# 7. Simulator identities                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def gnk_draws():
    """200k g-and-k draws at the config's theta_true = [A, log B, g, log k]."""
    theta = torch.tensor(list(OmegaConf.load("configs/gnk.yaml").theta_true),
                         dtype=torch.float32)
    torch.manual_seed(0)
    x = GandKSimulator().simulate(theta.unsqueeze(0), n=200_000).reshape(-1)
    return theta, x


def _gnk_quantile(theta: torch.Tensor, p: float) -> float:
    """Closed-form g-and-k quantile function."""
    A, B, g, k = theta[0], theta[1].exp(), theta[2], theta[3].exp()
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p))
    return float(A + B * (1 + 0.8 * torch.tanh(0.5 * g * z)) * (1 + z**2) ** k * z)


def test_gnk_median_is_A(gnk_draws):
    theta, x = gnk_draws
    assert abs(x.median().item() - theta[0].item()) < 0.02


@pytest.mark.parametrize("p", [0.1, 0.25, 0.75, 0.9])
def test_gnk_quantiles_match_closed_form(gnk_draws, p):
    theta, x = gnk_draws
    assert abs(torch.quantile(x, p).item() - _gnk_quantile(theta, p)) < 0.03


def test_sir_parameter_transforms_round_trip():
    sim = SIRSimulator()
    torch.manual_seed(0)
    theta_u = sim.prior.sample((200,))
    assert torch.allclose(sim.to_unconstrained(sim.to_constrained(theta_u)), theta_u, atol=1e-4)
    theta_c = torch.tensor([[0.6, 0.15, 0.6, 20.0]])
    assert torch.allclose(sim.to_constrained(sim.to_unconstrained(theta_c)), theta_c, atol=1e-5)


def test_sir_summary_on_hand_made_trajectory():
    sim = SIRSimulator(T=5, N=100)
    y = torch.tensor([[0.0, 2.0, 10.0, 3.0, 1.0],
                      [4.0, 0.0, 0.0, 0.0, 0.0]])
    expected = torch.tensor([[16 / 100, 2 / 4, 10 / 100],   # attack, t_peak, peak
                             [4 / 100, 0 / 4, 4 / 100]])
    assert torch.allclose(sim.summary(y), expected)


def test_sir_summary_rejects_summarised_input():
    with pytest.raises(ValueError):
        SIRSimulator(T=5, N=100).summary(torch.zeros(4, 3))


# --------------------------------------------------------------------------- #
# 8. Bootstrap helpers                                                         #
# --------------------------------------------------------------------------- #
M, D = 400, 3


@pytest.fixture(scope="module")
def theta_samples():
    torch.manual_seed(1)
    return torch.randn(M, D, dtype=torch.float64) @ torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, -0.3, 0.7]], dtype=torch.float64
    ) + torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)


@pytest.fixture(scope="module")
def uniform_w():
    return torch.full((M,), 1.0 / M, dtype=torch.float64)


def test_weighted_mean_cov_uniform_matches_sample_moments(theta_samples, uniform_w):
    mu, Sigma = weighted_mean_cov(theta_samples, uniform_w)
    assert torch.allclose(mu, theta_samples.mean(0))
    assert torch.allclose(Sigma, torch.cov(theta_samples.T, correction=0), atol=1e-10)


@pytest.mark.parametrize("q", [0.051, 0.503, 0.951])
def test_weighted_quantile_uniform_is_empirical_quantile(theta_samples, uniform_w, q):
    """The result splits the sample at q: at least q of it at or below, less than q strictly below.

    q is kept off the 1/M grid: when q*M is an integer, float rounding in the
    CDF's cumsum can move the answer up by one sample, which is harmless.
    """
    v = theta_samples[:, 0]
    r = weighted_quantile(v, uniform_w, q)
    assert (v <= r).double().mean() >= q - 1e-12
    assert (v < r).double().mean() < q


def test_multinomial_counts_shape_and_totals():
    counts = multinomial_bootstrap_counts(n=25, B=7)
    assert counts.shape == (7, 25)
    assert torch.equal(counts.sum(1), torch.full((7,), 25.0))


def test_reweight_rows_are_normalised():
    torch.manual_seed(2)
    ell = torch.rand(M, 25, dtype=torch.float64)
    counts = multinomial_bootstrap_counts(25, 6).double()
    W = reweight_for_bootstrap_beta(ell, ell.mean(1), counts, beta_new=0.3, beta_base=0.1)
    assert W.shape == (6, M) and (W >= 0).all()
    assert torch.allclose(W.sum(1), torch.ones(6, dtype=torch.float64))


def test_reweight_is_uniform_without_resampling_or_beta_change():
    """counts = 1 reproduces the original data and beta_new = beta_base keeps the target."""
    torch.manual_seed(3)
    ell = torch.rand(M, 25, dtype=torch.float64)
    counts = torch.ones(4, 25, dtype=torch.float64)
    W = reweight_for_bootstrap_beta(ell, ell.mean(1), counts, beta_new=0.7, beta_base=0.7)
    assert torch.allclose(W, torch.full_like(W, 1.0 / M))


def test_coverage_indicator_center_and_far(theta_samples, uniform_w):
    mu, _ = weighted_mean_cov(theta_samples, uniform_w)
    assert bootstrap_coverage_indicator(theta_samples, uniform_w, mu, alpha=0.05) == 1
    assert bootstrap_coverage_indicator(theta_samples, uniform_w, mu + 50.0, alpha=0.05) == 0


# --------------------------------------------------------------------------- #
# 9. Jacobian / Laplacian helpers on quadratics                                #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quadratic():
    torch.manual_seed(4)
    d = 3
    M1 = torch.randn(d, d, dtype=torch.float64)
    M2 = torch.randn(d, d, dtype=torch.float64)
    x = torch.randn(d, dtype=torch.float64)
    return M1, M2, x


def test_scalar_fn_gives_vector_score_and_scalar_laplacian(quadratic):
    """fn: (1, d) -> (1,) must give J of shape (d,) and a scalar Laplacian.

    This is the contract ``GeneralPosterior.log_q`` relies on; returning
    (1, 1) instead produced the earlier shape errors.
    """
    M1, _, x = quadratic
    fn = lambda xb: (xb @ M1 * xb).sum(-1)              # (1, d) -> (1,)
    J, lap = jacobian_and_laplacian(x, fn)
    assert J.shape == (x.shape[0],) and lap.shape == ()
    assert torch.allclose(J, (M1 + M1.T) @ x)
    assert torch.allclose(lap, torch.trace(M1 + M1.T))


def test_vector_fn_gives_jacobian_rows_and_laplacian_per_output(quadratic):
    M1, M2, x = quadratic
    fn = lambda xb: torch.stack([(xb @ M1 * xb).sum(-1), (xb @ M2 * xb).sum(-1)], dim=-1)  # (1, d) -> (1, 2)
    J, lap = jacobian_and_laplacian(x, fn)
    assert J.shape == (2, x.shape[0]) and lap.shape == (2,)
    assert torch.allclose(J, torch.stack([(M1 + M1.T) @ x, (M2 + M2.T) @ x]))
    assert torch.allclose(lap, torch.stack([torch.trace(M1 + M1.T), torch.trace(M2 + M2.T)]))


# --------------------------------------------------------------------------- #
# 10. Slice sampler recovers a known Gaussian                                  #
# --------------------------------------------------------------------------- #
TARGET_MEAN = np.array([1.0, -2.0])
TARGET_COV = np.array([[1.0, 0.6], [0.6, 2.0]])
SLICE_SETTINGS = MCMCSettings(num_samples=3000, num_chains=2, warmup_steps=300, thin=1)


@pytest.fixture(scope="module")
def slice_samples():
    prec = np.linalg.inv(TARGET_COV)

    def log_p(theta):
        d = np.asarray(theta) - TARGET_MEAN
        return -0.5 * d @ prec @ d

    init = MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    torch.manual_seed(5)
    return run_multivariate_slice_sampler_tuned(log_p, init, SLICE_SETTINGS, seed=5)


def test_slice_sampler_output_shape(slice_samples):
    s = SLICE_SETTINGS
    assert slice_samples.shape == (s.num_chains * s.num_samples, 2)
    assert np.isfinite(slice_samples).all()


def test_slice_sampler_recovers_mean(slice_samples):
    assert np.allclose(slice_samples.mean(0), TARGET_MEAN, atol=0.15)


def test_slice_sampler_recovers_covariance(slice_samples):
    assert np.allclose(np.cov(slice_samples.T), TARGET_COV, atol=0.25)


# --------------------------------------------------------------------------- #
# 11. Calibration moves beta the right way and respects beta_min               #
# --------------------------------------------------------------------------- #
BETA_START, BETA_MIN, N_ITER = 1.0, 0.5, 8


class _FlatGeneralPosterior:
    """Zero losses, so every importance weight is uniform and ESS stays at M."""

    def __init__(self, n=10, d=2):
        self.x_obs = torch.zeros(n, 1)
        self.d = d

    def per_x_losses(self, theta):
        return torch.zeros(theta.shape[0], self.x_obs.shape[0]), torch.zeros(theta.shape[0])

    def compute_theta_hat_true(self, **kwargs):
        return torch.zeros(self.d)

    def sample(self, beta):
        raise AssertionError("weights never degenerate here, so no resampling should happen")


class _FixedConjugatePosterior:
    """Always returns N(0, I); theta_hat decides whether it is covered."""

    def __init__(self, theta_hat, d=2):
        self.prior_mean = torch.zeros(d)
        self._theta_hat = theta_hat

    def compute_theta_hat_true(self, x_obs):
        return self._theta_hat

    def calibrated_posterior(self, x_boot, beta):
        return torch.zeros(len(self.prior_mean)), torch.eye(len(self.prior_mean))


def _run_general(covered: int):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(general_calibrate, "bootstrap_coverage_indicator", lambda *a, **k: covered)
        theta = torch.randn(50, 2)
        _, hist = general_calibrate.calibrate_beta(
            _FlatGeneralPosterior(), theta, beta_base=BETA_START, B=5, T=N_ITER, beta_min=BETA_MIN)
    return hist


def _run_gpc(covered: int):
    theta_hat = torch.zeros(2) if covered else torch.full((2,), 10.0)
    _, hist = calibrate_beta_gpc(
        torch.zeros(10, 1), _FixedConjugatePosterior(theta_hat), initial_beta=BETA_START,
        num_iterations=N_ITER, num_bootstraps=5, beta_min=BETA_MIN)
    return hist


@pytest.fixture(scope="module", params=[("general", 0), ("general", 1), ("gpc", 0), ("gpc", 1)],
                ids=lambda p: f"{p[0]}-coverage{p[1]}")
def calibration_run(request):
    method, covered = request.param
    hist = (_run_general if method == "general" else _run_gpc)(covered)
    return covered, hist


def test_calibration_sees_the_forced_coverage(calibration_run):
    covered, hist = calibration_run
    assert len(hist["coverages"]) == N_ITER
    assert all(c == covered for c in hist["coverages"])


def test_calibration_moves_beta_the_right_way(calibration_run):
    """Under-coverage widens the posterior (beta down); over-coverage narrows it (beta up)."""
    covered, hist = calibration_run
    betas = [BETA_START] + hist["betas"]
    steps = np.diff(betas)
    if covered:
        assert (steps > 0).all()
    else:
        assert (steps <= 0).all() and steps[0] < 0


def test_calibration_never_goes_below_beta_min(calibration_run):
    covered, hist = calibration_run
    assert min(hist["betas"]) >= BETA_MIN - 1e-12
    if not covered:
        assert abs(hist["betas"][-1] - BETA_MIN) < 1e-12   # 8 capped steps reach the floor


# --------------------------------------------------------------------------- #
# 12. Robust statistics resist gross outliers; plain moments do not            #
# --------------------------------------------------------------------------- #
TRUE_MEAN = torch.tensor([1.0, -1.0], dtype=torch.float64)
TRUE_COV = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)


@pytest.fixture(scope="module")
def contaminated_estimates():
    """10% of 400 points moved 8 units away; robust and plain estimates of the rest."""
    torch.manual_seed(6)
    np.random.seed(6)                      # MinCovDet draws from numpy's global RNG
    n = 400
    x = MultivariateNormal(TRUE_MEAN, TRUE_COV).sample((n,))
    x[: n // 10] += 8.0
    return {robust: estimate_stats(x, robust) for robust in (True, False)}


def _errors(est):
    mu, Sigma_inv = est
    true_inv = torch.linalg.inv(TRUE_COV)
    return ((mu - TRUE_MEAN).norm().item(),
            ((Sigma_inv - true_inv).norm() / true_inv.norm()).item())


def test_robust_estimates_are_close_to_truth(contaminated_estimates):
    mean_err, prec_err = _errors(contaminated_estimates[True])
    assert mean_err < 0.2 and prec_err < 0.3


def test_plain_estimates_are_pulled_by_outliers(contaminated_estimates):
    """Documents why the robust path exists: the plain mean moves ~0.8 per coordinate."""
    mean_err, _ = _errors(contaminated_estimates[False])
    assert mean_err > 0.8


def test_robust_beats_plain(contaminated_estimates):
    robust, plain = _errors(contaminated_estimates[True]), _errors(contaminated_estimates[False])
    assert robust[0] < plain[0] and robust[1] < plain[1]
