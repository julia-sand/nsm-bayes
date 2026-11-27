"""Fast unit tests with known answers for the numerical building blocks.

Each expensive result (a fitted posterior, a large simulation, an MCMC run, a
calibration history, robust estimates) is computed once in a module-scoped
fixture and shared by every test that checks a property of it.
"""
import numpy as np
import pytest
import torch
from functools import partial
from omegaconf import OmegaConf
from torch.distributions import MultivariateNormal, Uniform

import nsm_bayes.models.general.posterior as general_posterior
from nsm_bayes.inference.bootstrap import (
    bootstrap_coverage_indicator,
    multinomial_bootstrap_counts,
    reweight_for_bootstrap_beta,
    weighted_mean_cov,
    weighted_quantile,
)
from nsm_bayes.inference.beta_calibration_loop import calibrate_beta
from nsm_bayes.models.conjugate.conj_model import EnergyBasedModel
from nsm_bayes.models.conjugate.conj_posterior import ConjugatePosterior
from nsm_bayes.models.general.posterior import GeneralPosterior
from nsm_bayes.simulators.benchmark_simulators import GandKSimulator,SIRSimulator
from nsm_bayes.utils.linalg import jacobian_and_laplacian
from nsm_bayes.utils.statistics import estimate_stats, chi2_ppf


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

    model = EnergyBasedModel(d_x=d_x, d_theta=d_theta, hidden_dim=16)
    model.standardizer_theta.fit(prior.sample((500,)))
    model.standardizer_x.fit(torch.randn(500, d_x) * 2.0 + 1.0)

    posterior = ConjugatePosterior(model, prior.loc, prior.covariance_matrix,
                                   robust=False, weight_type="imq")
    x_obs = torch.randn(30, d_x) * 2.0 + 1.0
    with torch.no_grad():
        results = {b: posterior.compute_posterior(x_obs, b) for b in BETAS}
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


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 0.9])
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
    assert torch.allclose(sim.summary(y, add_noise=False), expected)


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
# 10. GeneralPosterior.sample (sbi MCMC) recovers a known Gaussian            #
# --------------------------------------------------------------------------- #
SM_TARGET_MEAN = torch.tensor([1.0, -2.0], dtype=torch.float64)
SM_TARGET_COV = torch.tensor([[1.0, 0.6], [0.6, 2.0]], dtype=torch.float64)
SM_TARGET_PREC = torch.linalg.inv(SM_TARGET_COV)


def _gaussian_log_posterior(prior, theta, beta):
    """Known 2-D Gaussian log-density (mean SM_TARGET_MEAN, cov SM_TARGET_COV).

    Stand-in for ``GeneralPosterior.posterior_log_prob``: mirrors its
    contract, returning ``-inf`` outside the prior support and an
    unnormalised log-density as a float otherwise.

    Parameters
    ----------
    prior : torch.distributions.Distribution
        The posterior's prior, used only for its support.
    theta : array-like
        Parameter vector of shape ``(d_theta,)``.
    beta : float
        Unused (the stand-in ignores the learning rate).

    Returns
    -------
    float
        Unnormalised log-density, or ``-inf`` outside the prior support.
    """
    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    lp = prior.log_prob(theta_t).sum()
    if torch.isneginf(lp):
        return -np.inf
    d = torch.as_tensor(theta, dtype=torch.float64) - SM_TARGET_MEAN
    return float(lp.double() - 0.5 * d @ SM_TARGET_PREC @ d)


def _build_gaussian_posterior(num_samples: int, num_chains: int = 2,
                              warmup_steps: int = 300) -> GeneralPosterior:
    """GeneralPosterior on tiny data, with the conftest stub flow.

    The prior is bounded (a box far from the Gaussian mass, so the sampling
    target is the Gaussian itself) so the ``-inf`` branch of
    :meth:`GeneralPosterior._potential` is exercisable. The flow is the
    conftest ``_StubNLE`` stand-in and is never evaluated: the tests
    monkeypatch ``posterior_log_prob``, so only the flow's attributes matter
    (and ``sample`` does not touch it at all). ``_StubNLE`` is used directly
    rather than through :class:`GeneralModel` because the latter demands a
    ``prior.loc``, which the bounded test prior does not have.
    """
    from tests.conftest import _StubNLE

    prior = Uniform(torch.full((2,), -20.0), torch.full((2,), 20.0))
    mcmc = {"num_chains": num_chains, "warmup_steps": warmup_steps,
            "thin": 1, "init_strategy": "proposal"}
    flow = _StubNLE(prior)
    torch.manual_seed(0)
    x_obs = torch.randn(5, 2)
    return GeneralPosterior(x_obs=x_obs, flow_model=flow, prior=prior,
                            mcmc=mcmc, num_samples=num_samples, robust=False)


def _patch_log_prob(monkeypatch, post: GeneralPosterior) -> None:
    """Point ``post.posterior_log_prob`` at the known Gaussian stand-in."""
    monkeypatch.setattr(post, "posterior_log_prob",
                        partial(_gaussian_log_posterior, post.prior))


@pytest.mark.parametrize("num_chains", [1, 2])
def test_general_posterior_sample_shape_and_dtype(num_chains, monkeypatch):
    torch.manual_seed(7)
    post = _build_gaussian_posterior(num_samples=64, num_chains=num_chains,
                                     warmup_steps=32)
    _patch_log_prob(monkeypatch, post)
    s = post.sample(1.0)
    assert s.shape == (64, 2)
    assert s.dtype == post.x_obs.dtype and s.device == post.x_obs.device
    assert torch.isfinite(s).all()


def test_general_posterior_sample_recovers_gaussian(monkeypatch):
    torch.manual_seed(8)
    post = _build_gaussian_posterior(num_samples=3000)
    _patch_log_prob(monkeypatch, post)
    s = post.sample(1.0)
    assert s.shape == (3000, 2)
    assert torch.allclose(s.mean(0).double(), SM_TARGET_MEAN, atol=0.15)
    assert torch.allclose(torch.cov(s.T.double(), correction=0), SM_TARGET_COV, atol=0.25)


def test_general_posterior_potential_is_minus_inf_outside_prior_support(monkeypatch):
    """The test prior is bounded: _potential returns -inf (no error) outside."""
    post = _build_gaussian_posterior(num_samples=32, warmup_steps=8)
    _patch_log_prob(monkeypatch, post)
    theta = torch.tensor([[100.0, 0.0], [1.0, -2.0]], dtype=torch.float32)
    v = post._potential(theta, post.x_obs, beta=1.0)
    assert v.shape == (2,)
    assert torch.isneginf(v[0]) and torch.isfinite(v[1])


# --------------------------------------------------------------------------- #
# 11. Calibration control flow: coverage is recorded, the floor is respected   #
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
        # Called once at beta_init by prepare_calibration and, in the
        # refresh tests, again by refresh_pool at the updated beta; accept
        # either.
        return torch.randn(50, self.d)

    def prepare_calibration(self, beta, alpha, B, x_obs=None,
                            ess_threshold=0.3):
        self.alpha = alpha
        self.B = B
        self.theta_samples = self.sample(beta)
        self.beta_base = beta
        self.ell, self.L_orig = self.per_x_losses(self.theta_samples)
        self.theta_hat = self.compute_theta_hat_true()

    def estimate_coverage(self, beta, ess_threshold=0.3):
        # Same importance-reweighted coverage step as the real method; the
        # indicator is read from the module so tests can monkeypatch it.
        n = self.x_obs.shape[0]
        W = reweight_for_bootstrap_beta(
            self.ell, self.L_orig,
            multinomial_bootstrap_counts(n, self.B, device=self.x_obs.device,
                                         dtype=self.x_obs.dtype),
            beta_new=beta, beta_base=self.beta_base,
        )
        indicator = general_posterior.bootstrap_coverage_indicator
        cover = sum(
            indicator(self.theta_samples, W[b], self.theta_hat, self.alpha)
            for b in range(self.B)
        )
        ess_mean = (1.0 / (W ** 2).sum(dim=1)).mean().item()
        assert ess_mean >= ess_threshold * self.theta_samples.shape[0]
        return cover / self.B, ess_mean, self.theta_samples.shape[0]

    def refresh_pool(self, beta):
        # With flat losses ESS never drops, so this is never called in the
        # default tests; implement it for completeness.
        self.theta_samples = self.sample(beta)
        self.ell, self.L_orig = self.per_x_losses(self.theta_samples)
        self.beta_base = beta


class _FixedConjugatePosterior:
    """Always returns N(0, I); theta_hat decides whether it is covered."""

    def __init__(self, theta_hat, d=2):
        self.prior_mean = torch.zeros(d)
        self._theta_hat = theta_hat
        self.x_obs = torch.zeros(10, 1)

    def prepare_calibration(self, beta, alpha, B, x_obs=None,
                            ess_threshold=0.3):
        self.alpha = alpha
        self.B = B
        self.theta_hat = self._theta_hat
        if x_obs is not None:
            self.x_obs = x_obs

    def estimate_coverage(self, beta, ess_threshold=0.3):
        d = len(self.prior_mean)
        n_obs = self.x_obs.shape[0]
        crit = chi2_ppf(1.0 - self.alpha, d)
        covered = 0
        for _ in range(self.B):
            indices = torch.randint(0, n_obs, (n_obs,), device=self.x_obs.device)
            x_boot = self.x_obs[indices]
            mu_boot, Sigma_boot = self.compute_posterior(x_boot, beta)
            from nsm_bayes.utils.statistics import mahalanobis_sq
            if mahalanobis_sq(self.theta_hat, mu_boot, Sigma_boot).item() <= crit:
                covered += 1
        # Closed-form posterior: no MCMC pool, so ESS is infinite and the
        # pool size is zero (same convention as the real method).
        return covered / self.B, float("inf"), 0

    def compute_posterior(self, x_boot, beta):
        return torch.zeros(len(self.prior_mean)), torch.eye(len(self.prior_mean))


def _run_general(covered: int):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(general_posterior, "bootstrap_coverage_indicator",
                   lambda *a, **k: covered)
        _, hist = calibrate_beta(
            _FlatGeneralPosterior(), initial_beta=BETA_START, B=5, n_iterations=N_ITER,
            beta_min=BETA_MIN)
    return hist

def _run_gpc(covered: int):
    theta_hat = torch.zeros(2) if covered else torch.full((2,), 10.0)
    _, hist = calibrate_beta(
        _FixedConjugatePosterior(theta_hat), initial_beta=BETA_START, B=5, n_iterations=N_ITER,
        x_obs=torch.zeros(10, 1), beta_min=BETA_MIN)
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

def test_calibration_never_goes_below_beta_min(calibration_run):
    covered, hist = calibration_run
    assert min(hist["betas"]) >= BETA_MIN - 1e-12
    if not covered:
        assert abs(hist["betas"][-1] - BETA_MIN) < 1e-12   # 8 capped steps reach the floor
    assert len(hist["ess"]) == N_ITER
    assert all(e >= 0.0 for e in hist["ess"])


# --------------------------------------------------------------------------- #
# 11b. GeneralPosterior.refresh_pool                                          #
# --------------------------------------------------------------------------- #

class _RefreshTestPosterior:
    """Minimal stand-in exposing the same attributes and methods as
    GeneralPosterior for the refresh_pool code path, without a real flow."""

    def __init__(self, n=10, d=2):
        self.x_obs = torch.zeros(n, 1)
        self.d = d
        self.theta_samples = torch.randn(50, d)
        self.beta_base = 1.0
        self.ell = torch.zeros(50, n)
        self.L_orig = torch.zeros(50)
        self.theta_hat = torch.zeros(d)
        self.refresh_called = False
        self.refresh_beta = None

    def per_x_losses(self, theta):
        return (torch.zeros(theta.shape[0], self.x_obs.shape[0]),
                torch.zeros(theta.shape[0]))

    def sample(self, beta):
        return torch.randn(50, self.d)

    def refresh_pool(self, beta):
        # Replicates GeneralPosterior.refresh_pool using the same code path.
        # Import the real implementation via the module.
        from nsm_bayes.models.general.posterior import GeneralPosterior
        GeneralPosterior.refresh_pool(self, beta)


def test_refresh_pool_updates_state():
    """refresh_pool resamples and recomputes losses at the new beta."""
    p = _RefreshTestPosterior()
    old_samples = p.theta_samples.clone()
    old_beta_base = p.beta_base

    p.refresh_pool(2.0)

    assert p.beta_base == 2.0
    assert p.theta_samples.shape == (50, p.d)
    assert p.ell.shape == (50, p.x_obs.shape[0])
    assert p.L_orig.shape == (50,)
    # New samples should differ from old (random draw)
    assert not torch.allclose(old_samples, p.theta_samples)


def test_pool_not_refreshed_on_final_iteration():
    """The last iteration's refresh is a no-op: no downstream consumer uses
    the pool afterwards, so the loop skips it to save MCMC cost.

    Force ESS below the threshold on *every* call to ``estimate_coverage``.
    With ``n_iterations`` calls, only the first ``n_iterations - 1`` may
    trigger a refresh.
    """
    refresh_calls: list[float] = []

    class _AlwaysLowESSPosterior(_FlatGeneralPosterior):
        def __init__(self, n=10, d=2):
            super().__init__(n=n, d=d)
            self.refresh_calls = refresh_calls

        def estimate_coverage(self, beta, ess_threshold=0.3):
            # Always report ESS below threshold so the refresh branch is
            # taken on every iteration that gets past the guard.
            cov, _, pool = super().estimate_coverage(
                beta, ess_threshold=ess_threshold)
            return cov, 0.05 * pool, pool

        def refresh_pool(self, beta):
            self.refresh_calls.append(beta)
            self.theta_samples = self.sample(beta)
            self.ell, self.L_orig = self.per_x_losses(self.theta_samples)
            self.beta_base = beta

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(general_posterior, "bootstrap_coverage_indicator",
                   lambda *a, **k: 1)
        post = _AlwaysLowESSPosterior()
        calibrate_beta(post, initial_beta=BETA_START, B=5, n_iterations=N_ITER,
                       beta_min=BETA_MIN)

    # Exactly n_iterations - 1 refreshes: the final iteration must be skipped.
    assert len(post.refresh_calls) == N_ITER - 1
    assert all(math.isfinite(b) and b > 0 for b in post.refresh_calls)


def test_refresh_pool_requires_theta_hat():
    """refresh_pool raises RuntimeError when theta_hat has not been set."""
    p = _RefreshTestPosterior()
    p.theta_hat = None
    # The guard keys off theta_hat being None but the message refers to the
    # missing prepare_calibration call.
    with pytest.raises(RuntimeError,
                       match="prepare_calibration to be called first"):
        p.refresh_pool(1.0)


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


# --------------------------------------------------------------------------- #
# 13. EnergyBasedModel built from a checkpoint alone                           #
# --------------------------------------------------------------------------- #
def _fitted_model(d_x: int, d_theta: int) -> EnergyBasedModel:
    """Untrained model with both standardisers fitted on random data."""
    torch.manual_seed(0)
    model = EnergyBasedModel(d_x=d_x, d_theta=d_theta, hidden_dim=16)
    model.standardizer_x.fit(torch.randn(300, d_x) * 2.0 + 1.0)
    model.standardizer_theta.fit(torch.randn(300, d_theta) * 0.5 - 0.5)
    return model


@pytest.mark.parametrize("d_x,d_theta", [(1, 4), (3, 2)])
def test_checkpoint_reconstruction(d_x, d_theta, tmp_path):
    model = _fitted_model(d_x, d_theta)
    path = tmp_path / "nets.pt"
    model.save_checkpoint(path)

    rebuilt = EnergyBasedModel(checkpoint=path)
    assert (rebuilt.d_x, rebuilt.d_theta, rebuilt.hidden_dim) == (d_x, d_theta, 16)

    # State dicts are identical, key for key.
    rebuilt_sd = rebuilt.state_dict()
    assert set(rebuilt_sd) == set(model.state_dict())
    for key in model.state_dict():
        assert torch.equal(rebuilt_sd[key], model.state_dict()[key])

    # Identical network outputs on random input.
    x = torch.randn(7, d_x)
    with torch.no_grad():
        assert torch.equal(rebuilt.T_phi(x), model.T_phi(x))
        assert torch.equal(rebuilt.b_phi(x), model.b_phi(x))


@pytest.mark.parametrize("d_x,d_theta", [(1, 4), (3, 2)])
def test_checkpoint_construction_rejects_mismatched_d_x(d_x, d_theta, tmp_path):
    path = tmp_path / "nets.pt"
    _fitted_model(d_x, d_theta).save_checkpoint(path)
    with pytest.raises(ValueError, match="d_x"):
        EnergyBasedModel(checkpoint=path, d_x=d_x + 1)


def test_construction_without_checkpoint_or_dimensions_raises():
    with pytest.raises(ValueError):
        EnergyBasedModel()


@pytest.mark.parametrize("d_x,d_theta", [(1, 4), (3, 2)])
def test_load_checkpoint_on_dims_built_model(d_x, d_theta, tmp_path):
    model = _fitted_model(d_x, d_theta)
    path = tmp_path / "nets.pt"
    model.save_checkpoint(path)

    loaded = EnergyBasedModel(d_x=d_x, d_theta=d_theta, hidden_dim=16)
    loaded.load_checkpoint(path)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key in model.state_dict():
        assert torch.equal(loaded.state_dict()[key], model.state_dict()[key])
# --------------------------------------------------------------------------- #
# 14. Beta calibration: functional tests for gpc / bisect / grid               #
#     (structure, step magnitude, and interface contracts -- no convergence to #
#      a specific beta value, no assertion on the update direction)            #
# --------------------------------------------------------------------------- #
import math

from nsm_bayes.inference.beta_update_rules import (
    best_beta_from_grid,
    grid_start,
    step_bisect,
    step_gpc,
)

TARGET = 0.95


def _grid_search(grid, coverage_fn, target_coverage=TARGET, beta_min=0.01):
    """Fixed-grid selection: ``grid_start`` evaluates the grid, then
    :func:`best_beta_from_grid` picks the closest point to the target."""
    _, _, _, hist = grid_start(grid, coverage_fn,
                               target_coverage=target_coverage, beta_min=beta_min)
    return best_beta_from_grid(hist, target_coverage, beta_min), hist


def test_step_gpc_move_magnitude_and_floor():
    """gpc step: ``log(beta) += clip(k_t * (coverage - target), ±max_log_step)``,
    floored at ``log(beta_min)``.  Asserts the magnitude of the move (and that
    a zero gap gives a zero move), not which way beta moves for a given
    coverage."""
    # zero coverage gap -> zero move
    new0 = step_gpc(0.0, TARGET, t=0, max_log_step=0.25, beta_min=1e-4,
                    target_coverage=TARGET)
    assert new0 == 0.0
    # a coverage gap of |cov - target| = 0.2 with k_0 = 1 gives a step of 0.2
    # (below the clip cap, above the floor)
    for cov in (TARGET - 0.2, TARGET + 0.2):
        new = step_gpc(0.0, cov, t=0, max_log_step=0.25, beta_min=1e-4,
                       target_coverage=TARGET)
        assert abs(new) == pytest.approx(0.2)
    # a large gap is clipped to max_log_step
    for cov in (TARGET - 5.0, TARGET + 5.0):
        new = step_gpc(0.0, cov, t=0, max_log_step=0.25, beta_min=1e-4,
                       target_coverage=TARGET)
        assert abs(new) <= 0.25 + 1e-12
        assert abs(abs(new) - 0.25) <= 1e-12
    # step magnitude shrinks with t (k_t = 10/(t+10) schedule)
    s0 = abs(step_gpc(0.0, TARGET - 5.0, t=0, max_log_step=10.0, beta_min=1e-4,
                      target_coverage=TARGET))
    s100 = abs(step_gpc(0.0, TARGET - 5.0, t=100, max_log_step=10.0, beta_min=1e-4,
                        target_coverage=TARGET))
    assert 0.0 < s100 < s0
    # floor at log(beta_min)
    assert step_gpc(math.log(0.011), TARGET - 5.0, t=0, max_log_step=0.25,
                    beta_min=0.01, target_coverage=TARGET) == math.log(0.01)


def test_gpc_loop_run_structure():
    """GPC loop: runs n_iterations steps, history is complete, every step
    respects the log-step cap and the beta_min floor, and the coverage
    recorded at each step matches the coverage function -- no assertion on
    which way beta moves or its final value."""
    def cov(b):
        return max(0.0, min(1.0, 1.05 - math.log10(b)))

    class _GPCPosterior(_NoGridPosterior):
        def estimate_coverage(self, beta, ess_threshold=0.3):
            return cov(beta), 10.0, 10

    n_iterations = 20
    start = 0.3
    beta, hist = calibrate_beta(
        _GPCPosterior(), initial_beta=start, method="gpc",
        alpha=0.05, B=1, n_iterations=n_iterations, max_log_step=0.25, beta_min=0.01,
    )
    # one beta, one coverage and one ESS recorded per iteration
    assert len(hist["betas"]) == n_iterations
    assert len(hist["coverages"]) == n_iterations
    assert len(hist["ess"]) == n_iterations
    assert all(e == 10.0 for e in hist["ess"])
    # returned beta is the last one in the history and is positive
    assert beta == hist["betas"][-1]
    assert beta > 0.0
    # the recorded coverage at each step matches the coverage function
    assert hist["coverages"][0] == cov(start)
    # every step is clipped at max_log_step in log space
    for lo, hi in zip(hist["betas"][:-1], hist["betas"][1:]):
        assert abs(math.log(hi) - math.log(lo)) <= 0.25 + 1e-12
    # beta never goes below beta_min
    assert all(b >= 0.01 for b in hist["betas"])


def test_step_bisect_bracketing_structure():
    """bisect step: returns a bracket that shrinks towards the crossing and
    moves the candidate into it -- structural checks only."""
    lo, hi = math.log(0.1), math.log(10.0)
    # coverage above target: the current point becomes the new lo
    log_beta = 0.5 * (lo + hi)
    new_log_beta, new_bracket = step_bisect(
        log_beta, 0.99, t=0, bracket=(lo, hi),
        max_log_step=0.25, beta_min=0.01, target_coverage=TARGET)
    nlo, nhi = new_bracket
    assert nlo == log_beta and nhi == hi
    # the next candidate is the midpoint of the new bracket
    assert abs(new_log_beta - 0.5 * (nlo + nhi)) < 1e-12
    # coverage below target: the current point becomes the new hi
    new_log_beta, new_bracket = step_bisect(
        new_log_beta, 0.1, t=1, bracket=new_bracket,
        max_log_step=0.25, beta_min=0.01, target_coverage=TARGET)
    nlo, nhi = new_bracket
    assert nlo == log_beta and nhi == 0.5 * (log_beta + hi)
    assert abs(new_log_beta - 0.5 * (nlo + nhi)) < 1e-12


def test_grid_search_selection_is_closest_to_target():
    """Grid search returns the grid point whose coverage is closest to the
    target (selection rule + tie-break + beta_min floor, no fixed beta value)."""
    grid = [0.1, 1.0, 10.0]

    # a point exactly on the target must beat points off target
    beta, hist = _grid_search(grid, lambda b: (TARGET - 0.05 * abs(math.log10(b)), 1.0, 10))
    assert beta == 1.0
    # history carries the whole grid in order with one coverage and ESS per point
    assert hist["betas"] == grid
    assert len(hist["coverages"]) == len(grid)
    assert len(hist["ess"]) == len(grid)

    # off-target: the closest point wins (0.1 is at gap 0.05, 10.0 at gap 0.9)
    beta, _ = _grid_search(grid, lambda b: ({0.1: 0.9, 1.0: 0.1, 10.0: 0.05}[b], 2.0, 10))
    assert beta == 0.1

    # tie in |gap| (exact, via a dyadic offset): the higher coverage wins
    d = 2.0 ** -14
    beta, _ = _grid_search(grid, lambda b: ({0.1: TARGET - d, 1.0: TARGET + d, 10.0: TARGET - d}[b], 2.0, 10))
    assert beta == 1.0
    # tie in |gap| and coverage: the larger beta wins
    beta, _ = _grid_search(grid, lambda b: ({0.1: 0.9, 1.0: 0.9, 10.0: 0.9}[b], 2.0, 10))
    assert beta == 10.0

    # constant coverage: history still carries the grid in order
    _, hist = _grid_search(grid, lambda b: (0.5, 3.0, 10))
    assert hist["betas"] == grid and len(hist["coverages"]) == len(grid)
    assert hist["ess"] == [3.0] * len(grid)

    # floor at beta_min
    beta, _ = _grid_search([0.001], lambda b: (TARGET, 1.0, 10))
    assert beta == 0.01
    # non-positive grid values are rejected
    with pytest.raises(ValueError, match="positive"):
        _grid_search([-1.0, 1.0], lambda b: (0.5, 1.0, 10))


def test_grid_start_refines_crossing_and_extracts_slope_and_bracket():
    # coverage monotonically decreasing in beta, crossing the target at b=10^0.8
    # (1.35 - 0.5*log10(b) = 0.95 -> log10(b) = 0.8 -> b ~= 6.31, inside [1, 10])
    cov = lambda b: (1.35 - 0.5 * math.log10(b), 4.0, 10)
    start, slope, bracket, hist = grid_start(
        [0.1, 1.0, 10.0], cov, target_coverage=TARGET, beta_min=0.01)
    # crossing is bracketed between 1.0 and 10.0; refinement stays inside
    assert math.log(1.0) < math.log(start) < math.log(10.0)
    assert slope < 0  # coverage falls with beta
    assert bracket is not None and bracket[0] < bracket[1]
    assert math.log(1.0) <= bracket[0] < bracket[1] <= math.log(10.0)
    # interpolated crossing lies between the bracket endpoints
    assert bracket[0] <= math.log(start) <= bracket[1]
    # history carries the grid points, one coverage and ESS per point
    assert len(hist["betas"]) == 3
    assert len(hist["coverages"]) == len(hist["betas"])
    assert hist["ess"] == [4.0] * len(hist["betas"])
    # no crossing: all above target -> largest grid value
    start, slope, bracket, _ = grid_start([0.1, 1.0], lambda b: (0.99, 4.0, 10),
                                          target_coverage=TARGET, beta_min=0.01)
    assert start == 1.0 and slope is None and bracket is None
    # no crossing: all below target -> smallest grid value
    start, slope, bracket, _ = grid_start([0.1, 1.0], lambda b: (0.01, 4.0, 10),
                                          target_coverage=TARGET, beta_min=0.01)
    assert start == 0.1 and slope is None and bracket is None


class _NoGridPosterior:
    """Minimal general-method stand-in: no grid search, gpc only."""

    supports_grid_search = False

    def prepare_calibration(self, beta, alpha, B, x_obs=None, ess_threshold=0.3):
        pass

    def estimate_coverage(self, beta, ess_threshold=0.3):
        return 0.5, 10.0, 10


class _GridPosterior(_NoGridPosterior):
    """Same stand-in with the grid-search capability on."""

    supports_grid_search = True


@pytest.mark.parametrize("method", ["bisect", "grid"])
def test_general_posterior_rejects_non_gpc_methods(method):
    with pytest.raises(ValueError, match="only 'gpc' is"):
        calibrate_beta(_NoGridPosterior(), initial_beta=1.0, method=method, n_iterations=1)


def test_general_posterior_rejects_array_initial_beta():
    with pytest.raises(ValueError, match="supports_grid_search"):
        calibrate_beta(_NoGridPosterior(), initial_beta=np.array([0.1, 1.0]),
                       method="gpc", n_iterations=1)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate_beta(_NoGridPosterior(), initial_beta=1.0, method="gd", n_iterations=1)


def test_grid_and_bisect_require_array_initial_beta():
    for method in ("grid", "bisect"):
        with pytest.raises(ValueError, match="requires an array initial_beta"):
            calibrate_beta(_GridPosterior(), initial_beta=1.0, method=method, n_iterations=1)


def test_list_and_tuple_initial_beta_rejected():
    for bad in ([0.1, 1.0], (0.1, 1.0)):
        with pytest.raises(TypeError, match="lists and tuples"):
            calibrate_beta(_GridPosterior(), initial_beta=bad, method="grid", n_iterations=1)


@pytest.mark.parametrize("bad", [np.array([]), np.array([0.1]).reshape(1, 1), 0.0, -1.0,
                                 np.array([0.0, 1.0])])
def test_invalid_initial_beta_rejected(bad):
    method = "grid" if (isinstance(bad, np.ndarray) and bad.ndim == 1 and bad.size) else "gpc"
    with pytest.raises((ValueError,)):
        calibrate_beta(_GridPosterior(), initial_beta=bad, method=method, n_iterations=1)


def test_unknown_kwargs_raise_type_error():
    # strict keyword validation: anything not in the signature is rejected
    for bad in ("init_beta", "beta_minimum", "num_bootstrap", "n_boot", "ess_thresh"):
        with pytest.raises(TypeError, match=bad):
            calibrate_beta(_NoGridPosterior(), n_iterations=1, **{bad: 1.0})


def test_array_initial_beta_adds_history_extras():
    # coverage crossing the target inside [1, 10]: grid search + gpc loop
    cov = lambda b: 1.35 - 0.5 * math.log10(b)

    class _CrossingPosterior(_GridPosterior):
        def estimate_coverage(self, beta, ess_threshold=0.3):
            return cov(beta), 5.0, 10

    beta, hist = calibrate_beta(_CrossingPosterior(), initial_beta=np.array([0.1, 1.0, 10.0]),
                                method="gpc", alpha=0.05, B=1, n_iterations=2, beta_min=0.01)
    assert set(hist) >= {"betas", "coverages", "ess", "grid", "initial_beta", "slope"}
    assert len(hist["betas"]) == 2 and len(hist["coverages"]) == 2
    assert len(hist["ess"]) == 2
    assert 1.0 < hist["initial_beta"] < 10.0
    assert hist["slope"] < 0
    # the grid history (grid plus refinement mids) carries one ESS per point
    assert len(hist["grid"]["ess"]) == len(hist["grid"]["betas"])


def test_grid_method_returns_grid_point():
    # the grid search needs a crossing, so use a coverage curve through the
    # target (crossing at 10.0): 10.0 is closest to 0.95 on the grid
    cov = lambda b: max(0.0, min(1.0, 1.95 - math.log10(b)))

    class _GridOnlyPosterior(_GridPosterior):
        def estimate_coverage(self, beta, ess_threshold=0.3):
            return cov(beta), 7.0, 10

    beta, hist = calibrate_beta(_GridOnlyPosterior(),
                                initial_beta=np.array([0.1, 1.0, 10.0]),
                                method="grid", alpha=0.05, B=1, n_iterations=5, beta_min=0.01)
    # gaps 0.05 (0.1) vs 0.05 (1.0) vs 0.0 (10.0) -> exact hit wins
    assert beta == 10.0
    assert hist["betas"] == [0.1, 1.0, 10.0]  # grid history, not loop history
    assert len(hist["coverages"]) == 3
    assert hist["ess"] == [7.0] * 3


# ---------------------------------------------------------------------------
# optimize_beta_bo: GP-based Bayesian-optimisation beta selection
# ---------------------------------------------------------------------------


def test_optimize_beta_bo_rejects_bad_inputs():
    from nsm_bayes.inference.beta_bo_calibration import optimize_beta_bo

    # Mismatched lengths
    with pytest.raises(ValueError, match="equal lengths"):
        optimize_beta_bo([0.1, 0.2], [0.9])

    # beta_min must be smaller than the largest supplied beta (posterior path)
    with pytest.raises(ValueError, match="beta_min"):
        optimize_beta_bo([0.1, 0.2], [0.9, 0.8], beta_min=0.5,
                         posterior=object(), n_iter=0)

    # Betas at or below beta_min are rejected too
    with pytest.raises(ValueError, match="beta_min"):
        optimize_beta_bo([0.001, 0.002, 0.005, 0.01], [0.9, 0.93, 0.97, 0.99],
                         target_coverage=0.95, beta_min=0.01,
                         posterior=object(), n_iter=0)


def test_optimize_beta_bo_recovers_crossing_beta():
    """With no extra evaluations (``n_iter=0``) the observed beta closest to
    the target wins."""
    from nsm_bayes.inference.beta_bo_calibration import optimize_beta_bo

    # coverage(b) = 0.8 + 0.5 * b  ->  crossing at b = 0.3
    betas = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0], dtype=float)
    coverages = np.array([0.8 + 0.5 * b for b in betas])

    beta, cov = optimize_beta_bo(betas, coverages, target_coverage=0.95,
                                 posterior=object(), n_iter=0)
    assert np.isfinite(beta) and beta > 0
    assert beta == pytest.approx(0.3)
    assert cov == pytest.approx(0.95)


def test_optimize_beta_bo_fallback_when_no_crossing():
    """Without a posterior, GP bisection raises if the predicted coverage
    never reaches the target, instead of guessing a point."""
    from nsm_bayes.inference.beta_bo_calibration import optimize_beta_bo

    # coverage is always well below target 0.95
    betas = np.array([0.01, 0.02, 0.05, 0.1], dtype=float)
    coverages = np.array([0.50, 0.55, 0.70, 0.80])

    with pytest.raises(ValueError, match="bracket"):
        optimize_beta_bo(betas, coverages, target_coverage=0.95)


def test_optimize_beta_bo_result_deterministic():
    """Repeated calls with the same inputs give the same beta."""
    from nsm_bayes.inference.beta_bo_calibration import optimize_beta_bo

    betas = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0], dtype=float)
    coverages = np.array([0.60, 0.78, 0.90, 0.96, 0.99, 1.0])

    results = set()
    for _ in range(3):
        beta, _ = optimize_beta_bo(betas, coverages, target_coverage=0.95)
        results.add(float(beta))
    assert len(results) == 1  # fully deterministic



@pytest.mark.parametrize("df, crit", [
    (1, 3.841459),
    (2, 5.991465),
    (3, 7.814728),
    (10, 18.307038),
])
def test_chi2_ppf_matches_tabulated_quantiles(df, crit):
    """The torch bisection inverse CDF recovers the tabulated 95%
    chi-square critical values."""
    assert chi2_ppf(0.95, df) == pytest.approx(crit, abs=1e-5)
