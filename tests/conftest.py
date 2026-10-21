"""Shared fixtures for the ``scripts/run_*`` regression tests.

The experiment drivers in :mod:`nsm_bayes.experiments` are fully
deterministic under their ``set_seed`` calls, so the tests call
``run_experiment`` directly with a tiny config and a temporary
``save_dir``: no hydra and nothing is written into the repository's
``data/`` directory.

Two heavy components are shrunk so a "fast" full run stays in the
tens-of-seconds range:

* the normalising-flow likelihood trainer (``nsm_bayes.models.general.
  normflow.SNLE``) is replaced by a lightweight :class:`_StubNLE`, and
* the score-matching trainer's epoch budget is capped via
  ``monkeypatch.setattr`` on ``train_q_phi``'s defaults (the model passes
  only the required args, so the defaults are what get used).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

# Make ``import scripts.run_*`` resolve regardless of the pytest rootdir.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CONFIGS = _ROOT / "configs"


# --------------------------------------------------------------------------- #
# Tiny config helpers                                                          #
# --------------------------------------------------------------------------- #
def _compose(name: str):
    """Compose ``configs/<name>.yaml`` through Hydra so ``defaults`` lists
    (e.g. ``sir_undercounting`` inheriting from ``sir``) are resolved."""
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIGS), version_base=None):
        from hydra import compose
        return compose(config_name=name)


def _tiny_base(name: str) -> dict:
    """Compose ``configs/<name>.yaml`` and overwrite the expensive knobs."""
    cfg = OmegaConf.to_container(_compose(name), resolve=True)
    cfg.update(
        {
            "num_repeat": 1,
            "num_samples": 256,      # training-data size
            "n_obs": 40,             # observed-data size
            "num_posterior_samples": 30,
            "num_chains": 1,
            "warmup_steps": 15,
            "thin": 1,
            "T": 2,                  # beta-calibration iterations
            "B": 6,                  # bootstraps per iteration
        }
    )
    if name in ("sir", "sir_undercounting"):
        cfg.update({"N_sir": 300, "T_sir": 40})
    return cfg


@pytest.fixture
def gnk_cfg():
    return OmegaConf.create(_tiny_base("gnk"))


@pytest.fixture
def sir_cfg():
    return OmegaConf.create(_tiny_base("sir"))


@pytest.fixture
def turin_cfg():
    """Tiny Turin config: shrink the expensive simulator knobs too."""
    cfg = _tiny_base("turin")
    # Each training row is one receiver simulated under its own theta, so
    # 12 training receivers (plus one observed dataset of 40) keep a fast
    # end-to-end run on CPU.
    cfg.update({"num_samples": 12, "n_obs": 40})
    return OmegaConf.create(cfg)


@pytest.fixture
def sir_uc_cfg():
    return OmegaConf.create(_tiny_base("sir_undercounting"))


# --------------------------------------------------------------------------- #
# Heavy-component stand-ins (monkeypatched inside the autouse fixture)         #
# --------------------------------------------------------------------------- #
class _StubNLE:
    """A cheap Gaussian stand-in for an sbi ``SNLE`` likelihood estimator.

    It exposes exactly the surface the scripts use:
    ``log_prob(theta, condition)``, ``parameters()`` and
    ``state_dict()``/``load_state_dict``.
    The data dimension is inferred from the first ``log_prob`` call, and the
    "likelihood" is a standard normal centred on the first training row, so it
    is a valid (if crude) density the rest of the pipeline can consume.
    """

    def __init__(self, prior, density_estimator: str = "maf"):
        self._prior = prior
        self._density_estimator = density_estimator
        self._mu = None
        self._param = torch.nn.Parameter(torch.tensor(0.0))

    # -- sbi API surface --------------------------------------------------- #
    def append_simulations(self, theta_sim, x_sim):
        self._mu = x_sim.mean(dim=0).detach()
        return self

    def train(self, **kwargs):
        return self

    def parameters(self):
        return iter([self._param])

    def state_dict(self):
        return {"mu": self._mu, "param": self._param.data}

    def load_state_dict(self, state):
        self._mu = state["mu"]
        self._param.data = state["param"]

    def log_prob(self, theta, condition=None):
        """A smooth, twice-differentiable log-density in ``condition`` (the
        observation) that also depends on ``theta``.

        Matches the sbi ``SNLE`` API: ``log_prob(theta, condition=x_obs)``.
        ``theta`` may be a single vector of shape ``(d_theta,)`` or a batch
        of shape ``(*batch_shape, d_theta)``.  ``condition`` is the observed
        data of shape ``(n_obs, d_x)`` or ``(d_x,)``.

        ``calibrate_beta`` computes ``jacrev``/``hessian`` of this w.r.t.
        ``condition`` and calls ``.backward()`` w.r.t. ``theta``, so the stub
        must be a non-constant smooth function of both.  A single Gaussian
        with mean affine in ``theta`` satisfies that and is cheap to evaluate.
        """
        d_x = condition.shape[-1]
        # deterministic, theta-dependent mean; fixed scale so the stub
        # likelihood is not too concentrated (keeps posterior wide enough
        # for z-score checks in regression tests)
        if self._mu is None:
            self._mu = condition.reshape(-1, d_x).mean(dim=0).detach()
        # Use a large fixed scale so the stub likelihood is very flat;
        # the posterior then stays close to the prior, keeping z-scores
        # (posterior mean vs truth) small for the regression test.
        scale = torch.tensor(10.0, device=condition.device, dtype=condition.dtype)
        # theta: (*batch_shape, d_theta); take the first d_x components as a
        # deterministic shift on the data mean so the likelihood is a smooth,
        # non-constant function of theta.
        theta_lead = theta.shape[:-1]
        th_shift = theta[..., :d_x]
        # mean shape: (*batch_shape, d_x)
        mean = self._mu + th_shift
        var = scale ** 2
        # condition may be (d_x,) or (n_obs, d_x); expand to match theta batch
        cond = condition
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        # broadcast: cond (n_obs, d_x) or (1, d_x), mean (*batch, d_x)
        # log N(cond | mean, var * I), summed over the event dim (d_x)
        diff = cond.unsqueeze(0) - mean  # (*batch, n_obs, d_x) or (batch, d_x)
        logp = -0.5 * (
            d_x * torch.log(torch.as_tensor(2.0 * 3.141592653589793, device=condition.device))
            + d_x * torch.log(var)
            + (diff ** 2).sum(dim=-1) / var
        )
        # keep a (zero-valued) differentiable link to a parameter so the
        # autograd graph also touches a leaf, matching the real SNLE surface
        return logp + self._param * 0.0

    def build_posterior(self, sample_with: str = "mcmc", mcmc_method: str = "slice_np"):
        """Return a posterior stub whose ``.sample`` draws from the prior.

        The stub ignores all MCMC-specific kwargs (``num_chains``, ``thin``,
        ``warmup_steps``, ``init_strategy``, ...) and just returns i.i.d.
        prior draws of shape ``(num_chains * sample_shape, d_theta)`` so the
        downstream pipeline (which only consumes the samples) can proceed.
        """
        return _StubPosterior(self._prior)


class _StubPosterior:
    """A trivial posterior that samples i.i.d. from the prior."""

    def __init__(self, prior):
        self._prior = prior

    def sample(self, sample_shape=(), x=None, **kwargs):
        n = int(torch.tensor(sample_shape).prod().item()) if sample_shape else 1
        num_chains = int(kwargs.get("num_chains", 1))
        return self._prior.sample((num_chains * n,))


@pytest.fixture(autouse=True)
def _fast_runs(monkeypatch):
    """Swap in the cheap NLE and cap score-matching training for every test."""
    import nsm_bayes.models.conjugate.conj_train as conj_train
    import nsm_bayes.models.general.normflow as normflow

    # 1) Replace the normalising-flow trainer with the Gaussian stand-in.
    monkeypatch.setattr(normflow, "SNLE", _StubNLE)

    # 2) Cap the score-matching training budget. ConjugateModel.fit calls
    #    train_q_phi with only the required args, so patching it in its own
    #    module is enough (the fit call resolves the name at runtime).
    orig_train_q_phi = conj_train.train_q_phi
    def _fast_train_q_phi(*args, **kwargs):
        kwargs.setdefault("num_epochs", 2)
        kwargs.setdefault("early_stopping_patience", 1)
        return orig_train_q_phi(*args, **kwargs)
    monkeypatch.setattr(conj_train, "train_q_phi", _fast_train_q_phi)

    yield

# --------------------------------------------------------------------------- #
# Small assertion helpers                                                      #
# --------------------------------------------------------------------------- #
def load_tensor(save_dir: Path, name: str):
    import pickle
    with open(save_dir / name, "rb") as fh:
        return pickle.load(fh)


def assert_posterior_artifacts(save_dir: Path, ind: int = 0, d_theta: int = 4,
                               beta_min: float = 1e-6):
    """Assert the Case-1 posterior artifacts are present and structurally valid."""
    import torch
    mu = load_tensor(save_dir, f"posterior_mean_case1_{ind}.pkl")
    cov = load_tensor(save_dir, f"posterior_covariance_case1_{ind}.pkl")
    beta = load_tensor(save_dir, f"beta_case1_{ind}.pkl")
    hist = load_tensor(save_dir, f"gpc_history_case1_{ind}.pkl")

    assert mu.shape == (d_theta,), f"posterior mean shape {tuple(mu.shape)}"
    assert torch.isfinite(mu).all(), "posterior mean has non-finite entries"
    assert cov.shape == (d_theta, d_theta), f"posterior cov shape {tuple(cov.shape)}"
    assert torch.allclose(cov, cov.T, atol=1e-3), "posterior covariance not symmetric"
    eig = torch.linalg.eigvalsh(cov)
    assert eig.min().item() > 0, f"posterior covariance not positive definite (min eig {eig.min()})"
    assert float(beta) >= beta_min, f"beta_case1 {beta} < beta_min {beta_min}"
    for key in ("betas", "coverages"):
        assert key in hist, f"gpc history missing key {key!r}"
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in hist[key])
