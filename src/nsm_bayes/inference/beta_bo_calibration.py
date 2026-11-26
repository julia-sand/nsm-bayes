"""Calibrate beta using BoTorch GP bisection or noisy target-seeking EI."""

import torch

from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

__all__ = [
    "fit_gp",
    "find_target_beta",
    "optimize_beta_bisection",
    "optimize_beta_bo",
]


def _training_tensors(betas, coverages):
    """Convert observations to BoTorch's required shape and precision."""
    train_x = torch.as_tensor(
        betas, dtype=torch.float64
    ).detach().reshape(-1, 1)

    train_y = torch.as_tensor(
        coverages,
        dtype=torch.float64,
        device=train_x.device,
    ).detach().reshape(-1, 1)

    if train_x.numel() == 0 or train_x.shape[0] != train_y.shape[0]:
        raise ValueError(
            "betas and coverages must be nonempty and have equal lengths."
        )

    if not (
        torch.isfinite(train_x).all()
        and torch.isfinite(train_y).all()
    ):
        raise ValueError("Observations must be finite.")

    return train_x, train_y


def fit_gp(betas, coverages):
    """Fit a coverage GP with learned observation noise."""
    train_x, train_y = _training_tensors(betas, coverages)

    model_gp = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=1),
        outcome_transform=Standardize(m=1),
    )

    mll = ExactMarginalLogLikelihood(
        model_gp.likelihood, model_gp
    )
    fit_gpytorch_mll(mll)

    model_gp.eval()
    return model_gp


# -------------------------------------------------------------------
# Bisection on GP-predicted coverage
# -------------------------------------------------------------------

def find_target_beta(
    model_gp,
    target=0.95,
    beta_min=None,
    beta_tol=1e-6,
    max_iter=100,
):
    """Find a beta whose GP-predicted coverage equals the target."""
    model_gp.eval()

    # Recover original beta units from normalized training inputs.
    train_x = model_gp.input_transform.untransform(
        model_gp.train_inputs[0]
    )

    lower = train_x.min().item() if beta_min is None else beta_min
    upper = train_x.max().item()

    if lower >= upper:
        raise ValueError("Lower bound must be smaller than upper bound.")

    with torch.no_grad():
        endpoints = train_x.new_tensor([[lower], [upper]])

        # posterior() returns predictions in original coverage units.
        means = model_gp.posterior(endpoints).mean[:, 0]
        f_lower = means[0].item() - target
        f_upper = means[1].item() - target

        if f_lower == 0:
            return lower, means[0].item()
        if f_upper == 0:
            return upper, means[1].item()

        if f_lower * f_upper > 0:
            raise ValueError(
                "Endpoint predictions do not bracket the target."
            )

        for _ in range(max_iter):
            middle = (lower + upper) / 2

            x = train_x.new_tensor([[middle]])
            coverage = model_gp.posterior(x).mean.item()
            f_middle = coverage - target

            if f_middle == 0 or (upper - lower) / 2 <= beta_tol:
                return middle, coverage

            if f_lower * f_middle < 0:
                upper = middle
            else:
                lower = middle
                f_lower = f_middle

    raise RuntimeError("Bisection did not converge.")


def optimize_beta_bisection(
    betas,
    coverages,
    target_coverage=0.95,
    beta_min=1e-4,
):
    """Bisection wrapper returning the calibrated ``beta``.

    Fits a GP to the supplied ``(log beta, coverage)`` pairs and finds the
    ``beta`` at which the GP mean crosses the target coverage; see
    :func:`find_target_beta` for the return contract.
    """
    model_gp = fit_gp(betas, coverages)

    return find_target_beta(
        model_gp,
        target=target_coverage,
        beta_min=beta_min,
    )


# -------------------------------------------------------------------
# Bayesian optimization using noisy target-seeking log-EI
# -------------------------------------------------------------------

def optimize_beta_bo(
    betas,
    coverages,
    target_coverage=0.95,
    beta_min=1e-4,
    posterior=None,
    n_iter=20,
):
    """
    Calibrate ``beta`` so the coverage matches ``target_coverage``.

    If `posterior` is provided, each selected ``beta`` is evaluated with
    ``posterior.estimate_coverage(beta=beta)`` and the best observed ``beta``
    is returned.

    If `posterior` is None, the GP is fit to the supplied observations and
    bisection is used, with no new evaluations. This branch delegates to
    :func:`optimize_beta_bisection` with its own defaults for
    ``target_coverage`` and ``beta_min`` (they are not forwarded).
    """

    if posterior is None:
        return optimize_beta_bisection(betas,coverages,
                        target_coverage=target_coverage,
                        beta_min=beta_min,)

    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative.")

    train_x, train_y = _training_tensors(betas, coverages)

    # Use the observed beta range as the finite acquisition domain.
    upper = train_x.max().item()

    if beta_min >= upper:
        raise ValueError(
            "beta_min must be smaller than the largest supplied beta."
        )

    bounds = train_x.new_tensor([[beta_min], [upper]])

    # Apply the target objective to posterior coverage samples.
    objective = GenericMCObjective(
        lambda samples, X=None: -(
            samples[..., 0] - target_coverage
        ).square()
    )

    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([128])
    )

    for _ in range(n_iter):
        model_gp = fit_gp(train_x, train_y)

        acquisition = qLogNoisyExpectedImprovement(
            model=model_gp,
            X_baseline=train_x,
            objective=objective,
            sampler=sampler,
            prune_baseline=True,
        )

        candidate, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        next_beta = candidate[0, 0].item()
        coverage = posterior.estimate_coverage(beta=next_beta)[0]

        new_y = torch.as_tensor(
            coverage,
            dtype=train_y.dtype,
            device=train_y.device,
        ).detach().reshape(1, 1)

        if not torch.isfinite(new_y).all().item():
            raise ValueError(
                f"Coverage evaluation at beta={next_beta} must be finite."
            )


        train_x = torch.cat(
            [train_x, candidate.detach()], dim=0
        )
        train_y = torch.cat([train_y, new_y], dim=0)

    best_index = (
        train_y[:, 0] - target_coverage
    ).abs().argmin().item()

    return (
        train_x[best_index, 0].item(),
        train_y[best_index, 0].item(),
    )
