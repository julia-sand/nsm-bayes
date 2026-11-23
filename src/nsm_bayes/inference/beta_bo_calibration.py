"""Bayesian optimisation to calibrate beta using a GP and pre-computed coverages."""

import numpy as np
import torch
import gpytorch

__all__ = ["optimize_beta_bo","fit_gp"]

class CoverageGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)


def fit_gp(
    betas,
    coverages,
):
    """
    Find the beta whose coverage is closest to `target_coverage`.

    betas, coverages : 1D arrays of pre-computed (beta, coverage) pairs.
        BO "evaluates" a beta by looking up its pre-computed coverage.
    Returns
    -------
    best_beta : float   beta where the GP's predicted coverage is closest to target
    info      : dict    observed betas/coverages, best observed beta, GP prediction
    """

    #turn the data into tensors
    train_x = torch.tensor(betas,dtype=torch.float32,requires_grad=False)
    train_y = torch.tensor(coverages,dtype=torch.float32,requires_grad=False)
    train_x = train_x[:,None]

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    ).to(dtype=torch.float32)

    #initialise model
    model_gp = CoverageGP(train_x, train_y, likelihood).to(dtype=torch.float32)

    # initial values for modeling raw coverage proportions.
    model_gp.mean_module.initialize(constant=train_y.mean().item())
    model_gp.covar_module.initialize(outputscale=0.1)
    model_gp.covar_module.base_kernel.initialize(lengthscale=0.2)
    likelihood.initialize(noise=1e-3)

    model_gp.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model_gp.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gp)

    for _ in range(300):
        optimizer.zero_grad()
        loss = -mll(model_gp(train_x), train_y)
        loss.backward()
        optimizer.step()

    # Predict the underlying coverage function, not a new noisy observation.
    model_gp.eval()
    likelihood.eval()
    return model_gp 


def find_target_beta(
    model_gp,
    target=0.95,
    beta_min=None,
    beta_tol=1e-6,
    max_iter=100,
):
    #find the beta where the coverage is as close as possible to the target
    
    #get input data 
    model_gp.eval()
    train_x = model_gp.train_inputs[0]

    if beta_min is None:
        lower = train_x.min().item()
    else:
        lower = beta_min

    upper = 1

    if lower >= upper:
        raise ValueError("Lower bound must be smaller than upper bound.")

    with torch.no_grad():
        endpoints = torch.tensor(
            [[lower], [upper]],
            dtype=train_x.dtype,
            device=train_x.device,
        )
        means = model_gp(endpoints).mean
        f_lower = means[0].item() - target
        f_upper = means[1].item() - target

        if f_lower == 0:
            return lower, target
        if f_upper == 0:
            return upper, target

        if f_lower * f_upper > 0:
            raise ValueError(
                "Endpoint predictions do not bracket the target."
            )

        for _ in range(max_iter):
            middle = (lower + upper) / 2

            x = torch.tensor(
                [[middle]],
                dtype=train_x.dtype,
                device=train_x.device,
            )
            coverage = model_gp(x).mean.item()
            f_middle = coverage - target

            if f_middle == 0 or (upper - lower) / 2 <= beta_tol:
                return middle, coverage

            if f_lower * f_middle < 0:
                upper = middle
            else:
                lower = middle
                f_lower = f_middle

    raise RuntimeError("Bisection did not converge.")

def optimize_beta_bo(betas, coverages, target_coverage=0.95, beta_min = 1e-4):
    
    model_gp = fit_gp(betas,coverages)
    best_beta, predicted_coverage = find_target_beta(model_gp,target_coverage)
    
    return best_beta, predicted_coverage 