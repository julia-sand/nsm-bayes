import pickle
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from nsm_bayes.gpc import calibrate_beta, calibrate_beta_gpc
from nsm_bayes.method import (
    ScoreMatchingLogPosterior,
    robust_mean_cov,
    w_imq_squared,
)
from nsm_bayes.models.conjugate import ConjugateModel
from nsm_bayes.models.general import make_nle_logprob, train_normflow
from nsm_bayes.simulators.benchmark_simulators.simulators import TurinSimulator
from nsm_bayes.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.utils import run_mcmc


def _run(cfg: DictConfig, save_dir: Path):

    #####------Load config values-----######
    num_repeat = cfg.num_repeat # Number of repetitions of the experiment

    dtype = torch.float32
    theta_true = torch.log(torch.tensor(cfg.theta_true, dtype=dtype)) # True parameter value

    n_obs = cfg.n_obs # Number of observed data samples
    n_sim = cfg.num_samples # Number of training data samples

    epsilon = cfg.epsilon # Percentage of outliers in the observed data

    d_x = cfg.d_x # Data dimension

    # Setting directory for saving data
    save_dir.mkdir(parents=True, exist_ok=True) # Create the directory

    # Initialize the Turin channel model simulator
    sim = TurinSimulator(
        B=cfg.B_hz,
        Ns=cfg.Ns,
        device=cfg.device,
        prior_mean=cfg.prior_mean if "prior_mean" in cfg else None,
        prior_cov=cfg.prior_cov if "prior_cov" in cfg else None,
    )

    # Gaussian prior (configs/turin.yaml)
    prior = sim.prior()

    prior_mean = prior.loc
    d_theta = prior_mean.shape[0]  # Number of parameters
    prior_cov = prior.covariance_matrix

    for ind in range(num_repeat):

        torch.manual_seed(ind + 123)

        #######-------Generate training data-----######
        start_time = time.perf_counter() # Record the start time


        theta, x_sim = sim.generate_training_data(prior, n_sim)

        #####-----Train NLE using sbi library----#####
        dev = x_sim.device
        x_sim = x_sim.to(device=dev, dtype=dtype)
        theta = theta.to(device=dev, dtype=dtype)

        likelihood_estimator = train_normflow(
            theta, x_sim, prior, density_estimator=cfg.density_estimator
        )

        end_time = time.perf_counter() # Record the end time
        cost_nle_training = end_time - start_time

        # Save the time taken to train the NLE
        with open(save_dir / f"cost_nle_training_{ind}.pkl", "wb") as f:
            pickle.dump(cost_nle_training, f)

        # Save the training data
        with open(save_dir / f"theta_{ind}.pkl", "wb") as f:
            pickle.dump(theta, f)
        with open(save_dir / f"x_sim_{ind}.pkl", "wb") as f:
            pickle.dump(x_sim, f)

        #####-----Train q_phi using score-matching for NSM-conjugate------#####
        # Training on the simulated data
        model = ConjugateModel(
            d_x=d_x,
            hidden_dim=cfg.hidden_dim,
            d_theta=d_theta
        ).to(device=dev, dtype=dtype)

        start_time = time.perf_counter()
        model.fit(x_sim, theta)

        cost_sm_case1_train = time.perf_counter() - start_time
        # Save the training time
        with open(save_dir / f"cost_sm_case1_train_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_case1_train, f)
        model.save_checkpoint(cfg,save_dir,ind)

        #######-------Generate observed data-----######
        x_obs_mis = sim.generate_observed_data(
            theta_true,
            n_obs=n_obs,
            epsilon=epsilon,   # % outliers
        )
        x_obs_mis = x_obs_mis.to(device=dev, dtype=dtype)

        # Save the observed data with outliers
        with open(save_dir / f"x_obs_mis_{ind}.pkl", "wb") as f:
            pickle.dump(x_obs_mis, f)

        start_time = time.perf_counter() # Record the start time
        # NLE posterior samples under misspecification
        samples_nle_mis = run_mcmc(x_obs_mis, likelihood_estimator,
                num_pos_samples = cfg.num_posterior_samples,
                num_chains = cfg.num_chains,
                num_workers=cfg.num_chains,
                thin = cfg.thin,
                warmup_steps = cfg.warmup_steps)
        end_time = time.perf_counter() # Record the end time
        cost_nle_mcmc = end_time - start_time

        # Save the time taken
        with open(save_dir / f"cost_nle_mcmc_{ind}.pkl", "wb") as f:
            pickle.dump(cost_nle_mcmc, f)

        with open(save_dir / f"samples_nle_mis_{ind}.pkl", "wb") as f:
            pickle.dump(samples_nle_mis, f)


        #########-----Compute conjugate posterior (Case 1)-------#######
        start_time = time.perf_counter() # Record the start time
        
        # Calibrating beta
        calibrated_beta, gpc_history = calibrate_beta_gpc(
            x_obs_mis,
            model,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            initial_beta=cfg.beta_base_case1,
            target_coverage=1.0 - cfg.alpha,
            num_iterations=cfg.T,        # Number of updates to beta
            num_bootstraps=cfg.B,        # B, number of samples to estimate coverage
            learning_rate_fn = lambda t: 10.0 / (t + 10.),
            beta_min = cfg.beta_min_case1
        )
        
        mu_n, Sigma_n = model.calibrated_posterior(
            x_obs_mis, 
            model, 
            prior_mean, 
            prior_cov, 
            cfg.robust_flag
        )
        
        end_time = time.perf_counter() # Record the end time
        cost_sm_case1_posterior = (end_time - start_time)

        # Save the posterior mean
        with open(save_dir / f"posterior_mean_case1_{ind}.pkl", "wb") as f:
            pickle.dump(mu_n, f)

        # Save the posterior covariance
        with open(save_dir / f"posterior_covariance_case1_{ind}.pkl", "wb") as f:
            pickle.dump(Sigma_n, f)

        with open(save_dir / f"gpc_history_case1_{ind}.pkl", "wb") as f:
            pickle.dump(gpc_history, f)
        with open(save_dir / f"beta_case1_{ind}.pkl", "wb") as f:
            pickle.dump(calibrated_beta, f)

        # Save the time
        with open(save_dir / f"cost_sm_case1_posterior_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_case1_posterior, f)

        print("Iteration number: ", ind)

if __name__ == "__main__":
    pass

@hydra.main(version_base=None, config_path="../configs", config_name="turin")
def run_turin(cfg: DictConfig):
    """Hydra entry point for the run_turin experiment."""
    save_dir = (Path(get_original_cwd()) / "data" / cfg.experiment_name / f"eps={cfg.epsilon}" / f"de={cfg.density_estimator}")
    _run(cfg=cfg, save_dir=save_dir)
