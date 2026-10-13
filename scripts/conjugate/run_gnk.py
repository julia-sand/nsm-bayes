import pickle
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from nsm_bayes.gpc import calibrate_beta_gpc

from nsm_bayes.models.conjugate import ConjugateModel
from nsm_bayes.simulators.add_noise import add_outliers_by_proportion
from nsm_bayes.simulators.benchmark_simulators.simulators import (
    GandKSimulator,
)

def _run(cfg: DictConfig, save_dir: Path):
    #####------Load config values-----######
    num_repeat = cfg.num_repeat # Number of repetitions of the experiment
    num_samples = cfg.num_samples # Number of training data samples
    dtype = torch.float32
    prior_mean = torch.tensor(cfg.prior_mean, dtype=dtype) if "prior_mean" in cfg else None # Prior mean vector
    prior_cov  = torch.tensor(cfg.prior_cov,  dtype=dtype) if "prior_cov" in cfg else None # Prior covariance matrix
    theta_true = torch.tensor(cfg.theta_true, dtype=dtype) # True parameter value
    n_obs = cfg.n_obs # Number of observed data samples
    epsilon = cfg.epsilon # Percentage of outliers in the observed data
    outlier_values = cfg.outlier_values # Location of outliers
    # Setting directory for saving data
    save_dir.mkdir(parents=True, exist_ok=True) # Create the directory
    # Initialize the g-and-k simulator
    sim = GandKSimulator(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
    
    # To store theta values for all repetitions
    all_thetas = []
    
    for ind in range(num_repeat):
        random.seed(ind+1)
        np.random.seed(ind+1)
        torch.manual_seed(ind+1)
        d_theta = prior_mean.shape[0]  # Number of parameters
        d_x = 1 # Data dimension
        #######-------Generate training data-----######
        start_time = time.perf_counter() # Record the start time
        prior = sim.prior() # Define the Gaussian prior
        # Generate simulations
        theta = prior.sample((num_samples,))
        x_sim = sim.simulate(theta).unsqueeze(-1)
        
        with open(save_dir/ f"x_sim_{ind}.pkl", "wb") as f:
            pickle.dump(x_sim, f)
        
        #Generate observations from the simulator
        x_obs = sim.simulate(theta_true, n=n_obs).unsqueeze(-1)

        #Generate 'real data', by adding outliers
        x_obs_mis, outlier_indices = add_outliers_by_proportion(x_obs, epsilon, outlier_values) # Adding outliers
        
        # Save the observed data without outliers
        with open(save_dir/ f"x_obs_{ind}.pkl", "wb") as f:
            pickle.dump(x_obs, f)
    
        # Save the observed data with outliers
        with open(save_dir/ f"x_obs_mis_{ind}.pkl", "wb") as f:
            pickle.dump(x_obs_mis, f)
        
        # Save the outlier indices
        with open(save_dir/ f"outlier_indices_{ind}.pkl", "wb") as f:
            pickle.dump(outlier_indices, f)

        start_time = time.perf_counter() # Record the start time
        
        model = ConjugateModel(d_x=d_x, d_theta=d_theta, hidden_dim=cfg.hidden_dim)
        start_time = time.perf_counter() # Record the start time
        
        # Train the model on the true data 
        model.fit(x_sim=x_sim, theta_sim=theta)
        
        end_time = time.perf_counter() # Record the end time of the training
        
        # Save the time taken for training
        cost_sm_case1_train = end_time - start_time
        with open(save_dir/ f"cost_sm_case1_train_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_case1_train, f)
        
        # Save the trained neural net
        model.save_checkpoint(cfg, save_dir, ind)
        
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
            learning_rate_fn = lambda t: 10.0 / (t + 10.)
        )

        mu_n, Sigma_n = model.calibrated_posterior(
            x_obs_mis, 
            calibrated_beta, 
            prior_mean, 
            prior_cov, 
            cfg.robust_flag
        )
        
        end_time = time.perf_counter() # Record the end time
        cost_sm_case1_posterior = (end_time - start_time) 
        # Save the posterior mean 
        with open(save_dir/ f"posterior_mean_case1_{ind}.pkl", "wb") as f:
            pickle.dump(mu_n, f)
        # Save the posterior covariance
        with open(save_dir/ f"posterior_covariance_case1_{ind}.pkl", "wb") as f:
            pickle.dump(Sigma_n, f)
        with open(save_dir/ f"gpc_history_case1_{ind}.pkl", "wb") as f:
            pickle.dump(gpc_history, f)
        with open(save_dir / f"beta_case1_{ind}.pkl", "wb") as f:
            pickle.dump(calibrated_beta, f)
        # Save the time 
        with open(save_dir/ f"cost_sm_case1_posterior_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_case1_posterior, f)
        print("Iteration number: ", ind)
    # Save all thetas into one CSV file
    # all_thetas is a list of arrays, each of shape (num_samples, d_theta)
    # we concatenate them along axis 0 to get (num_repeat * num_samples, d_theta)
    thetas_array = np.concatenate(all_thetas, axis=0)
    np.savetxt(save_dir / "theta_all.csv", thetas_array, delimiter=",")
    

@hydra.main(version_base=None, config_path="../configs", config_name="gnk")
def run_gnk(cfg: DictConfig):
    """Hydra entry point for the run_gnk experiment."""
    save_dir = Path(get_original_cwd()) / "data" / cfg.experiment_name
    _run(cfg=cfg, save_dir=save_dir)

if __name__ == "__main__":
    run_gnk()