import pickle
import time
from pathlib import Path
import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.distributions import MultivariateNormal
from nsm_bayes.gpc import calibrate_beta, calibrate_beta_gpc
from nsm_bayes.method import (
    ScoreMatchingLogPosterior,
    compute_posterior_case1,
    robust_mean_cov,
    w_imq_squared,
)
from nsm_bayes.models.conjugate import ConjugateModel
from nsm_bayes.models.general import make_nle_logprob, train_normflow
from nsm_bayes.simulators.add_noise import apply_undercounting_trajectory
from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator
from nsm_bayes.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.utils import run_mcmc, sample_mean_and_covariance


def _run(cfg: DictConfig, save_dir: Path):
    
    #####------Load config values-----######
    num_repeat = cfg.num_repeat # Number of repetitions of the experiment
    num_sim = cfg.num_samples # Number of training data samples
    dtype = torch.float32
    prior_mean = torch.tensor([
        torch.log(torch.tensor(cfg.prior_mean[0])),         # log beta
        torch.log(torch.tensor(cfg.prior_mean[1])),        # log gamma
        torch.logit(torch.tensor(cfg.prior_mean[2])),        # logit rho
        torch.log(torch.tensor(cfg.prior_mean[3])),           # log I0
        ],  dtype=dtype)
    prior_std  = torch.tensor(cfg.prior_std,  dtype=dtype) 
    prior_cov = torch.diag(prior_std ** 2) # Prior covariance matrix
    theta_true = torch.tensor([
        torch.log(torch.tensor(cfg.theta_true[0])),
        torch.log(torch.tensor(cfg.theta_true[1])),
        torch.logit(torch.tensor(cfg.theta_true[2])),
        torch.log(torch.tensor(cfg.theta_true[3]))], dtype=dtype) # True parameter value 
    n_obs = cfg.n_obs # Number of observed data samples
    epsilon = cfg.epsilon # Percentage of outliers in the observed data
    sim = SIRSimulator(
        prior_mean=prior_mean if "prior_mean" in cfg else None,
        prior_cov=prior_cov if "prior_cov" in cfg else None,
    )
    # Setting directory for saving data
    save_dir.mkdir(parents=True, exist_ok=True) # Create the directory
    
    for ind in range(num_repeat):
        torch.manual_seed(ind+123)
        d_theta = prior_mean.shape[0]  # Number of parameters
        d_x = cfg.d_x # Data dimension
        #######-------Generate training data-----######
        start_time = time.perf_counter() # Record the start time
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov) # Define the Gaussian prior
        #######-------Generate simulations----######
        N = cfg.N_sir
        T = cfg.T_sir
        theta = prior.sample((num_sim,))
        y_sim = sim.simulate(theta, T=T, N=N)
        x_sim = sim.summary(y_sim, N)
        
        #######-------Generate observed data-----######
        theta_batch = theta_true.unsqueeze(0).repeat(n_obs, 1)
        y_obs = sim.simulate(theta_batch, T=T, N=N)   # (n_obs, T)
        y_cor, is_contam = apply_undercounting_trajectory(y_obs, epsilon=epsilon, q=cfg.q, per_time=False)
        x_obs_mis = sim.summary(y_cor, N)
        
        # Save the observed data without outliers
        with open(save_dir/ f"y_obs_{ind}.pkl", "wb") as f:
            pickle.dump(y_obs, f)
        
        # Save the observed data with outliers
        with open(save_dir/ f"x_obs_mis_{ind}.pkl", "wb") as f:
            pickle.dump(x_obs_mis, f)
        start_time = time.perf_counter() # Record the start time
        
        model = ConjugateModel(d_x=d_x, d_theta=d_theta, hidden_dim=cfg.hidden_dim)
        start_time = time.perf_counter() # Record the start time
        # Fit the standardizers and train the networks (score matching)
        training_history = model.fit(x_sim=x_sim, theta_sim=theta)
        end_time = time.perf_counter() # Record the end time
        cost_sm_case1_train = end_time - start_time
        # Save the time 
        with open(save_dir/ f"cost_sm_case1_train_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_case1_train, f)
        
        # Save the trained neural nets
        model.save_checkpoint(cfg, save_dir, ind)
        #########-----Compute conjugate posterior (Case 1)-------#######
        start_time = time.perf_counter() # Record the start time
        #x_obs_normalized = standardizer_x(x_obs_mis) # Normalize the observed data
        
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

        mu_n, Sigma_n = model.calibrated_posterior(x_obs_mis,
                                               model, 
                                               prior_mean, 
                                               prior_cov,
                                               cfg.robust_flag
                                               )
        
        ##need: posterior mean and variance 
        
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

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../configs", config_name="sir_undercounting")
    def run_sir_undercounting(cfg: DictConfig):
        """Hydra entry point for the run_sir_undercounting experiment."""
        save_dir = Path(get_original_cwd()) / "data" / cfg.experiment_name
        _run(cfg=cfg, save_dir=save_dir)

    run_sir_undercounting() 
