import csv
import pickle
import time
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from nsm_bayes.gpc import calibrate_beta, calibrate_beta_gpc

from nsm_bayes.models.conjugate import ConjugateModel
from nsm_bayes.models.general import make_nle_logprob, train_normflow
from nsm_bayes.simulators.add_noise import simulate_contaminated_dataset
from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator
from nsm_bayes.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.utils import run_mcmc

def _run(cfg: DictConfig, save_dir: Path):
    #####------Load config values-----######
    num_repeat = cfg.num_repeat # Number of repetitions of the experiment
    num_sim = cfg.num_samples # Number of training data samples
    dtype = torch.float32
    # Initialize simulator
    sim = SIRSimulator(
        obs_model=cfg.obs_model if "obs_model" in cfg else "poisson",
        prior_mean=cfg.prior_mean if "prior_mean" in cfg else None,
        prior_cov=cfg.prior_cov if "prior_cov" in cfg else None,
    )
    prior = sim.prior()
    prior_mean = prior.loc
    prior_cov = prior.covariance_matrix

    # True parameters in unconstrained space
    theta_true_constrained = torch.tensor(cfg.theta_true, dtype=dtype).unsqueeze(0)
    theta_true = sim.to_unconstrained(theta_true_constrained).squeeze(0)
    n_obs = cfg.n_obs # Number of observed data samples
    epsilon = cfg.epsilon # Percentage of outliers in the observed data
    # Setting directory for saving data
    save_dir.mkdir(parents=True, exist_ok=True) # Create the directory
    
    all_thetas = []
    
    for ind in range(num_repeat):
        torch.manual_seed(ind+123)
        d_theta = prior_mean.shape[0]  # Number of parameters
        d_x = cfg.d_x # Data dimension
        #######-------Generate training data-----######
        start_time = time.perf_counter() # Record the start time
        #######-------Generate simulations----######
        N = cfg.N_sir
        T = cfg.T_sir
        # Use class methods for training data generation
        theta, y_sim, x_sim = sim.generate_training_data(prior, num_sim, T, N)
        #y_sim = prior.sample((num_sim,))
        #####-----Run NLE and MCMC using sbi library----#####
        likelihood_estimator = train_normflow(theta, x_sim, prior)
        end_time = time.perf_counter() # Record the end time
        cost_nle_training = end_time - start_time
        # Saving the likelihood estimator network
        # (Keeping this as it is not model.save_checkpoint)
        torch.save(likelihood_estimator, save_dir / f"likelihood_estimator_full_{ind}.pt")
        # Save the time taken to train NLE
        with open(save_dir/ f"cost_nle_training_{ind}.pkl", "wb") as f:
            pickle.dump(cost_nle_training, f)
        
        # Store theta for CSV
        all_thetas.append(theta.detach().cpu().numpy())
        
        with open(save_dir/ f"x_sim_{ind}.pkl", "wb") as f:
            pickle.dump(x_sim, f)
        #######-------Generate observed data-----######
        theta_batch = theta_true.unsqueeze(0).repeat(n_obs, 1)
        y_obs = sim.simulate(theta_batch, T=T, N=N)   # (n_obs, T)
        x_obs = sim.summary(y_obs, N) 
        y_obs_mis, is_contam, theta_used = simulate_contaminated_dataset(
            theta_true=theta_true,
            n_obs=n_obs,
            simulate_fn=sim.simulate,
            T=T,
            N=N,
            epsilon=epsilon,
            prior=prior,
        )
        x_obs_mis = sim.summary(y_obs_mis, N)
        # Save the observed data without outliers
        with open(save_dir/ f"x_obs_{ind}.pkl", "wb") as f:
            pickle.dump(x_obs, f)
        # Save the observed data with outliers
        with open(save_dir/ f"x_obs_mis_{ind}.pkl", "wb") as f:
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
        with open(save_dir/ f"cost_nle_mcmc_{ind}.pkl", "wb") as f:
            pickle.dump(cost_nle_mcmc, f)
        with open(save_dir/ f"samples_nle_mis_{ind}.pkl", "wb") as f:
            pickle.dump(samples_nle_mis, f)

        
        model = ConjugateModel(d_x=d_x, d_theta=d_theta, hidden_dim=cfg.hidden_dim)
        start_time = time.perf_counter() # Record the start time
        # Fit the standardizers and train the networks (score matching)
        training_history = model.fit(x_sim=x_sim, theta_sim=theta)
        end_time = time.perf_counter() # Record the end time
        cost_sm_case1_train = end_time - start_time
        # Save the time 
        with open(save_dir/ f"cost_sm_case1_train_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_case1_train, f)
        
        model.save_checkpoint(cfg,save_dir,ind)

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
        
        mu_n, Sigma_n = model.calibrated_posterior(x_obs_mis, 
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
    thetas_array = np.concatenate(all_thetas, axis=0)
    np.savetxt(save_dir / "theta_all.csv", thetas_array, delimiter=",")
if __name__ == "__main__":
    pass
@hydra.main(version_base=None, config_path="../configs", config_name="sir")
def run_sir(cfg: DictConfig):
    """Hydra entry point for the run_sir experiment."""
    save_dir = Path(get_original_cwd()) / "data" / cfg.experiment_name
    _run(cfg=cfg, save_dir=save_dir)
