import pickle
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sbi.utils.sbiutils import standardizing_net
from torch.distributions import MultivariateNormal

from nsm_bayes.gpc import calibrate_beta, calibrate_beta_gpc
from nsm_bayes.method import (
    ScoreMatchingLogPosterior,
    compute_posterior_case1,
    robust_mean_cov,
    w_imq_squared,
)
from nsm_bayes.models.conjugate.bphi_net import BphiNet
from nsm_bayes.models.conjugate.conj_train import train_q_phi
from nsm_bayes.models.conjugate.tphi_net import TphiNet
from nsm_bayes.models.general import make_nle_logprob, train_normflow
from nsm_bayes.simulators.add_noise import add_outliers_by_proportion
from nsm_bayes.simulators.benchmark_simulators.simulators import (
    sample_gandk_fully_reparameterized,
)
from nsm_bayes.simulators.simulators import get_simulations
from nsm_bayes.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.utils import run_mcmc, sample_mean_and_covariance


def _run(cfg: DictConfig, save_dir: Path):

    #####------Load config values-----######
    num_repeat = cfg.num_repeat # Number of repetitions of the experiment

    num_samples = cfg.num_samples # Number of training data samples

    dtype = torch.float32
    prior_mean = torch.tensor(cfg.prior_mean, dtype=dtype) # Prior mean vector
    prior_cov  = torch.tensor(cfg.prior_cov,  dtype=dtype) # Prior covariance matrix
    theta_true = torch.tensor(cfg.theta_true, dtype=dtype) # True parameter value

    n_obs = cfg.n_obs # Number of observed data samples
    n_obs_ref = cfg.n_obs_ref # Number of observed data samples

    epsilon = cfg.epsilon # Percentage of outliers in the observed data
    outlier_values = cfg.outlier_values # Location of outliers

    # Setting directory for saving data
    save_dir.mkdir(parents=True, exist_ok=True) # Create the directory

    for ind in range(num_repeat):
        random.seed(ind+1)
        np.random.seed(ind+1)
        torch.manual_seed(ind+1)


        d_theta = prior_mean.shape[0]  # Number of parameters
        d_x = 1 # Data dimension

        #######-------Generate training data-----######
        start_time = time.perf_counter() # Record the start time

        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov) # Define the Gaussian prior

        # Generate simulations
        theta, x_sim = get_simulations(prior, sample_gandk_fully_reparameterized, num_samples, d_x)

        #####-----Run NLE and MCMC using sbi library----#####

        likelihood_estimator = train_normflow(theta, x_sim, prior)

        end_time = time.perf_counter() # Record the end time
        cost_nle_training = end_time - start_time

        # Saving the likelihood estimator network
        torch.save(likelihood_estimator, save_dir / f"likelihood_estimator_full_{ind}.pt")

        # Save the time taken to train NLE
        with open(save_dir/ f"cost_nle_training_{ind}.pkl", "wb") as f:
            pickle.dump(cost_nle_training, f)

        # Save the training data
        with open(save_dir/ f"theta_{ind}.pkl", "wb") as f:
            pickle.dump(theta, f)

        with open(save_dir/ f"x_sim_{ind}.pkl", "wb") as f:
            pickle.dump(x_sim, f)

        #######-------Generate observed data-----######
        x_obs = sample_gandk_fully_reparameterized(theta_true, n_obs).unsqueeze(-1)

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

        #######-------Run MCMC using weighted score-matching loss (General)------#######

        # c = median_heuristic(x_obs_mis)
        c = 1.
        mu_hat, Sigma_hat = robust_mean_cov(x_obs_mis)

        if cfg.robust_flag== False:
            mu_hat, Sigma_hat = sample_mean_and_covariance(x_obs_mis)

        Sigma_inv = torch.linalg.inv(Sigma_hat + 1e-6 * torch.eye(d_x, device=x_obs_mis.device, dtype=x_obs_mis.dtype))

        start_time = time.perf_counter() 
        # Initial run of MCMC
        sm_lp_base = ScoreMatchingLogPosterior(
            x_obs=x_obs_mis,
            prior=prior,
            beta=cfg.beta_base_general,
            q_phi_log_prob=likelihood_estimator,
            mu_hat=mu_hat, Sigma_inv=Sigma_inv, c=c, weight_type="imq"
        )

        theta_samples_base = run_multivariate_slice_sampler_tuned(
            log_posterior_fn=sm_lp_base,
            prior=prior,
            num_samples=cfg.num_posterior_samples, num_chains=cfg.num_chains, warmup_steps=cfg.warmup_steps, thin=cfg.thin
        )
        theta_samples_base = torch.from_numpy(theta_samples_base).to(x_obs.dtype).to(x_obs.device)

        def refresh_sampler_at(beta_new: float) -> torch.Tensor:
            sm_lp = ScoreMatchingLogPosterior(
                x_obs=x_obs_mis, prior=prior, beta=beta_new,
                q_phi_log_prob=likelihood_estimator,
                mu_hat=mu_hat, Sigma_inv=Sigma_inv, c=c, weight_type="imq"
            )
            arr = run_multivariate_slice_sampler_tuned(
                log_posterior_fn=sm_lp, prior=prior,
                num_samples=cfg.num_posterior_samples, num_chains=cfg.num_chains, warmup_steps=cfg.warmup_steps, thin=cfg.thin
            )
            return torch.from_numpy(arr).to(x_obs_mis.dtype).to(x_obs_mis.device)

        beta, history_general = calibrate_beta(
            theta_samples_base=theta_samples_base,
            beta_base=cfg.beta_base_general,
            x_obs=x_obs_mis,
            q_phi_log_prob=make_nle_logprob(likelihood_estimator),
            mu_hat=mu_hat,
            Sigma_inv=Sigma_inv,
            weight_type="imq",
            c=c,
            alpha=cfg.alpha,  
            B=cfg.B,
            T=cfg.T,
            step_schedule=lambda t: 10.0/(t+10.),
            refresh_sampler=refresh_sampler_at
        )

        log_posterior_calculator = ScoreMatchingLogPosterior(
            x_obs=x_obs_mis,
            prior=prior,
            beta=beta,
            q_phi_log_prob=likelihood_estimator,
            mu_hat=mu_hat,
            Sigma_inv=Sigma_inv,
            c=c,
            weight_type="imq"
        )

        samples_sm_general = run_multivariate_slice_sampler_tuned(
            log_posterior_fn=log_posterior_calculator,
            prior=prior,
            num_samples=cfg.num_posterior_samples,
            num_chains=cfg.num_chains,
            warmup_steps=cfg.warmup_steps,
            thin=cfg.thin
        )

        end_time = time.perf_counter() # Record the end time
        cost_sm_general_mcmc = end_time - start_time
        # Save the time 
        with open(save_dir/ f"cost_sm_general_mcmc_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_general_mcmc, f)

        with open(save_dir/ f"samples_sm_general_{ind}.pkl", "wb") as f:
            pickle.dump(samples_sm_general, f)

        with open(save_dir / f"gpc_history_general_{ind}.pkl", "wb") as f:
            pickle.dump(history_general, f)
        with open(save_dir / f"beta_general_{ind}.pkl", "wb") as f:
            pickle.dump(beta, f)

        #######-------Train q_phi of the proposed method (Case 1)------#######

        T_phi_net = TphiNet(d_x, cfg.hidden_dim, d_theta)
        b_phi_net = BphiNet(d_x, cfg.hidden_dim)

        # Standardize the data
        standardizer_x = standardizing_net(x_sim)
        standardizer_theta = standardizing_net(theta)

        # Apply the standardizers to get the normalized data for training
        x_sim_normalized = standardizer_x(x_sim)
        theta_sim_normalized = standardizer_theta(theta)

        start_time = time.perf_counter() # Record the start time
        # Training on the normalized data
        training_history = train_q_phi(
            x_sim=x_sim_normalized,
            theta=theta_sim_normalized,
            T_phi_net=T_phi_net,
            b_phi_net=b_phi_net
        )
        end_time = time.perf_counter() # Record the end time
        cost_sm_case1_train = end_time - start_time
        # Save the time 
        with open(save_dir/ f"cost_sm_case1_train_{ind}.pkl", "wb") as f:
            pickle.dump(cost_sm_case1_train, f)

        # Save the trained neural nets
        torch.save(
            {
                "T_phi_state_dict": T_phi_net.state_dict(),
                "B_phi_state_dict": b_phi_net.state_dict(),
                "standardizer_x_state_dict": standardizer_x.state_dict(),
                "standardizer_theta_state_dict": standardizer_theta.state_dict(),
                "config": dict(cfg),
            },
            save_dir / f"case1_nets_{ind}.pt"
        )

        #########-----Compute conjugate posterior (Case 1)-------#######
        start_time = time.perf_counter() # Record the start time

        x_obs_normalized = standardizer_x(x_obs_mis) # Normalize the observed data
        prior_mean_normalized = standardizer_theta(prior_mean) # Normalize the prior mean
        scales = standardizer_theta.std
        prior_cov_normalized = prior_cov / torch.outer(scales, scales) # Normalize the prior covariance

        mu_hat_obs, Sigma_hat_obs = robust_mean_cov(x_obs_normalized)
                
        if cfg.robust_flag == False:
            mu_hat_obs, Sigma_hat_obs = sample_mean_and_covariance(x_obs_mis)
            
        Sigma_inv_obs = torch.linalg.inv(Sigma_hat_obs + 1e-6 * torch.eye(d_x, device=x_obs_mis.device, dtype=x_obs_mis.dtype))

        c_case1 = 1.
        
        # Calibrating beta
        calibrated_beta, gpc_history = calibrate_beta_gpc(
            x_obs=x_obs_mis,
            T_phi_net=T_phi_net,
            b_phi_net=b_phi_net,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            standardizer_x=standardizer_x,
            standardizer_theta=standardizer_theta,
            initial_beta=cfg.beta_base_case1,        
            target_coverage=1.0 - cfg.alpha,
            num_iterations=cfg.T,        # Number of updates to beta
            num_bootstraps=cfg.B,        # B, number of samples to estimate coverage
            learning_rate_fn = lambda t: 10.0 / (t + 10.)
        )

        mu_n_normalized, Sigma_n_normalized = compute_posterior_case1(
            x_obs_normalized, T_phi_net, b_phi_net, calibrated_beta, prior_mean_normalized, prior_cov_normalized, w_imq_squared, mu_hat_obs, Sigma_inv_obs, c_case1)

        # Inverse-transform the posterior mean and covariance to the original scale
        mu_n = mu_n_normalized * standardizer_theta.std + standardizer_theta.mean

        scales = standardizer_theta.std
        Sigma_n = Sigma_n_normalized * torch.outer(scales, scales)

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
    run_gnk() 
        




@hydra.main(version_base=None, config_path="../configs", config_name="gnk")
def run_gnk(cfg: DictConfig):
    """Hydra entry point for the run_gnk experiment."""
    save_dir = Path(get_original_cwd()) / "data" / cfg.experiment_name
    _run(cfg=cfg, save_dir=save_dir)
