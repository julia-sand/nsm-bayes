import pickle
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.distributions import MultivariateNormal

from nsm_bayes.gpc import calibrate_beta
from nsm_bayes.method import (
    ScoreMatchingLogPosterior,
    robust_mean_cov,
    sigma_inverse,
)
from nsm_bayes.models.general import make_nle_logprob, train_normflow
from nsm_bayes.models.general.normflow import run_mcmc
from nsm_bayes.simulators.add_noise import apply_undercounting_trajectory
from nsm_bayes.simulators.benchmark_simulators.sir import SIRSimulator
from nsm_bayes.slice_sampler import run_multivariate_slice_sampler_tuned
from nsm_bayes.utils import sample_mean_and_covariance, set_seed


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
    n_obs_ref = cfg.n_obs_ref # Number of observed data samples
    epsilon = cfg.epsilon # Percentage of outliers in the observed data
    sim = SIRSimulator(
        prior_mean=prior_mean if "prior_mean" in cfg else None,
        prior_cov=prior_cov if "prior_cov" in cfg else None,
    )
    # Setting directory for saving data
    save_dir.mkdir(parents=True, exist_ok=True) # Create the directory
    for ind in range(num_repeat):
        set_seed(cfg.seed_offset + ind)
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
        #######-------Run neural score-matching Bayes------#######
        if not cfg.robust_flag:
            mu_hat, Sigma_hat = sample_mean_and_covariance(x_obs_mis)
        else:
            mu_hat, Sigma_hat = robust_mean_cov(x_obs_mis)
        Sigma_inv = sigma_inverse(Sigma_hat)
        start_time = time.perf_counter() 
        # Initial run of MCMC
        sm_lp_base = ScoreMatchingLogPosterior(
            x_obs=x_obs_mis,
            prior=prior,
            beta=cfg.beta_base_general,
            q_phi_log_prob=likelihood_estimator,
            mu_hat=mu_hat, Sigma_inv=Sigma_inv, weight_type="imq"
        )
        theta_samples_base = run_multivariate_slice_sampler_tuned(
            log_posterior_fn=sm_lp_base,
            prior=prior,
            num_samples=cfg.num_posterior_samples, num_chains=cfg.num_chains, warmup_steps=cfg.warmup_steps, thin=cfg.thin
        )
        theta_samples_base = torch.from_numpy(theta_samples_base).to(x_obs_mis.dtype).to(x_obs_mis.device)
        def refresh_sampler_at(beta_new: float) -> torch.Tensor:
            sm_lp = ScoreMatchingLogPosterior(
                x_obs=x_obs_mis, prior=prior, beta=beta_new,
                q_phi_log_prob=likelihood_estimator,
                mu_hat=mu_hat, Sigma_inv=Sigma_inv, weight_type="imq"
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
            alpha=cfg.alpha,  
            B=cfg.B,
            T=cfg.T,
            beta_min=cfg.beta_min_general,
            refresh_sampler=refresh_sampler_at
        )
        log_posterior_calculator = ScoreMatchingLogPosterior(
            x_obs=x_obs_mis,
            prior=prior,
            beta=beta,
            q_phi_log_prob=likelihood_estimator,
            mu_hat=mu_hat,
            Sigma_inv=Sigma_inv,
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
        
        print("Iteration number: ", ind)

@hydra.main(version_base=None, config_path="../../configs", config_name="sir_undercounting")
def run_sir_undercounting(cfg: DictConfig):
    """Hydra entry point for the run_sir_undercounting experiment."""
    save_dir = Path(get_original_cwd()) / "data" / cfg.experiment_name
    _run(cfg=cfg, save_dir=save_dir)

if __name__ == "__main__":
    
    run_sir_undercounting() 
