import torch

####----Function to get simulations from a prior and simulator----####
def get_simulations(prior, simulator, num_samples, data_dim):
    # Sample from prior
    theta = prior.sample([num_samples])
    x_sim = torch.zeros(num_samples, data_dim) 
    # Sample from the simulator
    for i in range(num_samples):
        x_sim[i,:] = simulator(theta[i, :])
    return theta, x_sim