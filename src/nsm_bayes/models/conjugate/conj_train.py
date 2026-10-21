import copy
from typing import Dict

import torch
import torch.optim as optim
from torch import nn
from torch.func import vmap
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.func import grad, jacfwd, vmap

from nsm_bayes.utils.linalg import jacobian_and_hessian


# Function to train the surrogate model for the conjugate case
def train_q_phi(  #add training params to be passed explicitly?
    x_sim: torch.Tensor,
    theta: torch.Tensor,
    T_phi_net: nn.Module,
    b_phi_net: nn.Module,
    learning_rate: float = 5e-4,
    num_epochs: int = 1000,
    batch_size: int = 128,
    weight_decay: float = 1e-5,
    validation_split: float = 0.2,
    early_stopping_patience: int = 20,
) -> Dict:
    """
    Trains the q_phi density estimator using Adam.

    Args:
        x_sim (torch.Tensor): Simulation data.
        theta (torch.Tensor): Corresponding simulation parameters.
        T_phi_net (nn.Module): The T_phi network model.
        b_phi_net (nn.Module): The b_phi network model.

    Returns:
        Dict: A dictionary containing the training history:
            {
                'train_losses': List[float],
                'val_losses': List[float]
            }
    """
    # --- 1. Training hyperparameters (defaults match the original fixed values) ---
    # scheduler_patience = 15
    
    # --- 2. Define Fixed `Sigma_inv` Internally ---
    # Assumes unweighted score matching loss
    Sigma_inv = torch.eye(x_sim.shape[1], device=x_sim.device)

    # --- 3. Setup Optimizer and DataLoaders ---
    params_to_train = list(T_phi_net.parameters()) + list(b_phi_net.parameters())
    optimizer = optim.Adam(params_to_train, lr=learning_rate, weight_decay=weight_decay)
    
    full_dataset = TensorDataset(x_sim, theta)
    n_train = int((1.0 - validation_split) * len(full_dataset))
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- 4. Initialize Training State ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    history = {'train_losses': [], 'val_losses': []}

    print("Starting training for q_phi...")
    for epoch in range(num_epochs):
        # Training Phase
        T_phi_net.train()
        b_phi_net.train()
        epoch_train_loss = 0.0
        for x_batch, theta_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_training_loss(x_batch, theta_batch, T_phi_net, b_phi_net, Sigma_inv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_losses'].append(avg_train_loss)

        # Validation Phase
        T_phi_net.eval()
        b_phi_net.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, theta_batch in val_loader:
                loss = calculate_training_loss(x_batch, theta_batch, T_phi_net, b_phi_net, Sigma_inv)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        history['val_losses'].append(avg_val_loss)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {
                'T_phi': copy.deepcopy(T_phi_net.state_dict()),
                'b_phi': copy.deepcopy(b_phi_net.state_dict())
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+1}.")
            break
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # --- 5. Load Best Model and Finalize ---
    if best_model_state is not None:
        final_epoch = epoch + 1 - epochs_no_improve
        print(f"Training finished. Loading best model from epoch {final_epoch} with validation loss {best_val_loss:.6f}.")
        T_phi_net.load_state_dict(best_model_state['T_phi'])
        b_phi_net.load_state_dict(best_model_state['b_phi'])
    else:
        print("Warning: Training finished but no best model was saved. The final model is the last one trained.")

    return history

def calculate_training_loss(x_batch, theta_batch, T_phi, b_phi, Sigma_inv):
    """
    Weighted score-matching loss (per-batch mean) for the exponential-family case:
       log q(x;θ) = θᵀT(x) + b(x)
       L(x,θ)     = ‖∇ₓ log q‖²_{Σ⁻¹} + 2 tr(Σ⁻¹ ∇²ₓ log q)
    """
    B, d_x = x_batch.shape
    assert theta_batch.shape[0] == B, "Batch sizes of x and theta must match."

    Sigma_inv = Sigma_inv.to(device=x_batch.device, dtype=x_batch.dtype)
    assert Sigma_inv.shape == (d_x, d_x), "Sigma_inv must be (d_x, d_x)."

    # Freeze dropout/batchnorm while keeping gradients
    was_training_T, was_training_b = T_phi.training, b_phi.training
    T_phi.eval()
    b_phi.eval()

    losses = vmap(single_pair_loss, in_dims=(0, 0, None, None, None))(
        x_batch, theta_batch, T_phi, b_phi, Sigma_inv
    )

    if was_training_T:
        T_phi.train()
    if was_training_b:
        b_phi.train()

    return losses.mean()

def calculate_training_loss1(x_batch, theta_batch, T_phi, b_phi, Sigma_inv):
    """
    Weighted score-matching loss (per-batch mean) for the exponential-family case:
       score(x;θ) = J_T(x)^T θ + ∇ b(x)
       L(x,θ)     = (score^T Σ^{-1} score) + 2 tr( Σ^{-1} ( Σ_i θ_i ∇^2 T_i(x) + ∇^2 b(x) ) )

    Shapes:
      x_batch:      (B, d_x)
      theta_batch:  (B, d_theta)
      T_phi(x):     (d_theta,)         -> J_T(x): (d_theta, d_x)
      b_phi(x):     scalar             -> ∇ b(x): (d_x,),  ∇^2 b(x): (d_x, d_x)
      Σ^{-1}:       (d_x, d_x)
    """
    B, d_x = x_batch.shape
    assert theta_batch.shape[0] == B, "Batch sizes of x and theta must match."

    # Ensure Σ^{-1} matches dtype/device
    Sigma_inv = Sigma_inv.to(device=x_batch.device, dtype=x_batch.dtype)
    assert Sigma_inv.shape == (d_x, d_x), "Sigma_inv must be (d_x, d_x)."

    # Temporarily set eval() to freeze any dropout/bn, but keep grads
    was_training_T, was_training_b = T_phi.training, b_phi.training
    T_phi.eval(); b_phi.eval()

    def loss_for_single_pair(xi, thetai):

        J_T,H_T = jacobian_and_hessian(xi,T_phi)
        grad_b, H_b = jacobian_and_hessian(xi,b_phi)

        # score(x;θ): (d_x,)
        score = J_T.T @ thetai + grad_b

        # ∇^2 log q(x|θ) = Σ_i θ_i ∇^2 T_i(x) + ∇^2 b(x)  -> (d_x, d_x)
        H_logq = torch.einsum('i,ijk->jk', thetai, H_T) + H_b

        # score^T Σ^{-1} score
        score_term = torch.dot(score, Sigma_inv @ score)

        # 2 * tr( Σ^{-1} H_logq ) = 2 * sum(Σ^{-1} * H_logq)
        div_term = 2.0 * torch.sum(Sigma_inv * H_logq)

        return (score_term + div_term) #/ d_x

    # Vectorize across batch
    losses = vmap(loss_for_single_pair, in_dims=(0, 0))(x_batch, theta_batch)

    # Restore training flags
    if was_training_T: T_phi.train()
    if was_training_b: b_phi.train()

    return losses.mean()




def score_with_aux(x, theta, T_phi, b_phi):
    """∇ₓ log q, returned twice so jacfwd can give the Hessian and the score together."""
    score = grad(log_q, argnums=0)(x, theta, T_phi, b_phi)
    return score, score


def log_q(x, theta, T_phi, b_phi):
    """Unnormalized log q(x|θ) = θᵀT(x) + b(x) for a single x of shape (d_x,)."""
    return T_phi(x) @ theta + b_phi(x).sum()


def single_pair_loss(x, theta, T_phi, b_phi, Sigma_inv):
    """Score-matching loss for one (x, θ) pair."""
    H_logq, score = jacfwd(score_with_aux, argnums=0, has_aux=True)(x, theta, T_phi, b_phi)
    score_term = score @ (Sigma_inv @ score)
    div_term = 2.0 * torch.sum(Sigma_inv * H_logq)
    return score_term + div_term
