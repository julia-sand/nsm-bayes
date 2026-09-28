import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
from typing import Dict
from method import calculate_training_loss
    
class TphiNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class BphiNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initializes the b_phi network.
        Args:
            input_dim (int): Dimension of x (D_X).
            hidden_dim (int): Size of the hidden layers.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    



# Function to train q_phi of case 1
def train_q_phi(
    x_sim: torch.Tensor,
    theta: torch.Tensor,
    T_phi_net: nn.Module,
    b_phi_net: nn.Module
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
    # --- 1. Define Fixed Hyperparameters Internally ---
    learning_rate = 5e-4
    num_epochs = 1000  # Max epochs; early stopping will likely trigger first
    batch_size = 128
    weight_decay = 1e-5
    validation_split = 0.2
    early_stopping_patience = 20
    # scheduler_patience = 15
    
    # --- 2. Define Fixed `Sigma_inv` Internally ---
    # Assumes unweighted score matching loss
    d_x = x_sim.shape[1]
    Sigma_inv = torch.eye(d_x, device=x_sim.device)

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
