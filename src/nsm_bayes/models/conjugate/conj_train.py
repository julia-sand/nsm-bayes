r"""Score-matching training of the conjugate surrogate.

The networks are trained by minimising the (Hyvärinen) score-matching
objective of the unnormalised density
:math:`\log q(x \mid \theta) = \theta^\top T_\phi(x) + b_\phi(x)` over
simulated pairs :math:`(\theta_j, x_j)`:

.. math::

    \mathcal{L}(x, \theta) = \lVert \nabla_x \log q(x \mid \theta)
        \rVert^2_{\Sigma^{-1}}
        + 2 \operatorname{tr}\!\big(\Sigma^{-1} \nabla_x^2
        \log q(x \mid \theta)\big),

with :math:`\Sigma^{-1} = I` during training.  Derivatives with respect to
``x`` are computed with :mod:`torch.func`.
"""
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
    """Train ``T_phi`` and ``b_phi`` by score matching with early stopping.

    Parameters
    ----------
    x_sim : torch.Tensor
        Simulated data of shape ``(N, d_x)``, normally already standardised.
    theta : torch.Tensor
        Corresponding parameters, shape ``(N, d_theta)``.
    T_phi_net : torch.nn.Module
        Sufficient-statistic network; updated in place.
    b_phi_net : torch.nn.Module
        Base-measure network; updated in place.
    learning_rate : float, default 5e-4
        Adam learning rate.
    num_epochs : int, default 1000
        Maximum number of epochs.
    batch_size : int, default 128
        Mini-batch size.
    weight_decay : float, default 1e-5
        Adam (L2) weight decay.
    validation_split : float, default 0.2
        Fraction of the pairs held out, by a random split, for validation.
    early_stopping_patience : int, default 20
        Stop after this many consecutive epochs without an improvement in
        validation loss.

    Returns
    -------
    dict
        ``{"train_losses": list[float], "val_losses": list[float]}``, the
        mean batch loss per epoch.

    See Also
    --------
    calculate_training_loss : The per-batch objective.

    Notes
    -----
    Gradients are clipped to a global norm of 1. When training stops, the
    weights from the epoch with the lowest validation loss are restored
    into both networks. Progress is printed every 10 epochs.
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
    """Mean score-matching loss over a batch of ``(x, theta)`` pairs.

    Parameters
    ----------
    x_batch : torch.Tensor
        Data of shape ``(B, d_x)``.
    theta_batch : torch.Tensor
        Parameters of shape ``(B, d_theta)``.
    T_phi : torch.nn.Module
        Sufficient-statistic network.
    b_phi : torch.nn.Module
        Base-measure network.
    Sigma_inv : torch.Tensor
        Metric of shape ``(d_x, d_x)``; cast to the dtype and device of
        `x_batch`.

    Returns
    -------
    torch.Tensor
        Scalar mean of :func:`single_pair_loss` over the batch.

    Raises
    ------
    AssertionError
        If the batch sizes differ or `Sigma_inv` has the wrong shape.

    Notes
    -----
    The networks are put in eval mode for the computation (gradients are
    still tracked) and their previous train/eval mode is then restored.
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
    """Alternative implementation of :func:`calculate_training_loss`.

    Builds the score and Hessian explicitly from the network derivatives,

    ``score = J_T(x)^T theta + grad b(x)`` and
    ``hess = sum_k theta_k hess T_k(x) + hess b(x)``,

    and returns the batch mean of
    ``score^T Sigma_inv score + 2 tr(Sigma_inv hess)``. It is not used by
    the training loop and is kept for reference.

    Parameters
    ----------
    x_batch : torch.Tensor
        Data of shape ``(B, d_x)``.
    theta_batch : torch.Tensor
        Parameters of shape ``(B, d_theta)``.
    T_phi : torch.nn.Module
        Sufficient-statistic network, ``(d_x,) -> (d_theta,)``.
    b_phi : torch.nn.Module
        Base-measure network, ``(d_x,) -> ()``.
    Sigma_inv : torch.Tensor
        Metric of shape ``(d_x, d_x)``.

    Returns
    -------
    torch.Tensor
        Scalar mean loss over the batch.
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
        """Score-matching loss for one ``(x, theta)`` pair.

        Parameters
        ----------
        xi : torch.Tensor
            Single observation of shape ``(d_x,)``.
        thetai : torch.Tensor
            Parameter vector of shape ``(d_theta,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

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
    """Score ``grad_x log q(x | theta)``, returned twice.

    Returning the score as both output and auxiliary value lets
    ``jacfwd(..., has_aux=True)`` produce the Hessian and the score in a
    single pass.

    Parameters
    ----------
    x : torch.Tensor
        Single observation of shape ``(d_x,)``.
    theta : torch.Tensor
        Parameter vector of shape ``(d_theta,)``.
    T_phi, b_phi : torch.nn.Module
        Surrogate networks.

    Returns
    -------
    score, score : tuple of torch.Tensor
        Two references to the score, each of shape ``(d_x,)``.
    """
    score = grad(log_q, argnums=0)(x, theta, T_phi, b_phi)
    return score, score


def log_q(x, theta, T_phi, b_phi):
    """Unnormalised surrogate log-density ``theta^T T_phi(x) + b_phi(x)``.

    Parameters
    ----------
    x : torch.Tensor
        Single observation of shape ``(d_x,)``.
    theta : torch.Tensor
        Parameter vector of shape ``(d_theta,)``.
    T_phi, b_phi : torch.nn.Module
        Surrogate networks.

    Returns
    -------
    torch.Tensor
        Scalar log-density (up to the normalising constant).
    """
    return T_phi(x) @ theta + b_phi(x).sum()


def single_pair_loss(x, theta, T_phi, b_phi, Sigma_inv):
    """Score-matching loss for a single ``(x, theta)`` pair.

    Parameters
    ----------
    x : torch.Tensor
        Single observation of shape ``(d_x,)``.
    theta : torch.Tensor
        Parameter vector of shape ``(d_theta,)``.
    T_phi, b_phi : torch.nn.Module
        Surrogate networks.
    Sigma_inv : torch.Tensor
        Metric of shape ``(d_x, d_x)``.

    Returns
    -------
    torch.Tensor
        Scalar ``s^T Sigma_inv s + 2 * sum(Sigma_inv * H)``, where ``s`` and
        ``H`` are the score and Hessian of ``log q`` with respect to ``x``.
    """
    H_logq, score = jacfwd(score_with_aux, argnums=0, has_aux=True)(x, theta, T_phi, b_phi)
    score_term = score @ (Sigma_inv @ score)
    div_term = 2.0 * torch.sum(Sigma_inv * H_logq)
    return score_term + div_term
