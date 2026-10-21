"""Network for the sufficient statistic ``T_phi(x)`` of the conjugate surrogate."""
import torch
from torch import nn


class TphiNet(nn.Module):
    """Vector network ``T_phi: R^{d_x} -> R^{d_theta}``.

    In the conjugate surrogate likelihood
    ``log q(x | theta) = theta^T T_phi(x) + b_phi(x)`` this network is the
    learned sufficient statistic. It is a one-hidden-layer MLP with
    ``tanh`` activation, Xavier-uniform weights and biases initialised to
    ``0.01``.

    Parameters
    ----------
    input_dim : int
        Data dimension ``d_x``.
    hidden_dim : int
        Width of the hidden layer.
    output_dim : int
        Parameter dimension ``d_theta``.

    See Also
    --------
    BphiNet : The base-measure network.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier-uniform weights and constant ``0.01`` biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ``T_phi``.

        Parameters
        ----------
        x : torch.Tensor
            Data of shape ``(..., d_x)``.

        Returns
        -------
        torch.Tensor
            Sufficient statistics of shape ``(..., d_theta)``.
        """
        return self.network(x)
