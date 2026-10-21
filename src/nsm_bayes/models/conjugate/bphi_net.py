"""Network for the base-measure term ``b_phi(x)`` of the conjugate surrogate."""
import torch
from torch import nn


class BphiNet(nn.Module):
    """Scalar network ``b_phi: R^{d_x} -> R``.

    In the conjugate surrogate likelihood
    ``log q(x | theta) = theta^T T_phi(x) + b_phi(x)`` this network is the
    log base measure. It is a one-hidden-layer MLP with ``tanh``
    activation, Xavier-uniform weights and biases initialised to ``0.01``.

    Parameters
    ----------
    input_dim : int
        Data dimension ``d_x``.
    hidden_dim : int
        Width of the hidden layer.

    See Also
    --------
    TphiNet : The sufficient-statistic network.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
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
        """Evaluate ``b_phi``.

        Parameters
        ----------
        x : torch.Tensor
            Data of shape ``(..., d_x)``.

        Returns
        -------
        torch.Tensor
            Values of shape ``(...)`` (the trailing unit dimension is
            squeezed).
        """
        return self.network(x).squeeze(-1)
