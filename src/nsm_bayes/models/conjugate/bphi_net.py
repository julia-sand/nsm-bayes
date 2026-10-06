import torch
from torch import nn


class BphiNet(nn.Module):
    """Network to approximate the b_phi mapping.

    Parameters
    ----------
    input_dim : int
        Dimension of x (D_X).
    hidden_dim : int
        Size of the hidden layers.
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
