
import torch
from torch import nn

from nsm_bayes.models.conjugate.bphi_net import BphiNet
from nsm_bayes.models.conjugate.conj_train import train_q_phi
from nsm_bayes.models.conjugate.tphi_net import TphiNet


class TorchStandardizer(nn.Module):
    """Per-dimension (x - mean) / std with statistics stored as buffers, so they are
    saved in state_dict() and follow .to(device). 
    Uses ddof=1 to match old sbi version
    """
 
    def __init__(self, dim: int, min_std: float = 1e-8):
        super().__init__()
        self.min_std = min_std
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.tensor(False))
 
    @torch.no_grad()
    def fit(self, data: torch.Tensor) -> "TorchStandardizer":
        """Compute mean / std from `data` (n, dim) and store them in place."""
        if data.dim() != 2 or data.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected (n, {self.mean.shape[0]}) data, got {tuple(data.shape)}")
        self.mean.copy_(data.mean(dim=0))
        #unbiased =True matches the old behaviour using sbi standardiser. 
        self.std.copy_(data.std(dim=0, unbiased=True).clamp_min(self.min_std))
        self.fitted.fill_(True)
        return self
 
    def _check(self):
        if not bool(self.fitted):
            raise RuntimeError("TorchStandardizer used before fit() or load_state_dict()")
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check()
        return (x - self.mean) / self.std
 
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        self._check()
        return z * self.std + self.mean
 
 
class ConjugateModel(nn.Module):
    def __init__(self, d_x: int, d_theta: int, hidden_dim: int = 128):
        super().__init__()
        self.T_phi = TphiNet(d_x, hidden_dim, d_theta)
        self.b_phi = BphiNet(d_x, hidden_dim)
        self.standardizer_x = TorchStandardizer(d_x)
        self.standardizer_theta = TorchStandardizer(d_theta)
        self.train_history: dict = {}
 
    # ---------------------------------------------------------------- train
    def fit(self, x_sim: torch.Tensor, theta_sim: torch.Tensor) -> "ConjugateModel":
        """Fit the standardizers on the simulations, then train T_phi / b_phi by
        score matching on the standardized data. Returns self."""
        self.to(x_sim.device)
        self.standardizer_x.fit(x_sim)
        self.standardizer_theta.fit(theta_sim)
        self.train_history = train_q_phi(
            x_sim=self.standardizer_x(x_sim),
            theta=self.standardizer_theta(theta_sim),
            T_phi_net=self.T_phi,
            b_phi_net=self.b_phi,
        )
        return self
    
        
    def save_checkpoint(self,cfg,save_dir,ind):
        # Save the trained neural nets
        torch.save({
            "T_phi_state_dict": self.T_phi.state_dict(),
            "B_phi_state_dict": self.b_phi.state_dict(),
            "standardizer_x_state_dict": self.standardizer_x.state_dict(),
            "standardizer_theta_state_dict": self.standardizer_theta.state_dict(),
            "config": dict(cfg),
        },
            save_dir / f"conjugate_nets_{ind}.pt"
        )