r"""Conjugate surrogate likelihood model for NSM-Bayes.

The surrogate is a neural exponential family

.. math::

    \log q_\phi(x \mid \theta) = \theta^\top T_\phi(x) + b_\phi(x),

which is linear in ``theta``.  Combined with a Gaussian prior and a
(weighted) score-matching loss, this gives a Gaussian generalised posterior
in closed form (see :mod:`nsm_bayes.models.conjugate.conj_posterior`).
Both ``x`` and ``theta`` are standardised before training; the fitted
standardisers are stored on the model.
"""

import torch
from torch import nn

from nsm_bayes.models.conjugate.bphi_net import BphiNet
from nsm_bayes.models.conjugate.conj_train import train_q_phi
from nsm_bayes.models.conjugate.tphi_net import TphiNet


class TorchStandardizer(nn.Module):
    """Per-dimension standardisation ``(x - mean) / std``.

    The statistics are registered as buffers, so they are included in
    ``state_dict()`` and follow ``.to(device)``.

    Parameters
    ----------
    dim : int
        Number of features.
    min_std : float, default 1e-8
        Lower bound applied to the fitted standard deviations, so constant
        features do not cause division by zero.

    Attributes
    ----------
    mean : torch.Tensor
        Fitted per-feature means, shape ``(dim,)``.
    std : torch.Tensor
        Fitted per-feature standard deviations, shape ``(dim,)``.
    fitted : torch.Tensor
        Boolean scalar; ``True`` once :meth:`fit` or ``load_state_dict`` has
        been called.

    Notes
    -----
    The standard deviation uses the unbiased estimator (``ddof=1``), which
    matches the standardiser previously taken from ``sbi``.
    """
 
    def __init__(self, dim: int, min_std: float = 1e-8):
        super().__init__()
        self.min_std = min_std
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.tensor(False))
 
    @torch.no_grad()
    def fit(self, data: torch.Tensor) -> "TorchStandardizer":
        """Estimate the mean and standard deviation from data.

        Parameters
        ----------
        data : torch.Tensor
            Data of shape ``(n, dim)``.

        Returns
        -------
        TorchStandardizer
            ``self``, with the buffers updated in place.

        Raises
        ------
        ValueError
            If `data` is not two-dimensional with ``dim`` columns.
        """
        if data.dim() != 2 or data.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected (n, {self.mean.shape[0]}) data, got {tuple(data.shape)}")
        self.mean.copy_(data.mean(dim=0))
        #unbiased =True matches the old behaviour using sbi standardiser. 
        self.std.copy_(data.std(dim=0, unbiased=True).clamp_min(self.min_std))
        self.fitted.fill_(True)
        return self
 
    def _check(self):
        """Raise ``RuntimeError`` if the statistics have not been set."""
        if not bool(self.fitted):
            raise RuntimeError("TorchStandardizer used before fit() or load_state_dict()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise data.

        Parameters
        ----------
        x : torch.Tensor
            Data of shape ``(..., dim)``.

        Returns
        -------
        torch.Tensor
            ``(x - mean) / std``, same shape as `x`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit` or ``load_state_dict``.
        """
        self._check()
        return (x - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Map standardised data back to the original scale.

        Parameters
        ----------
        z : torch.Tensor
            Standardised data of shape ``(..., dim)``.

        Returns
        -------
        torch.Tensor
            ``z * std + mean``, same shape as `z`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit` or ``load_state_dict``.
        """
        self._check()
        return z * self.std + self.mean
 
 
class ConjugateModel(nn.Module):
    """Neural exponential-family surrogate ``theta^T T_phi(x) + b_phi(x)``.

    Parameters
    ----------
    d_x : int
        Data (summary statistic) dimension.
    d_theta : int
        Parameter dimension.
    hidden_dim : int, default 128
        Hidden-layer width of both networks.

    Attributes
    ----------
    T_phi : TphiNet
        Sufficient-statistic network, ``R^{d_x} -> R^{d_theta}``.
    b_phi : BphiNet
        Base-measure network, ``R^{d_x} -> R``.
    standardizer_x : TorchStandardizer
        Standardiser for the data; both networks act on standardised ``x``.
    standardizer_theta : TorchStandardizer
        Standardiser for the parameters; ``theta`` in the surrogate is the
        standardised parameter.
    train_history : dict
        Training and validation losses from the last call to :meth:`fit`
        (empty before training or after :meth:`load_checkpoint`).

    See Also
    --------
    nsm_bayes.models.conjugate.conj_posterior.ConjugatePosterior :
        Closed-form generalised posterior built on a fitted model.
    """

    def __init__(self, d_x: int, d_theta: int, hidden_dim: int = 128):
        super().__init__()
        self.T_phi = TphiNet(d_x, hidden_dim, d_theta)
        self.b_phi = BphiNet(d_x, hidden_dim)
        self.standardizer_x = TorchStandardizer(d_x)
        self.standardizer_theta = TorchStandardizer(d_theta)
        self.train_history: dict = {}
 
    # ---------------------------------------------------------------- train
    def fit(self, x_sim: torch.Tensor, theta_sim: torch.Tensor) -> "ConjugateModel":
        """Fit the standardisers and train the networks by score matching.

        The model is first moved to the device of `x_sim`. Training uses
        :func:`~nsm_bayes.models.conjugate.conj_train.train_q_phi` with its
        default hyperparameters, on the standardised data.

        Parameters
        ----------
        x_sim : torch.Tensor
            Simulated data of shape ``(N, d_x)``.
        theta_sim : torch.Tensor
            Parameters used to simulate `x_sim`, shape ``(N, d_theta)``.

        Returns
        -------
        ConjugateModel
            ``self``, trained; the loss curves are in :attr:`train_history`.
        """
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
        """Save the networks, standardisers and config to disk.

        Parameters
        ----------
        cfg : Mapping
            Experiment configuration; stored as ``dict(cfg)`` under
            ``"config"``.
        save_dir : pathlib.Path
            Output directory.
        ind : int
            Repetition index, used in the filename.

        See Also
        --------
        load_checkpoint : Restore a model from the written file.

        Notes
        -----
        Writes ``save_dir / f"conjugate_nets_{ind}.pt"`` containing the keys
        ``T_phi_state_dict``, ``B_phi_state_dict``,
        ``standardizer_x_state_dict``, ``standardizer_theta_state_dict`` and
        ``config``. The training history is not saved.
        """
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

    def load_checkpoint(self, path) -> "ConjugateModel":
        """Load networks and standardisers written by :meth:`save_checkpoint`.

        Parameters
        ----------
        path : str or pathlib.Path
            Checkpoint file. Tensors are mapped onto the device of this
            model's parameters.

        Returns
        -------
        ConjugateModel
            ``self``, with weights and standardisers restored.

        Notes
        -----
        The file is read with ``weights_only=False`` because it contains the
        config dictionary, so only load checkpoints you trust.
        """
        device = next(self.parameters()).device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.T_phi.load_state_dict(ckpt["T_phi_state_dict"])
        self.b_phi.load_state_dict(ckpt["B_phi_state_dict"])
        self.standardizer_x.load_state_dict(ckpt["standardizer_x_state_dict"])
        self.standardizer_theta.load_state_dict(ckpt["standardizer_theta_state_dict"])
        return self