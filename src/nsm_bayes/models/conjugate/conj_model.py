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

import logging
from pathlib import Path

import torch
from torch import nn

from nsm_bayes.models.conjugate.bphi_net import BphiNet
from nsm_bayes.models.conjugate.conj_train import train_q_phi
from nsm_bayes.models.conjugate.tphi_net import TphiNet

_log = logging.getLogger(__name__)


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
 
 
class EnergyBasedModel(nn.Module):
    """Neural exponential-family surrogate ``theta^T T_phi(x) + b_phi(x)``.

    The model can be constructed in two ways:

    - **From dimensions** -- give ``d_x`` and ``d_theta`` (both required;
      ``hidden_dim`` defaults to 128). The networks are freshly initialised
      (Xavier-uniform, biases ``0.01``) and both standardisers are unfitted.
    - **From a checkpoint** -- give ``checkpoint`` (a path written by
      :meth:`save_checkpoint`). The dimensions and the dtype of the saved
      weights are read from the file, so no data or config is needed. Any
      dimensions also passed must match the checkpoint.

    Parameters
    ----------
    d_x : int, optional
        Data (summary statistic) dimension. Required unless `checkpoint` is
        given; when given together with `checkpoint` it must match the
        checkpoint's value.
    d_theta : int, optional
        Parameter dimension. Same rules as `d_x`.
    hidden_dim : int, optional
        Hidden-layer width of both networks. Defaults to 128 when building
        from dimensions; when `checkpoint` is given it must match the
        checkpoint's value (or be omitted).
    checkpoint : str or pathlib.Path, optional
        Checkpoint file written by :meth:`save_checkpoint`. Tensors are
        mapped to `device`.
    device : str or torch.device, default "cpu"
        Device the parameters are placed on.

    Returns
    -------
    EnergyBasedModel
        A model whose :attr:`d_x`, :attr:`d_theta` and :attr:`hidden_dim`
        match the dimensions above; in checkpoint mode the weights and
        standardiser buffers are exactly those of the file.

    Raises
    ------
    ValueError
        If no checkpoint is given and either dimension is missing, or if a
        dimension passed together with a checkpoint does not match the
        checkpoint.

    Examples
    --------
    >>> model = EnergyBasedModel(d_x=4, d_theta=3)          # doctest: +SKIP
    >>> model = EnergyBasedModel(checkpoint="nets.pt")       # doctest: +SKIP

    Notes
    -----
    Built-from-dimensions models are bit-identical to the historical
    ``EnergyBasedModel(d_x, d_theta, hidden_dim=128)`` constructor: the same
    random numbers initialise the same weights in the same order.

    See Also
    --------
    nsm_bayes.models.conjugate.conj_posterior.ConjugatePosterior :
        Closed-form generalised posterior built on a fitted model.
    """

    def __init__(self, d_x: int | None = None, d_theta: int | None = None,
                 hidden_dim: int | None = None,
                 checkpoint: str | Path | None = None,
                 device: str | torch.device = "cpu"):
        super().__init__()
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            d_x_ckpt, d_theta_ckpt, hidden_dim_ckpt = self._infer_dimensions(
                ckpt["T_phi_state_dict"], ckpt["standardizer_theta_state_dict"]
            )
            for name, passed, ckpt_val in (
                ("d_x", d_x, d_x_ckpt),
                ("d_theta", d_theta, d_theta_ckpt),
                ("hidden_dim", hidden_dim, hidden_dim_ckpt),
            ):
                if passed is not None and passed != ckpt_val:
                    raise ValueError(
                        f"{name}={passed!r} does not match the checkpoint "
                        f"({name}={ckpt_val})"
                    )
            d_x, d_theta, hidden_dim = d_x_ckpt, d_theta_ckpt, hidden_dim_ckpt
        else:
            if d_x is None or d_theta is None:
                raise ValueError(
                    "d_x and d_theta are required when no checkpoint is given"
                )
            if hidden_dim is None:
                hidden_dim = 128

        self.T_phi = TphiNet(d_x, hidden_dim, d_theta).to(device)
        self.b_phi = BphiNet(d_x, hidden_dim).to(device)
        self.standardizer_x = TorchStandardizer(d_x).to(device)
        self.standardizer_theta = TorchStandardizer(d_theta).to(device)
        self.train_history: dict = {}

        if checkpoint is not None:
            dtype = ckpt["T_phi_state_dict"]["network.0.weight"].dtype
            self._load_state(ckpt, dtype)

    # ------------------------------------------------------------- properties
    @property
    def d_x(self) -> int:
        """Data (summary statistic) dimension, from the standardiser's mean."""
        return int(self.standardizer_x.mean.shape[0])

    @property
    def d_theta(self) -> int:
        """Parameter dimension, from the parameter standardiser's mean."""
        return int(self.standardizer_theta.mean.shape[0])

    @property
    def hidden_dim(self) -> int:
        """Hidden-layer width of both networks."""
        return int(self.T_phi.network[0].out_features)
 
    # ---------------------------------------------------------------- train
    def fit(self, x_sim: torch.Tensor, theta_sim: torch.Tensor,
            save_file=None,
            num_epochs=1000) -> "EnergyBasedModel":
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
        EnergyBasedModel
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
            num_epochs=num_epochs
        )

        #stop tracking gradients 
        self.requires_grad_(False)

        if save_file is not None:
            self.save_checkpoint(save_file)

        return self
    
        
    def save_checkpoint(self, save_file):
        """Save the networks and standardisers to disk.

        Parameters
        ----------
        save_file : str or pathlib.Path
            Full path of the checkpoint file to write.

        See Also
        --------
        load_checkpoint : Restore a model from the written file.

        Notes
        -----
        Writes ``save_file`` containing the keys
        ``T_phi_state_dict``, ``B_phi_state_dict``,
        ``standardizer_x_state_dict``, ``standardizer_theta_state_dict``.
        """
        # Save the trained neural nets
        torch.save({
            "T_phi_state_dict": self.T_phi.state_dict(),
            "B_phi_state_dict": self.b_phi.state_dict(),
            "standardizer_x_state_dict": self.standardizer_x.state_dict(),
            "standardizer_theta_state_dict": self.standardizer_theta.state_dict(),
        },
            save_file
        )

    def load_checkpoint(self, path) -> "EnergyBasedModel":
        """Load networks and standardisers written by :meth:`save_checkpoint`.

        Parameters
        ----------
        path : str or pathlib.Path
            Checkpoint file. Tensors are mapped onto the device of this
            model's parameters.

        Returns
        -------
        EnergyBasedModel
            ``self``, with weights and standardisers restored. If the
            checkpoint's dimensions differ from this instance, a new
            :class:`EnergyBasedModel` of the correct size is constructed
            and returned instead.

        Notes
        -----
        The file is read with ``weights_only=False``; only load checkpoints
        you trust.
        """
        device = next(self.parameters()).device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        d_x_ckpt, d_theta_ckpt, hidden_dim = self._infer_dimensions(
            ckpt["T_phi_state_dict"], ckpt["standardizer_theta_state_dict"]
        )

        # Compare against this instance's actual network shape. If it
        # matches, load in place; otherwise build a fresh model of the
        # right size (keeping the checkpoint's hidden_dim) and load there.
        if (d_x_ckpt == self.d_x and d_theta_ckpt == self.d_theta
                and hidden_dim == self.hidden_dim):
            target = self
        else:
            target = EnergyBasedModel(
                d_x_ckpt, d_theta_ckpt, hidden_dim
            ).to(device)

        target._load_state(ckpt, ckpt["T_phi_state_dict"]["network.0.weight"].dtype)
        
        #disable gradients (on the loaded model, which may be a fresh instance)
        target.requires_grad_(False) 
        return target

    # ------------------------------------------------------------ checkpoint
    def _load_state(self, ckpt: dict, dtype: torch.dtype) -> None:
        """Load the four state dicts of `ckpt` into this model in place.

        Parameters
        ----------
        ckpt : dict
            Mapping returned by ``torch.load`` of a file written by
            :meth:`save_checkpoint`, with the keys
            ``T_phi_state_dict``, ``B_phi_state_dict``,
            ``standardizer_x_state_dict``, ``standardizer_theta_state_dict``.
        dtype : torch.dtype
            Dtype of the checkpoint's weight tensors; the loaded state is
            cast to it, so a checkpoint saves/loads bit-identically to the
            model that wrote it.
        """
        ckpt = {
            key: {n: v.to(dtype) for n, v in state.items()}
            for key, state in ckpt.items()
            if isinstance(state, dict)
        }
        self.T_phi.load_state_dict(ckpt["T_phi_state_dict"])
        self.b_phi.load_state_dict(ckpt["B_phi_state_dict"])
        self.standardizer_x.load_state_dict(ckpt["standardizer_x_state_dict"])
        self.standardizer_theta.load_state_dict(ckpt["standardizer_theta_state_dict"])

    @staticmethod
    def _infer_dimensions(t_phi_state: dict, std_theta_state: dict
                          ) -> tuple[int, int, int]:
        """Infer network dimensions from a ``TphiNet`` state dict.

        ``TphiNet`` is ``nn.Sequential(Linear, Tanh, Linear)``, so the
        state dict contains exactly two 2-D weight tensors: the first has
        shape ``(hidden_dim, d_x)`` and the second ``(d_theta, hidden_dim)``.

        Parameters
        ----------
        t_phi_state : dict
            State dict written for ``T_phi`` by :meth:`save_checkpoint`.
        std_theta_state : dict
            State dict written for ``standardizer_theta``; used as a
            fallback for ``d_theta`` if no linear weight is found.

        Returns
        -------
        tuple[int, int, int]
            ``(d_x, d_theta, hidden_dim)``.
        """
        linears = [v for v in t_phi_state.values() if v.ndim == 2]
        if len(linears) != 2:
            raise ValueError(
                f"Cannot infer T_phi dimensions: expected exactly two 2-D "
                f"linear weights in the state dict, found {len(linears)}."
            )
        hidden_dim, d_x = linears[0].shape
        d_theta = linears[1].shape[0]

        # Sanity check: the two linear layers must chain.
        if linears[1].shape[1] != hidden_dim:
            raise ValueError(
                "Cannot infer T_phi dimensions: linear layer shapes do not "
                f"chain ({tuple(linears[0].shape)} -> {tuple(linears[1].shape)})."
            )

        if linears[1].shape[0] != int(std_theta_state["mean"].numel()):
            # Warn only -- the standardiser may have been built with a
            # different d_theta in an older checkpoint format.
            _log.warning(
                "Inferred d_theta=%d from T_phi weights does not match "
                "standardizer_theta size=%d.",
                d_theta, int(std_theta_state["mean"].numel()),
            )
        return d_x, d_theta, hidden_dim