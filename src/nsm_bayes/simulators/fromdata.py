"""A simulator that serves pre-simulated data from ``.pkl`` files.

:class:`PickleSimulator` has the same interface as the benchmark
simulators (``prior``, ``simulate``, ``generate_training_data`` and
``generate_observed_data``), but instead of simulating it reads data that
was simulated elsewhere from a directory of pickle files.  The file names
below are the defaults; each can be changed with the matching
``*_file`` argument of :class:`PickleSimulator`:

==================== ========================== ========================
default file name    shape                      contents
==================== ========================== ========================
``theta.pkl``        ``(N, d_theta)``           parameters (required)
``x_sim.pkl``        ``(N, d_x)``               simulations at ``theta``
                                                (required)
``x_obs.pkl``        ``(M, d_x)``               observed data (required)
``theta_true.pkl``   ``(d_theta,)``             parameter ``x_obs`` was
                                                generated at (optional)
``prior_mean.pkl``   ``(d_theta,)``             Gaussian prior mean
                                                (optional)
``prior_cov.pkl``    ``(d_theta, d_theta)``     Gaussian prior covariance
                                                (optional)
==================== ========================== ========================

Row ``i`` of ``x_sim`` is a simulation at row ``i`` of ``theta``.  Each
file holds one array: a ``torch.Tensor``, ``numpy.ndarray``, nested list
or anything with a ``to_numpy()`` method (e.g. a pandas ``DataFrame``).
Arrays are converted to ``float32``; 1D arrays of samples are treated as
one-dimensional data, i.e. reshaped to ``(n, 1)``.

Files are read with :mod:`pickle`, which can execute arbitrary code: only
load files you trust.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.distributions import MultivariateNormal

__all__ = ["PickleSimulator"]

_log = logging.getLogger(__name__)

class PickleSimulator:
    """Simulator backed by pre-simulated data in a directory of ``.pkl`` files.

    The prior is read from the prior mean and covariance files when both
    exist; otherwise it is a Gaussian fitted to the stored parameters (with
    a warning), on the assumption that they are prior draws.

    Attributes
    ----------
    data_dir : pathlib.Path
        Directory the data is read from.
    files : dict[str, str or None]
        File name of each array, keyed by ``theta``, ``x_sim``, ``x_obs``,
        ``theta_true``, ``prior_mean`` and ``prior_cov``.
    prior_mean : torch.Tensor
        Prior mean vector, shape ``(d_theta,)``.
    prior_cov : torch.Tensor
        Prior covariance matrix, shape ``(d_theta, d_theta)``.
    prior : torch.distributions.MultivariateNormal
        Gaussian prior over ``theta``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        theta_file: str = "theta.pkl",
        x_sim_file: str = "x_sim.pkl",
        x_obs_file: str = "x_obs.pkl",
        theta_true_file: str | None = "theta_true.pkl",
        prior_mean_file: str | None = "prior_mean.pkl",
        prior_cov_file: str | None = "prior_cov.pkl",
    ) -> None:
        """Load the data in `data_dir` and build the prior.

        File names are relative to `data_dir`.  An optional file that does
        not exist is skipped; ``None`` skips it explicitly.

        Parameters
        ----------
        data_dir : str or pathlib.Path
            Directory containing the data files.
        theta_file : str, default "theta.pkl"
            Parameters, shape ``(N, d_theta)``.
        x_sim_file : str, default "x_sim.pkl"
            Simulations at each row of ``theta``, shape ``(N, d_x)``.
        x_obs_file : str, default "x_obs.pkl"
            Observed data, shape ``(M, d_x)``.
        theta_true_file : str or None, default "theta_true.pkl"
            Parameter the observed data was generated at, shape
            ``(d_theta,)`` (optional).
        prior_mean_file : str or None, default "prior_mean.pkl"
            Gaussian prior mean, shape ``(d_theta,)`` (optional).
        prior_cov_file : str or None, default "prior_cov.pkl"
            Gaussian prior covariance, shape ``(d_theta, d_theta)``
            (optional).
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.files = {
            "theta": theta_file,
            "x_sim": x_sim_file,
            "x_obs": x_obs_file,
            "theta_true": theta_true_file,
            "prior_mean": prior_mean_file,
            "prior_cov": prior_cov_file,
        }
        for key in ("theta", "x_sim", "x_obs"):
            if not self.files[key]:
                raise ValueError(f"{key}_file is required")
        f = self.files

        self._theta = self._load_samples("theta")
        self._x_sim = self._load_samples("x_sim")
        self._x_obs = self._load_samples("x_obs")
        self._theta_true = self._load("theta_true", required=False)

        n_sim, d_theta = self._theta.shape
        d_x = self._x_sim.shape[1]
        if self._x_sim.shape[0] != n_sim:
            raise ValueError(
                f"{f['theta']} has {n_sim} rows but {f['x_sim']} has {self._x_sim.shape[0]}"
            )
        if self._x_obs.shape[1] != d_x:
            raise ValueError(
                f"{f['x_obs']} has d_x={self._x_obs.shape[1]} but {f['x_sim']} has d_x={d_x}"
            )
        if self._theta_true is not None:
            self._theta_true = self._theta_true.reshape(-1)
            if self._theta_true.numel() != d_theta:
                raise ValueError(
                    f"{f['theta_true']} has {self._theta_true.numel()} entries, "
                    f"expected d_theta={d_theta}"
                )

        prior_mean = self._load("prior_mean", required=False)
        prior_cov = self._load("prior_cov", required=False)
        if (prior_mean is None) != (prior_cov is None):
            raise ValueError(
                f"Provide both {f['prior_mean']} and {f['prior_cov']}, or neither."
            )
        if prior_mean is None:
            _log.warning(
                "No prior mean / covariance files in %s: using a Gaussian "
                "fitted to %s as the prior.", self.data_dir, f["theta"],
            )
            prior_mean = self._theta.mean(dim=0)
            prior_cov = torch.atleast_2d(torch.cov(self._theta.T))

        self.prior_mean = prior_mean.reshape(-1)
        self.prior_cov = torch.atleast_2d(prior_cov)
        if self.prior_mean.numel() != d_theta or self.prior_cov.shape != (d_theta, d_theta):
            raise ValueError(
                f"Prior has mean {tuple(self.prior_mean.shape)} and covariance "
                f"{tuple(self.prior_cov.shape)}, but theta has dimension {d_theta}"
            )
        self.prior = MultivariateNormal(loc=self.prior_mean,
                                        covariance_matrix=self.prior_cov)

    # ------------------------------------------------------------------ #
    # Loading                                                             #
    # ------------------------------------------------------------------ #
    def _load(self, key: str, required: bool = True) -> torch.Tensor | None:
        """Load the file for `key` (see ``self.files``) as a finite float32 tensor."""
        name = self.files[key]
        if name is None:
            return None
        path = self.data_dir / name
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Required data file not found: {path}")
            return None
        with open(path, "rb") as fh:
            obj: Any = pickle.load(fh)
        if hasattr(obj, "to_numpy") and not isinstance(obj, torch.Tensor):
            obj = obj.to_numpy()
        try:
            t = torch.as_tensor(obj).detach().to(dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as err:
            raise TypeError(
                f"Could not convert {name} (type {type(obj).__name__}) to a tensor"
            ) from err
        if not torch.isfinite(t).all():
            raise ValueError(f"{name} contains non-finite values")
        return t

    def _load_samples(self, key: str) -> torch.Tensor:
        """Load a required array of samples as shape ``(n, d)``."""
        t = self._load(key)
        if t.ndim == 1:
            return t.unsqueeze(-1)
        if t.ndim != 2:
            raise ValueError(
                f"{self.files[key]} must have shape (n, d) (or (n,) for scalar data), "
                f"got {tuple(t.shape)}"
            )
        return t

    # ------------------------------------------------------------------ #
    # Simulator interface                                                 #
    # ------------------------------------------------------------------ #
    def simulate(
        self,
        theta: torch.Tensor,
        n: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Return the stored simulations for parameter vector(s) `theta`.

        Data in the files cannot be re-simulated, so every row of `theta`
        must exactly match a stored parameter row (as returned by
        :meth:`generate_training_data`).

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector of shape ``(d_theta,)`` or a batch of shape
            ``(n_theta, d_theta)``.
        n : int, default 1
            Number of simulations per parameter vector; each row of
            `theta` must appear at least `n` times in the stored parameters.
        **kwargs : dict
            Additional arguments (ignored).

        Returns
        -------
        torch.Tensor
            Simulations of shape ``(n_theta, d_x)`` for `n == 1` or
            ``(n_theta, n, d_x)`` for `n > 1`; the leading ``n_theta``
            dimension is dropped for a 1D `theta`.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        single = theta.dim() == 1
        if single:
            theta = theta.unsqueeze(0)

        out = []
        for row in theta:
            idx = torch.nonzero((self._theta == row).all(dim=1)).squeeze(-1)
            if idx.numel() < n:
                raise ValueError(
                    f"theta {row.tolist()} has {idx.numel()} stored simulation(s) in "
                    f"{self.files['theta']}, {n} requested; PickleSimulator can only return "
                    "stored simulations"
                )
            out.append(self._x_sim[idx[:n]])
        x = torch.stack(out, dim=0)  # (n_theta, n, d_x)

        if n == 1:
            x = x.squeeze(1)
        return x.squeeze(0) if single else x

    def generate_training_data(
        self,
        n: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw `n` stored ``(theta, x_sim)`` pairs without replacement.

        The pairs are a random subset (using the torch global RNG, so
        ``set_seed`` makes it reproducible).

        Parameters
        ----------
        n : int
            Number of training samples.
        **kwargs : dict
            Additional arguments (ignored).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (theta, x_sim), where theta is ``(n, d_theta)`` and
            x_sim is ``(n, d_x)``.
        """
        n_sim = self._theta.shape[0]
        if n > n_sim:
            raise ValueError(f"Requested {n} training samples but {self.files['theta']} only has {n_sim}")
        idx = torch.randperm(n_sim)[:n]
        return self._theta[idx].clone(), self._x_sim[idx].clone()

    def generate_observed_data(
        self,
        theta: torch.Tensor,
        n_obs: int,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Draw observed data with potential outliers.

        `n_obs` stored observations are drawn without replacement, and a
        fraction `epsilon` of them is replaced by stored simulations under
        random prior parameters, as in the
        benchmark simulators.

        Parameters
        ----------
        theta : torch.Tensor
            True parameter vector.  If a ``theta_true`` file was loaded it must
            match it, since the stored observations cannot be re-simulated
            at another parameter.
        n_obs : int
            Number of observations.
        epsilon : float, default 0.0
            Fraction of observations to replace with outliers.

        Returns
        -------
        torch.Tensor
            Observed data of shape ``(n_obs, d_x)``.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1)
        if self._theta_true is not None and not torch.allclose(theta, self._theta_true):
            raise ValueError(
                f"theta {theta.tolist()} does not match {self.files['theta_true']} "
                f"{self._theta_true.tolist()}, the parameter {self.files['x_obs']} "
                "was generated at"
            )
        n_avail = self._x_obs.shape[0]
        if n_obs > n_avail:
            raise ValueError(f"Requested {n_obs} observations but {self.files['x_obs']} only has {n_avail}")

        x_obs = self._x_obs[torch.randperm(n_avail)[:n_obs]].clone()

        n_out = int(torch.floor(torch.tensor(epsilon * n_obs)).item())
        outlier_indices = torch.randperm(n_obs)[:n_out]

        if n_out > 0:
            x_noise = self._x_sim[torch.randperm(self._x_sim.shape[0])[:n_out]]
            x_obs[outlier_indices] = x_noise

        return x_obs