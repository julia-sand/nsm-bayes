"""
Corner plot of the score-matching ("Ours") posterior samples, reproducing
the left-hand panel of the figure (NLE omitted).

Reads the pickles written by the normflow run scripts and resolves the data
directory from the YAML config:

    data/<experiment_name>/samples_sm_general_<ind>.pkl
    data/<experiment_name>/eps=<epsilon>/de=<de>/samples_sm_general_<ind>.pkl

The second layout is used when the config has ``epsilon`` and
``density_estimator`` keys (e.g. the Turin experiment).

Usage:
    python plot_gnk.py --config configs/gnk.yaml --ind 0
    python plot_gnk.py --config configs/turin.yaml --ind 3 --out corner.pdf
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omegaconf import OmegaConf

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.labelsize": 16,
})

OURS_COLOR = "#C0504D"


def to_numpy(a) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array."""
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def load_pickle(path: Path) -> object:
    """Load a pickled object from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_config(config_path: str | Path) -> OmegaConf.DictConfig:
    """Load a YAML config file and resolve Hydra-style ``defaults``.

    If the file contains a ``defaults`` list with string entries (e.g.
    ``- _base``), each corresponding ``<entry>.yaml`` in the same
    directory is loaded and merged (child overrides base).

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    OmegaConf.DictConfig
        Fully resolved configuration.
    """
    cfg_path = Path(config_path)
    cfg = OmegaConf.load(cfg_path)
    if "defaults" in cfg:
        defaults = cfg.get("defaults")
        if defaults:
            resolved = []
            for entry in defaults:
                if isinstance(entry, str) and entry != "_self_":
                    base_path = cfg_path.parent / f"{entry}.yaml"
                    if base_path.exists():
                        resolved.append(OmegaConf.load(base_path))
            if resolved:
                cfg = OmegaConf.merge(*resolved, cfg)
    return cfg


def resolve_save_dir(cfg: OmegaConf.DictConfig, data_root: str | Path) -> Path:
    """Resolve the data directory for a config.

    Standard layout:
        data/<experiment_name>/
    Turin-style layout (when ``epsilon`` and ``density_estimator`` are
    present in the config):
        data/<experiment_name>/eps=<epsilon>/de=<density_estimator>/

    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        Resolved configuration.
    data_root : str or Path
        Root data directory (e.g. ``"data"``).

    Returns
    -------
    Path
        Directory containing the sample pickles.
    """
    save_dir = Path(data_root) / cfg.experiment_name
    if "epsilon" in cfg and "density_estimator" in cfg:
        save_dir = save_dir / f"eps={cfg.epsilon}" / f"de={cfg.density_estimator}"
    return save_dir


def corner_plot(samples, theta_true=None, labels=None, limits=None,
                bins=30, max_scatter=500, seed=0, color=OURS_COLOR):
    """Lower-triangular corner plot: histograms on the diagonal, scatter below."""
    n, d = samples.shape
    labels = labels or [rf"$\theta_{{{i + 1}}}$" for i in range(d)]
    if limits is None:
        lo, hi = np.percentile(samples, [0.5, 99.5], axis=0)
        if theta_true is not None:
            lo, hi = np.minimum(lo, theta_true), np.maximum(hi, theta_true)
        pad = 0.08 * (hi - lo)
        limits = list(zip(lo - pad, hi + pad))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_scatter, n), replace=False)
    sub = samples[idx]

    fig, axes = plt.subplots(d, d, figsize=(2.0 * d, 2.0 * d))

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            if i == j:
                ax.hist(samples[:, i], bins=bins, range=limits[i],
                        color=color, alpha=0.85)
                if theta_true is not None:
                    ax.axvline(theta_true[i], color="k", ls="--", lw=2)
                ax.set_yticks([])
            else:
                ax.scatter(sub[:, j], sub[:, i], s=8, color=color,
                           alpha=0.6, linewidths=0)
                if theta_true is not None:
                    ax.scatter(theta_true[j], theta_true[i], marker="x",
                               s=60, color="k", linewidths=2, zorder=3)
                ax.set_ylim(limits[i])

            ax.set_xlim(limits[j])

            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j == 0 and i == 0:
                ax.set_ylabel(labels[0])
            elif i != j:
                ax.set_yticklabels([])

    # Legend in the empty cell right of the last diagonal-but-one panel
    handles = [Patch(color=color, alpha=0.85, label="Ours")]
    if theta_true is not None:
        handles.append(Line2D([0], [0], color="k", ls="--", lw=2,
                              label=r"True $\theta$"))
    axes[d - 2, d - 1].legend(handles=handles, loc="center", frameon=True,
                              fontsize=14)

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to the YAML config file.")
    p.add_argument("--ind", type=int, default=0, help="Repetition index.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--limits", type=float, nargs="+", default=None,
                   help="Axis limits as lo1 hi1 lo2 hi2 ...")
    p.add_argument("--max-scatter", type=int, default=500)
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    save_dir = resolve_save_dir(cfg, args.data_dir)

    theta_true = np.array(list(cfg.theta_true), dtype=float)

    pkl_path = save_dir / f"samples_sm_general_{args.ind}.pkl"
    samples = to_numpy(load_pickle(pkl_path))
    d = theta_true.shape[0]
    samples = samples.reshape(-1, d)
    print(f"Loaded {pkl_path} -> {samples.shape}")

    limits = None
    if args.limits is not None:
        assert len(args.limits) == 2 * d, "need lo/hi pair per parameter"
        limits = list(zip(args.limits[0::2], args.limits[1::2]))

    fig = corner_plot(samples, theta_true=theta_true, limits=limits,
                      bins=args.bins, max_scatter=args.max_scatter)

    out = args.out or str(save_dir / f"corner_ours_{args.ind}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()