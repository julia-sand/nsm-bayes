"""
Corner plot of the score-matching ("Ours") posterior samples, reproducing
the left-hand panel of the figure (NLE omitted).

Takes the directory that holds the pickles written by the normflow run
scripts, e.g.

    data/gnk/
    data/turin/eps=0.1/de=mdn/

and plots ``samples_sm_general_<IND>.pkl`` from it, saving
``corner_ours_<IND>.pdf`` next to it. The true parameter is read from
``configs/<folder>.yaml``, using the deepest folder in the path that has a
config (``data/turin/eps=0.1/de=mdn`` -> ``turin.yaml``). If none matches,
the plot is drawn without the true-theta markers.

Usage:
    python plot_gnk.py data/gnk

Plot settings are the constants below.
"""

from __future__ import annotations

import pickle
import sys
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

# Defaults
IND = 0              # Repetition index to plot
BINS = 30            # Histogram bins on the diagonal
MAX_SCATTER = 500    # Max points in each off-diagonal scatter
LIMITS = None        # Axis limits [(lo1, hi1), (lo2, hi2), ...]; None = automatic


def to_numpy(a) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array."""
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def load_pickle(path: Path) -> object:
    """Load a pickled object from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)


# This file lives at src/nsm_bayes/viz/, so the repo root is three levels up.
CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_theta_true(data_dir: Path) -> np.ndarray | None:
    """Read ``theta_true`` from the experiment's config.

    Configs are named after the experiment they write to (``gnk.yaml`` ->
    ``data/gnk/``), so the deepest folder in *data_dir* with a matching
    ``configs/<folder>.yaml`` is used: ``data/turin/eps=0.1/de=mdn`` ->
    ``turin.yaml``. String entries in the config's ``defaults`` list are
    merged in underneath it, which is how ``sir_undercounting.yaml`` gets
    ``theta_true`` from ``sir.yaml``.

    Returns ``None`` if no config is found or it has no ``theta_true``.
    """
    candidates = (CONFIGS_DIR / f"{part}.yaml"
                  for part in reversed(data_dir.absolute().parts))
    config = next((c for c in candidates if c.is_file()), None)
    if config is None:
        return None

    cfg = OmegaConf.load(config)
    bases = [OmegaConf.load(CONFIGS_DIR / f"{entry}.yaml")
             for entry in cfg.get("defaults", [])
             if isinstance(entry, str) and entry != "_self_"]
    cfg = OmegaConf.merge(*bases, cfg)
    print(f"Using config {config}")
    return np.array(list(cfg.theta_true), dtype=float) if "theta_true" in cfg else None


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
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python plot_gnk.py DATA_DIR   (e.g. data/gnk)")
    data_dir = Path(sys.argv[1])

    pkl_path = data_dir / f"samples_sm_general_{IND}.pkl"
    if not pkl_path.exists():
        available = sorted(q.name for q in data_dir.glob("samples_sm_general_*.pkl"))
        raise SystemExit(f"{pkl_path} not found. Available: {available or 'none'}")

    theta_true = load_theta_true(data_dir)
    if theta_true is None:
        print("No matching config found; plotting without true theta.")

    samples = to_numpy(load_pickle(pkl_path))
    d = theta_true.shape[0] if theta_true is not None else samples.shape[-1]
    samples = samples.reshape(-1, d)
    print(f"Loaded {pkl_path} -> {samples.shape}")

    fig = corner_plot(samples, theta_true=theta_true, limits=LIMITS,
                      bins=BINS, max_scatter=MAX_SCATTER)

    out = data_dir / f"corner_ours_{IND}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()