"""
Corner plot of the score-matching ("Ours") posterior samples for the g-and-k
experiment, reproducing the left-hand panel of the figure (NLE omitted).

Reads the pickles written by the g-and-k run script:
    data/<experiment_name>/samples_sm_general_<ind>.pkl
and theta_true from the Hydra config (config/gnk.yaml) unless given explicitly.

Usage:
    python plot_corner_gnk.py --ind 0
    python plot_corner_gnk.py --experiment-name my_exp --ind 3 --out corner.pdf
    python plot_corner_gnk.py --theta-true 3 1 2 0.5     # skip reading the config
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # only needed if the pickles hold torch tensors
    torch = None
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Serif / Computer Modern look without needing a LaTeX install
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.labelsize": 16,
})

OURS_COLOR = "#C0504D"


def to_numpy(a):
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_config(config_dir, config_name):
    from omegaconf import OmegaConf
    return OmegaConf.load(Path(config_dir) / f"{config_name}.yaml")


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
    p.add_argument("--data-dir", default="data")
    p.add_argument("--experiment-name", default=None,
                   help="Defaults to experiment_name in the config.")
    p.add_argument("--ind", type=int, default=0, help="Repetition index.")
    p.add_argument("--config-dir", default="config")
    p.add_argument("--config-name", default="gnk")
    p.add_argument("--theta-true", type=float, nargs="+", default=None)
    p.add_argument("--limits", type=float, nargs="+", default=None,
                   help="Axis limits as lo1 hi1 lo2 hi2 ..., e.g. "
                        "-10 3 0 3.2 -5 5 -3 0")
    p.add_argument("--max-scatter", type=int, default=500)
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = None
    if args.experiment_name is None or args.theta_true is None:
        cfg = load_config(args.config_dir, args.config_name)

    exp_name = args.experiment_name or cfg.experiment_name
    theta_true = (np.array(args.theta_true) if args.theta_true is not None
                  else np.array(list(cfg.theta_true), dtype=float))

    save_dir = Path(args.data_dir) / exp_name
    samples = to_numpy(load_pickle(save_dir / f"samples_sm_general_{args.ind}.pkl"))
    d = theta_true.shape[0]
    samples = samples.reshape(-1, d)  # flattens (chains, draws, d) if needed

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