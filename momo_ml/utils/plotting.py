import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


# ------------------------------------------------------------
# Utility: Create a new figure with consistent styling
# ------------------------------------------------------------
def _create_figure(figsize: Tuple[int, int] = (6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ------------------------------------------------------------
# 1. Distribution Comparison Plot
# ------------------------------------------------------------
def plot_distribution(
    ref: np.ndarray,
    cur: np.ndarray,
    bins: int = 20,
    title: str = "Prediction Distribution Comparison",
) -> plt.Figure:
    """
    Plot overlapping histograms of reference vs current distributions.

    Parameters
    ----------
    ref : np.ndarray
        Reference data.
    cur : np.ndarray
        Current data.
    bins : int
        Number of histogram bins.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _create_figure()

    combined = np.concatenate([ref, cur])
    hist_bins = np.linspace(combined.min(), combined.max(), bins + 1)

    ax.hist(ref, bins=hist_bins, alpha=0.6, label="Reference", density=True)
    ax.hist(cur, bins=hist_bins, alpha=0.6, label="Current", density=True)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


# ------------------------------------------------------------
# 2. Decile Shift Plot
# ------------------------------------------------------------
def plot_deciles(
    ref_deciles: np.ndarray,
    cur_deciles: np.ndarray,
    title: str = "Decile Shift",
) -> plt.Figure:
    """
    Plot decile boundaries or means from reference vs current.

    Parameters
    ----------
    ref_deciles : np.ndarray
        Reference decile edges or values.
    cur_deciles : np.ndarray
        Current decile edges or values.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _create_figure()

    idx = np.arange(len(ref_deciles))

    ax.plot(idx, ref_deciles, marker="o", label="Reference")
    ax.plot(idx, cur_deciles, marker="o", label="Current")

    ax.set_xticks(idx)
    ax.set_xlabel("Decile")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


# ------------------------------------------------------------
# 3. Feature Histogram Overlay
# ------------------------------------------------------------
def plot_feature_histograms(
    ref: np.ndarray,
    cur: np.ndarray,
    feature_name: str,
    bins: int = 20,
) -> plt.Figure:
    """
    Overlay histograms for a given feature.

    Parameters
    ----------
    ref : np.ndarray
        Reference values of a feature.
    cur : np.ndarray
        Current values of the same feature.
    feature_name : str
    bins : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _create_figure()

    combined = np.concatenate([ref, cur])
    hist_bins = np.linspace(combined.min(), combined.max(), bins + 1)

    ax.hist(ref, bins=hist_bins, alpha=0.5, label="Reference", density=True)
    ax.hist(cur, bins=hist_bins, alpha=0.5, label="Current", density=True)

    ax.set_title(f"Feature Drift: {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


# ------------------------------------------------------------
# 4. PSI Contribution Plot
# ------------------------------------------------------------
def plot_psi_buckets(
    breakpoints: np.ndarray,
    ref_dist: np.ndarray,
    cur_dist: np.ndarray,
    title: str = "PSI Bucket Contributions",
) -> plt.Figure:
    """
    Visualize PSI contributions from each bucket.

    Parameters
    ----------
    breakpoints : np.ndarray
        Quantile breakpoints used for PSI.
    ref_dist : np.ndarray
        Reference distribution per bucket.
    cur_dist : np.ndarray
        Current distribution per bucket.
    title : str

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    PSI contribution per bucket = (ref - cur) * log(ref / cur)
    """
    eps = 1e-12
    ref_s = np.where(ref_dist <= 0, eps, ref_dist)
    cur_s = np.where(cur_dist <= 0, eps, cur_dist)

    psi_contrib = (ref_s - cur_s) * np.log(ref_s / cur_s)
    bucket_labels = [
        f"{round(breakpoints[i], 3)}–{round(breakpoints[i+1],3)}"
        for i in range(len(breakpoints) - 1)
    ]

    fig, ax = _create_figure((8, 4))
    idx = np.arange(len(psi_contrib))

    ax.bar(idx, psi_contrib)
    ax.set_xticks(idx)
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.set_ylabel("PSI Contribution")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    return fig
