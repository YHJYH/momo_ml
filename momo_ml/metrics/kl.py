from typing import Iterable, Literal, Tuple
import numpy as np
import pandas as pd


def _probs_from_numeric(
    ref: pd.Series,
    cur: pd.Series,
    buckets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin numeric values using reference quantile edges and return discrete probs.
    """
    # Build bin edges from reference quantiles (stable across time)
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(ref.quantile(quantiles).values)

    # Edge case: if reference lacks variability (all identical), fall back to a single bin
    if edges.size <= 1:
        # everything goes to one bin -> zero KL
        return np.array([1.0]), np.array([1.0]) if cur.size > 0 else np.array([1.0])

    # Use pandas cut to assign bins (right-closed to match histogram default)
    ref_bins = pd.cut(ref, bins=edges, include_lowest=True, right=True)
    cur_bins = pd.cut(cur, bins=edges, include_lowest=True, right=True)

    ref_counts = ref_bins.value_counts(sort=False).values.astype(float)
    cur_counts = cur_bins.value_counts(sort=False).values.astype(float)

    ref_probs = (
        ref_counts / ref_counts.sum()
        if ref_counts.sum() > 0
        else np.zeros_like(ref_counts)
    )
    cur_probs = (
        cur_counts / cur_counts.sum()
        if cur_counts.sum() > 0
        else np.zeros_like(cur_counts)
    )

    return ref_probs, cur_probs


def _probs_from_categorical(
    ref: pd.Series,
    cur: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build aligned probability vectors for categorical data based on union of categories.
    """
    categories = pd.Index(ref.dropna().unique()).union(pd.Index(cur.dropna().unique()))
    ref_counts = (
        ref.dropna()
        .value_counts()
        .reindex(categories, fill_value=0)
        .astype(float)
        .values
    )
    cur_counts = (
        cur.dropna()
        .value_counts()
        .reindex(categories, fill_value=0)
        .astype(float)
        .values
    )

    ref_probs = (
        ref_counts / ref_counts.sum()
        if ref_counts.sum() > 0
        else np.zeros_like(ref_counts)
    )
    cur_probs = (
        cur_counts / cur_counts.sum()
        if cur_counts.sum() > 0
        else np.zeros_like(cur_counts)
    )
    return ref_probs, cur_probs


def compute_kl(
    ref: Iterable,
    cur: Iterable,
    *,
    buckets: int = 10,
    base: Literal["e", "2", "10"] = "e",
    epsilon: float = 1e-12,
) -> float:
    """
    Compute Kullback–Leibler divergence D_KL(P||Q) between reference P and current Q.

    Parameters
    ----------
    ref : Iterable
        Reference sample.
    cur : Iterable
        Current sample.
    buckets : int, default 10
        Number of quantile bins (for numeric data). Ignored for categorical data.
    base : {"e", "2", "10"}, default "e"
        Logarithm base. "e" for nats, "2" for bits, "10" for bans.
    epsilon : float, default 1e-12
        Smoothing constant to handle zero probabilities.

    Returns
    -------
    float
        D_KL(P||Q) >= 0. Larger values indicate greater divergence.
        Note: KL is asymmetric; D_KL(P||Q) != D_KL(Q||P).

    Notes
    -----
    - NaN values are dropped.
    - For numeric data: ref-quantile binning is used for stability across time.
    - For categorical data: frequency-based discrete distributions are used.
    - Epsilon smoothing prevents log(0) and division-by-zero issues.
    """
    ref_s = pd.Series(ref).dropna()
    cur_s = pd.Series(cur).dropna()

    # Determine numeric vs categorical by dtype (object or string -> categorical)
    if pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(cur_s):
        P, Q = _probs_from_numeric(
            ref_s.astype(float), cur_s.astype(float), buckets=buckets
        )
    else:
        P, Q = _probs_from_categorical(ref_s.astype(object), cur_s.astype(object))

    # Smoothing for numerical stability
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    P = np.where(P <= 0, epsilon, P)
    Q = np.where(Q <= 0, epsilon, Q)

    # Normalize again to ensure sum to 1 after smoothing
    P = P / P.sum()
    Q = Q / Q.sum()

    # Choose log base
    if base == "e":
        log_ratio = np.log(P / Q)
    elif base == "2":
        log_ratio = np.log2(P / Q)
    elif base == "10":
        log_ratio = np.log10(P / Q)
    else:
        raise ValueError("base must be one of {'e','2','10'}")

    kl = float(np.sum(P * log_ratio))
    return kl
