from typing import Iterable, Literal, Tuple
import warnings
import numpy as np
import pandas as pd


def _probs_from_numeric(
    ref: pd.Series,
    cur: pd.Series,
    buckets: int,
    handle_outside: str = "ignore",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin numeric values using reference quantile edges and return discrete probs.
    NOTE: To stay consistent with your KL implementation, this copies the logic/shape.
    """
    # Build bin edges from reference quantiles (stable across time)
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(ref.quantile(quantiles).values)

    # Degenerate case: no variability in reference → single bin with mass 1
    if edges.size <= 1:
        return np.array([1.0]), np.array([1.0])

    # Use pandas cut (right-closed; include_lowest=True), matching typical histogram semantics
    if handle_outside == "clip":
        ref_b = pd.cut(
            ref.clip(edges[0], edges[-1]),
            bins=edges,
            include_lowest=True,
            right=True,
        )
        cur_b = pd.cut(
            cur.clip(edges[0], edges[-1]),
            bins=edges,
            include_lowest=True,
            right=True,
        )
    elif handle_outside == "extend":
        # Add open-ended tails so that outside values are captured instead of dropped
        ext_edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        ref_b = pd.cut(ref, bins=ext_edges, include_lowest=True, right=True)
        cur_b = pd.cut(cur, bins=ext_edges, include_lowest=True, right=True)
    elif handle_outside == "ignore":
        # Values outside edges become NaN and are ignored by value_counts()
        ref_b = pd.cut(ref, bins=edges, include_lowest=True, right=True)
        cur_b = pd.cut(cur, bins=edges, include_lowest=True, right=True)
    else:
        raise ValueError("handle_outside must be one of {'ignore','clip','extend'}")

    ref_counts = ref_b.value_counts(sort=False).values.astype(float)
    cur_counts = cur_b.value_counts(sort=False).values.astype(float)

    ref_probs = (
        ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else np.zeros_like(ref_counts)
    )
    cur_probs = (
        cur_counts / cur_counts.sum() if cur_counts.sum() > 0 else np.zeros_like(cur_counts)
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
        ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else np.zeros_like(ref_counts)
    )
    cur_probs = (
        cur_counts / cur_counts.sum() if cur_counts.sum() > 0 else np.zeros_like(cur_counts)
    )
    return ref_probs, cur_probs


def compute_js(
    ref: Iterable,
    cur: Iterable,
    *,
    buckets: int = 10,
    base: Literal["e", "2", "10"] = "e",
    epsilon: float = 1e-12,
    handle_outside: str = "ignore",
) -> float:
    """
    Compute Jensen–Shannon divergence JS(P, Q) between reference P and current Q.

    JS is defined as:
        M = 0.5*(P + Q)
        JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    Parameters
    ----------
    ref : Iterable
        Reference sample. I.e. training data.
    cur : Iterable
        Current sample. I.e. production data.
    buckets : int, default 10
        Number of quantile bins (for numeric data). Ignored for categorical data.
    base : {"e", "2", "10"}, default "e"
        Logarithm base. "e" for nats, "2" for bits, "10" for bans.
    epsilon : float, default 1e-12
        Smoothing constant to handle zero probabilities.
    handle_outside : {"ignore","clip","extend"}, default "ignore"
        Strategy for handling numeric values outside reference-derived bin edges.

    Returns
    -------
    float
        JS(P, Q) >= 0. Bounded and symmetric.
        With base="2", JS is in [0, 1] for discrete distributions.

    Notes
    -----
    - NaN values are dropped.
    - For numeric data: ref-quantile binning is used for stability across time.
    - For categorical data: frequency-based discrete distributions are used.
    - Epsilon smoothing prevents log(0) and division-by-zero issues.
    - Unlike KL, JS is symmetric and always finite (given epsilon smoothing).
    """
    # Convert and drop NaNs early (aligns with your KL behavior)
    ref_s = pd.Series(np.asarray(ref)).dropna()
    cur_s = pd.Series(np.asarray(cur)).dropna()

    if ref_s.empty or cur_s.empty:
        warnings.warn(
            "One of the input series is empty after dropping NaN. Returning NaN."
        )
        return np.nan

    # Determine numeric vs categorical by dtype
    if pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(cur_s):
        P, Q = _probs_from_numeric(
            ref_s.astype(float),
            cur_s.astype(float),
            buckets=buckets,
            handle_outside=handle_outside,
        )
    else:
        P, Q = _probs_from_categorical(ref_s.astype(object), cur_s.astype(object))

    # Smoothing for numerical stability: replace zeros by epsilon
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    P = np.where(P <= 0, epsilon, P)
    Q = np.where(Q <= 0, epsilon, Q)

    # Normalize to ensure sum to 1 after smoothing
    P = P / P.sum()
    Q = Q / Q.sum()

    # Mixture distribution M
    M = 0.5 * (P + Q)
    M = np.where(M <= 0, epsilon, M)  # ultra-safe, though P/Q already positive
    M = M / M.sum()

    # Choose log base
    if base == "e":
        log = np.log
    elif base == "2":
        log = np.log2
    elif base == "10":
        log = np.log10
    else:
        raise ValueError("base must be one of {'e','2','10'}")

    # KL(P||M) and KL(Q||M)
    kl_pm = np.sum(P * (log(P) - log(M)))
    kl_qm = np.sum(Q * (log(Q) - log(M)))

    js = 0.5 * (kl_pm + kl_qm)
    return float(js)
