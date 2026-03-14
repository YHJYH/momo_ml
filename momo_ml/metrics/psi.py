import warnings
import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def compute_psi(ref, cur, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between reference and current distributions.

    Rules:
      - If both ref and cur are numeric, use the "numeric PSI" branch (binning based on ref quantiles).
      - Otherwise, always use the "categorical PSI" branch (based on category frequencies).

    This avoids errors when either side contains strings/object/string-dtype and we try to astype(float).
    """
    ref = np.asarray(ref)
    cur = np.asarray(cur)
    if ref.size == 0 or cur.size == 0:
        warnings.warn(
            "One of the input series is empty after dropping NaN. Returning NaN."
        )
        return np.nan

    ref_s = pd.Series(ref).dropna()
    cur_s = pd.Series(cur).dropna()

    # Only proceed with numeric branch if both sides have numeric dtype
    both_numeric = ptypes.is_numeric_dtype(ref_s) and ptypes.is_numeric_dtype(cur_s)

    # ============================ Categorical PSI (default branch) ============================
    if not both_numeric:
        # Convert to object for frequency counts; works with object/string-dtype/category/mixed types
        ref_o = ref_s.astype("object")
        cur_o = cur_s.astype("object")

        cats = pd.Index(ref_o.unique()).union(pd.Index(cur_o.unique()))

        ref_counts = ref_o.value_counts().reindex(cats, fill_value=0).astype(float)
        cur_counts = cur_o.value_counts().reindex(cats, fill_value=0).astype(float)

        # Normalize (if empty, fall back to uniform distribution to avoid 0/0)
        ref_sum = ref_counts.sum()
        cur_sum = cur_counts.sum()
        ref_dist = (
            (ref_counts / ref_sum) if ref_sum > 0 else np.ones(len(cats)) / len(cats)
        )
        cur_dist = (
            (cur_counts / cur_sum) if cur_sum > 0 else np.ones(len(cats)) / len(cats)
        )

        # Smooth to avoid log(0)
        eps = 1e-8
        ref_dist = np.where(ref_dist == 0, eps, ref_dist)
        cur_dist = np.where(cur_dist == 0, eps, cur_dist)

        return float(np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist)))

    # ============================ Numeric PSI ============================
    # We reach here only when both sides are indeed numeric
    ref_f = ref_s.astype(float)
    cur_f = cur_s.astype(float)

    # Use quantiles of reference sample for binning to ensure comparability over time
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(ref_f.quantile(quantiles).values)

    # If reference sample has no variance or binning fails, define PSI=0
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_f, bins=breakpoints)
    cur_counts, _ = np.histogram(cur_f, bins=breakpoints)

    ref_dist = ref_counts / max(len(ref_f), 1)
    cur_dist = cur_counts / max(len(cur_f), 1)

    eps = 1e-8
    ref_dist = np.where(ref_dist == 0, eps, ref_dist)
    cur_dist = np.where(cur_dist == 0, eps, cur_dist)

    return float(np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist)))
