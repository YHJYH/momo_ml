from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# def _ecdf(x: np.ndarray) -> np.ndarray:
#     """Compute empirical CDF values for a sorted array."""
#     n = x.size
#     return np.arange(1, n + 1) / n


def compute_ks(
    ref: Iterable, cur: Iterable, *, return_pvalue: bool = True
) -> Dict[str, Any]:
    """
    Compute the two-sample Kolmogorov–Smirnov (KS) statistic between
    reference distribution and current distribution.

    Parameters
    ----------
    ref : Iterable
        Reference (baseline) sample (numeric).
    cur : Iterable
        Current sample to compare (numeric).
    return_pvalue : bool, default True
        If True, return p-value (requires scipy). If scipy is not
        available, only the statistic is returned.

    Returns
    -------
    Dict[str, Any]
        {
          "statistic": float,   # KS D statistic in [0, 1]
          "pvalue": float|None, # p-value if available and required
          "n_ref": int,         # effective sample size after dropping NaN
          "n_cur": int
        }

    Notes
    -----
    - Larger statistic indicates larger distribution shift.
    - KS is sensitive to the entire distribution (location/shape).
    - Input NaN values are dropped.
    """
    ref_s = pd.Series(ref).dropna().astype(float)
    cur_s = pd.Series(cur).dropna().astype(float)

    n_ref = ref_s.size
    n_cur = cur_s.size

    if n_ref == 0 or n_cur == 0:
        return {"statistic": np.nan, "pvalue": None, "n_ref": n_ref, "n_cur": n_cur}

    if _HAVE_SCIPY and return_pvalue:
        stat, pval = ks_2samp(
            ref_s.values, cur_s.values, alternative="two-sided", mode="auto"
        )
        return {
            "statistic": float(stat),
            "pvalue": float(pval),
            "n_ref": n_ref,
            "n_cur": n_cur,
        }

    # Fallback: compute statistic only (no p-value)
    ref_sorted = np.sort(ref_s.values)
    cur_sorted = np.sort(cur_s.values)

    # Merge unique values as evaluation points
    all_vals = np.sort(np.unique(np.concatenate([ref_sorted, cur_sorted])))

    # Compute ECDF values at all_vals for ref and cur
    # For ECDF at x: proportion of samples <= x
    ref_idx = np.searchsorted(ref_sorted, all_vals, side="right")
    cur_idx = np.searchsorted(cur_sorted, all_vals, side="right")
    ref_ecdf = ref_idx / n_ref
    cur_ecdf = cur_idx / n_cur

    stat = np.max(np.abs(ref_ecdf - cur_ecdf))
    return {"statistic": float(stat), "pvalue": None, "n_ref": n_ref, "n_cur": n_cur}
