from typing import Iterable, Tuple
import warnings
import numpy as np
import pandas as pd

try:
    # SciPy is optional; if available we use its accurate 1D Wasserstein distance
    from scipy.stats import wasserstein_distance as _sp_wasserstein_distance
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _probs_from_categorical(ref: pd.Series, cur: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
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

    ref_probs = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else np.zeros_like(ref_counts)
    cur_probs = cur_counts / cur_counts.sum() if cur_counts.sum() > 0 else np.zeros_like(cur_counts)
    return ref_probs, cur_probs


def _wasserstein_1d_empirical_no_scipy(ref: np.ndarray, cur: np.ndarray) -> float:
    """
    Compute the 1D Wasserstein (Earth Mover's) distance between two empirical
    distributions given raw samples, WITHOUT SciPy (O((n+m) log(n+m))).

    Uses the identity:
        W1(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
    For empirical CDFs, integrate piecewise between sorted breakpoints.
    """
    x = np.sort(ref)
    y = np.sort(cur)
    n = x.size
    m = y.size

    # Merge unique breakpoints
    t = np.sort(np.unique(np.concatenate([x, y])))
    if t.size <= 1:
        return 0.0

    # CDF values are constant on [t[i], t[i+1])
    # Use counts <= t[i] (right side) as ECDF at the left endpoint of each interval
    idx_x = np.searchsorted(x, t, side="right")
    idx_y = np.searchsorted(y, t, side="right")
    F_x = idx_x / n
    F_y = idx_y / m

    # Interval lengths
    dt = np.diff(t)
    # On interval [t[i], t[i+1]), ECDF difference equals |F_x[i] - F_y[i]|
    diff = np.abs(F_x[:-1] - F_y[:-1])

    return float(np.sum(diff * dt))


def compute_wd(
    ref: Iterable,
    cur: Iterable,
) -> float:
    """
    Compute the 1D Wasserstein Distance (Earth Mover's Distance, p=1) between
    reference (P) and current (Q) distributions.

    Behavior
    --------
    - Numeric features:
        Uses *empirical* 1D Wasserstein distance directly from raw samples.
        If SciPy is available, uses `scipy.stats.wasserstein_distance`.
        Otherwise uses a dependency-free fallback integrating |F_P - F_Q|.
        NOTE: `buckets` and `handle_outside` are **ignored** for numeric data,
              since empirical computation is preferred and more accurate.

    - Categorical features:
        Uses the Hamming metric on categories (distance=1 if different, 0 if same),
        for which Wasserstein equals the total variation distance:
            WD_cat(P, Q) = 0.5 * ||P - Q||_1
        This is stable, symmetric, and always finite.

    Parameters
    ----------
    ref : Iterable
        Reference sample (training/baseline).
    cur : Iterable
        Current sample (production).
    buckets : int, default 10
        Kept for API style consistency with KL/JS; ignored for numeric WD.
    handle_outside : {"ignore","clip","extend"}, default "ignore"
        Kept for API style consistency; ignored for WD.

    Returns
    -------
    float
        Wasserstein distance >= 0. Scale-sensitive for numeric features (same units as the data).
        Bounded in [0, 1] for categorical case with Hamming metric.

    Notes
    -----
    - NaN values are dropped.
    - Numeric WD is computed from raw samples (no binning).
    - Categorical WD uses 0.5 * L1 distance between aligned probability vectors.
    - If either side becomes empty after dropping NaN, returns NaN and warns.
    """
    # Convert and drop NaNs early
    ref_s = pd.Series(np.asarray(ref)).dropna()
    cur_s = pd.Series(np.asarray(cur)).dropna()

    if ref_s.empty or cur_s.empty:
        warnings.warn("One of the input series is empty after dropping NaN. Returning NaN.")
        return np.nan

    # Numeric vs categorical
    is_ref_num = pd.api.types.is_numeric_dtype(ref_s)
    is_cur_num = pd.api.types.is_numeric_dtype(cur_s)

    if is_ref_num and is_cur_num:
        # Numeric: empirical Wasserstein on raw samples
        ref_vals = ref_s.astype(float).values
        cur_vals = cur_s.astype(float).values

        if _HAVE_SCIPY:
            return float(_sp_wasserstein_distance(ref_vals, cur_vals))
        else:
            return _wasserstein_1d_empirical_no_scipy(ref_vals, cur_vals)

    # Categorical: WD under Hamming distance is 0.5 * L1(P - Q)
    P, Q = _probs_from_categorical(ref_s.astype(object), cur_s.astype(object))
    wd_cat = 0.5 * float(np.sum(np.abs(P - Q)))
    return wd_cat
