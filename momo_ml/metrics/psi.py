
import numpy as np
import pandas as pd


def compute_psi(ref, cur, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI measures the shift in distribution between a reference dataset
    and a current dataset. Commonly used in credit risk and model monitoring.

    Parameters
    ----------
    ref : array-like
        Reference (baseline) data.
    cur : array-like
        Current data to compare.
    buckets : int
        Number of bins to split the data into.

    Returns
    -------
    float
        PSI value. Higher values indicate larger drift:
        - < 0.1 : No significant change
        - 0.1–0.25 : Moderate change
        - > 0.25 : Significant change (possible model drift)
    """
    ref = pd.Series(ref).dropna()
    cur = pd.Series(cur).dropna()

    # Determine bin boundaries based on reference distribution
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(ref.quantile(quantiles).values)

    # Assign bins
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    # Convert counts to % distribution
    ref_dist = ref_counts / len(ref)
    cur_dist = cur_counts / len(cur)

    # Replace zeros to avoid log division errors
    ref_dist = np.where(ref_dist == 0, 1e-8, ref_dist)
    cur_dist = np.where(cur_dist == 0, 1e-8, cur_dist)

    psi = np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist))
    return psi
