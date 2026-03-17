from typing import Dict, Any, Optional, List, Union, Literal
import numpy as np
import pandas as pd
import warnings

from momo_ml.metrics.psi import compute_psi
from momo_ml.metrics.ks import compute_ks
from momo_ml.metrics.kl import compute_kl
from momo_ml.metrics.js import compute_js


class PredictionDriftDetector:
    """
    Detect drift in model prediction outputs.

    Supports:
    - Summary statistics: mean, std, quantiles (continuous) or category proportions (categorical)
    - Distribution shift metrics: PSI, KL, JS (both continuous & categorical), KS (continuous only)
    - Histogram L1/L2 distance (continuous only)
    - Decile shift (quantile-based, continuous only)

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference (baseline) dataset.
    cur_df : pd.DataFrame
        Current dataset for comparison.
    pred_col : str
        Column name of model predictions.
    bins : int, default 20
        Number of bins for histogram-based distances (L1/L2) and PSI/KL/JS binning.
    quantiles : List[float], optional
        Quantiles to use for decile shift analysis. Default is [0.0, 0.1, ..., 1.0] (11 points).
    include_psi : bool, default True
        Whether to compute Population Stability Index (PSI).
    include_ks : bool, default False
        Whether to compute Kolmogorov-Smirnov statistic (only for continuous predictions).
    include_kl : bool, default False
        Whether to compute Kullback-Leibler divergence.
    include_js : bool, default False
        Whether to compute Jensen-Shannon divergence.
    kl_base : Literal['e', '2', '10'], default 'e'
        Logarithm base for KL/JS.
    kl_epsilon : float, default 1e-12
        Small value to avoid log(0) in KL/JS.
    kl_handle_outside : str, default 'ignore'
        How to handle values outside reference bins for KL/JS ('ignore', 'clip', 'extend').
    categorical_threshold : int, default 20
        If the number of unique prediction values is <= this threshold, treat as categorical
        for summary statistics and decile shift exclusion (KS also excluded). PSI/KL/JS
        automatically handle categorical data via the metrics functions.
    """

    def __init__(
        self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        pred_col: Optional[str],
        bins: int = 20,
        quantiles: Optional[List[float]] = None,
        include_psi: bool = True,
        include_ks: bool = False,
        include_kl: bool = False,
        include_js: bool = False,
        kl_base: Literal["e", "2", "10"] = "e",
        kl_epsilon: float = 1e-12,
        kl_handle_outside: str = "ignore",
        categorical_threshold: int = 20,
    ):
        self.ref_df = ref_df.copy()
        self.cur_df = cur_df.copy()
        self.pred_col = pred_col
        self.bins = bins
        self.quantiles = quantiles if quantiles is not None else np.linspace(0, 1, 11).tolist()
        self.include_psi = include_psi
        self.include_ks = include_ks
        self.include_kl = include_kl
        self.include_js = include_js
        self.kl_base = kl_base
        self.kl_epsilon = kl_epsilon
        self.kl_handle_outside = kl_handle_outside
        self.categorical_threshold = categorical_threshold

    # -------------------------------------------------------
    # Utility extractors
    # -------------------------------------------------------
    def _get_predictions(self):
        """Extract non-null prediction arrays. Returns empty arrays on missing column."""
        if self.pred_col is None:
            return np.array([]), np.array([])

        if (
            self.pred_col not in self.ref_df.columns
            or self.pred_col not in self.cur_df.columns
        ):
            return np.array([]), np.array([])

        ref = self.ref_df[self.pred_col].dropna().to_numpy()
        cur = self.cur_df[self.pred_col].dropna().to_numpy()
        return ref, cur

    def _is_continuous(self, ref: np.ndarray, cur: np.ndarray) -> bool:
        """Determine whether predictions should be treated as continuous."""
        ref_s = pd.Series(ref)
        cur_s = pd.Series(cur)
        if not pd.api.types.is_numeric_dtype(ref_s) or not pd.api.types.is_numeric_dtype(cur_s):
            return False

        unique_ref = ref_s.nunique()
        unique_cur = cur_s.nunique()
        if unique_ref <= self.categorical_threshold and unique_cur <= self.categorical_threshold:
            return False

        return True

    # -------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------
    def _summary_stats_continuous(self, ref: np.ndarray, cur: np.ndarray) -> Dict[str, Any]:
        ref = ref.astype(float)
        cur = cur.astype(float)
        return {
            "mean": {
                "reference": float(np.mean(ref)),
                "current": float(np.mean(cur)),
                "delta": float(np.mean(cur) - np.mean(ref)),
            },
            "std": {
                "reference": float(np.std(ref)),
                "current": float(np.std(cur)),
                "delta": float(np.std(cur) - np.std(ref)),
            },
            "min": {"reference": float(np.min(ref)), "current": float(np.min(cur))},
            "max": {"reference": float(np.max(ref)), "current": float(np.max(cur))},
            "q25": {"reference": float(np.percentile(ref, 25)), "current": float(np.percentile(cur, 25))},
            "q50": {"reference": float(np.median(ref)), "current": float(np.median(cur))},
            "q75": {"reference": float(np.percentile(ref, 75)), "current": float(np.percentile(cur, 75))},
        }

    def _summary_stats_categorical(self, ref: np.ndarray, cur: np.ndarray) -> Dict[str, Any]:
        """For categorical predictions, return value counts and proportions."""
        ref_counts = pd.Series(ref).value_counts()
        cur_counts = pd.Series(cur).value_counts()
        # Convert index to string to avoid type mismatches
        ref_counts.index = ref_counts.index.astype(str)
        cur_counts.index = cur_counts.index.astype(str)
        all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))

        ref_props = {k: ref_counts.get(k, 0) / len(ref) for k in all_cats}
        cur_props = {k: cur_counts.get(k, 0) / len(cur) for k in all_cats}
        delta_props = {k: cur_props[k] - ref_props.get(k, 0) for k in all_cats}

        return {
            "categories": all_cats,
            "reference_proportions": ref_props,
            "current_proportions": cur_props,
            "delta_proportions": delta_props,
        }

    # -------------------------------------------------------
    # Distribution shift metrics
    # -------------------------------------------------------
    def _compute_histogram_distances(self, ref: np.ndarray, cur: np.ndarray) -> Dict[str, float]:
        """Compute L1 and L2 distances between histograms (only for continuous)."""
        if ref.size == 0 or cur.size == 0:
            return {"l1_distance": np.nan, "l2_distance": np.nan}

        combined = np.concatenate([ref, cur])
        min_val, max_val = combined.min(), combined.max()
        if min_val == max_val:
            # All values identical in both sets
            return {"l1_distance": 0.0, "l2_distance": 0.0}

        hist_bins = np.linspace(min_val, max_val, self.bins + 1)
        ref_hist, _ = np.histogram(ref, bins=hist_bins, density=True)
        cur_hist, _ = np.histogram(cur, bins=hist_bins, density=True)
        ref_hist = np.nan_to_num(ref_hist)
        cur_hist = np.nan_to_num(cur_hist)

        l1 = float(np.sum(np.abs(ref_hist - cur_hist)))
        l2 = float(np.sqrt(np.sum((ref_hist - cur_hist) ** 2)))
        return {"l1_distance": l1, "l2_distance": l2}

    def _distribution_shift(self, ref: np.ndarray, cur: np.ndarray, continuous: bool) -> Dict[str, Any]:
        """
        Compute all requested distribution shift metrics using metrics functions.
        Histogram distances are only computed for continuous.
        """
        results = {}

        # Histogram L1/L2 (continuous only)
        if np.issubdtype(ref.dtype, np.number) and np.issubdtype(cur.dtype, np.number):
            results.update(self._compute_histogram_distances(ref, cur))

        # PSI
        if self.include_psi:
            try:
                results["psi"] = compute_psi(ref, cur, buckets=self.bins)
            except Exception as e:
                warnings.warn(f"PSI computation failed: {e}")
                results["psi"] = np.nan

        # KS (continuous only)
        if self.include_ks and continuous:
            try:
                ks_res = compute_ks(ref, cur, return_pvalue=True)
                results["ks"] = {
                    "statistic": ks_res["statistic"],
                    "pvalue": ks_res["pvalue"],
                }
            except Exception as e:
                warnings.warn(f"KS computation failed: {e}")
                results["ks"] = {"statistic": np.nan, "pvalue": np.nan}

        # KL divergence
        if self.include_kl:
            try:
                results["kl"] = compute_kl(
                    ref, cur,
                    buckets=self.bins,
                    base=self.kl_base,
                    epsilon=self.kl_epsilon,
                    handle_outside=self.kl_handle_outside,
                )
            except Exception as e:
                warnings.warn(f"KL computation failed: {e}")
                results["kl"] = np.nan

        # JS divergence
        if self.include_js:
            try:
                results["js"] = compute_js(
                    ref, cur,
                    buckets=self.bins,
                    base=self.kl_base,
                    epsilon=self.kl_epsilon,
                    handle_outside=self.kl_handle_outside,
                )
            except Exception as e:
                warnings.warn(f"JS computation failed: {e}")
                results["js"] = np.nan

        return results

    # -------------------------------------------------------
    # Decile shift (continuous only)
    # -------------------------------------------------------
    def _decile_shift(self, ref: np.ndarray, cur: np.ndarray) -> Dict[str, Any]:
        """Compare quantiles between reference and current predictions."""
        ref_q = np.quantile(ref, self.quantiles)
        cur_q = np.quantile(cur, self.quantiles)

        return {
            "quantiles": self.quantiles,
            "ref_values": ref_q.tolist(),
            "cur_values": cur_q.tolist(),
            "delta": (cur_q - ref_q).tolist(),
        }

    # -------------------------------------------------------
    # Public method
    # -------------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        """Run all prediction drift analyses and return a structured report."""
        if self.pred_col is None:
            return {"error": "pred_col must be provided for prediction drift analysis."}

        if (
            self.pred_col not in self.ref_df.columns
            or self.pred_col not in self.cur_df.columns
        ):
            return {"error": f"Column '{self.pred_col}' not found in both datasets."}

        ref, cur = self._get_predictions()

        if ref.size == 0 or cur.size == 0:
            return {"error": "No valid prediction values found."}

        # Determine whether to treat as continuous (for summary/deciles/KS/histogram)
        continuous = self._is_continuous(ref, cur)

        result = {
            "prediction_type": "continuous" if continuous else "categorical",
        }

        # Summary statistics
        if continuous:
            result["summary_statistics"] = self._summary_stats_continuous(ref, cur)
        else:
            result["summary_statistics"] = self._summary_stats_categorical(ref, cur)

        # Distribution shift metrics (PSI, KL, JS always attempted; KS/histogram only if continuous)
        result["distribution_shift"] = self._distribution_shift(ref, cur, continuous)

        # Decile shift (continuous only)
        if continuous:
            result["decile_shift"] = self._decile_shift(ref, cur)

        return result