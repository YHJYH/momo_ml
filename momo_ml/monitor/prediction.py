
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class PredictionDriftDetector:
    """
    Detect drift in model prediction outputs.

    Supports:
    - Distribution shift comparison (histogram-based)
    - Decile shift analysis
    - Summary statistics comparison

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference (baseline) dataset.
    cur_df : pd.DataFrame
        Current dataset for comparison.
    pred_col : str
        Column name of model predictions.
    """

    def __init__(self, ref_df: pd.DataFrame, cur_df: pd.DataFrame, pred_col: Optional[str]):
        self.ref_df = ref_df.copy()
        self.cur_df = cur_df.copy()
        self.pred_col = pred_col

    # -------------------------------------------------------
    # Utility extractors
    # -------------------------------------------------------
    def _get_predictions(self):
        """Extract non-null prediction arrays."""
        if self.pred_col is None:
            return np.array([]), np.array([])

        ref = self.ref_df[self.pred_col].dropna().astype(float).values
        cur = self.cur_df[self.pred_col].dropna().astype(float).values
        return ref, cur

    # -------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------
    def _summary_stats(self, ref: np.ndarray, cur: np.ndarray) -> Dict[str, Any]:
        return {
            "mean": {"reference": float(np.mean(ref)), "current": float(np.mean(cur)),
                     "delta": float(np.mean(cur) - np.mean(ref))},
            "std": {"reference": float(np.std(ref)), "current": float(np.std(cur)),
                    "delta": float(np.std(cur) - np.std(ref))},
            "min": {"reference": float(np.min(ref)), "current": float(np.min(cur))},
            "max": {"reference": float(np.max(ref)), "current": float(np.max(cur))},
        }

    # -------------------------------------------------------
    # Distribution shift (histogram difference)
    # -------------------------------------------------------
    def _distribution_shift(self, ref: np.ndarray, cur: np.ndarray, bins: int = 20) -> Dict[str, Any]:
        """
        Compare histograms of predictions. Computes L1 and L2 distances.
        """
        # Compute aligned histogram bins
        combined = np.concatenate([ref, cur])
        hist_bins = np.linspace(combined.min(), combined.max(), bins + 1)

        ref_hist, _ = np.histogram(ref, bins=hist_bins, density=True)
        cur_hist, _ = np.histogram(cur, bins=hist_bins, density=True)

        # Replace NaN densities with 0
        ref_hist = np.nan_to_num(ref_hist)
        cur_hist = np.nan_to_num(cur_hist)

        l1 = float(np.sum(np.abs(ref_hist - cur_hist)))
        l2 = float(np.sqrt(np.sum((ref_hist - cur_hist) ** 2)))

        return {
            "l1_distance": l1,
            "l2_distance": l2,
            "bins": bins,
        }

    # -------------------------------------------------------
    # Decile shift
    # -------------------------------------------------------
    def _decile_shift(self, ref: np.ndarray, cur: np.ndarray) -> Dict[str, Any]:
        """
        Compare decile (10-quantile) bucket means between ref and cur.
        Useful for stability detection in ranking models.
        """
        quantiles = np.linspace(0, 1, 11)

        ref_q = np.quantile(ref, quantiles)
        cur_q = np.quantile(cur, quantiles)

        return {
            "ref_deciles": ref_q.tolist(),
            "cur_deciles": cur_q.tolist(),
            "delta": (cur_q - ref_q).tolist(),
        }

    # -------------------------------------------------------
    # Public method
    # -------------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        """Run all prediction drift analyses and return a structured report."""
        if self.pred_col is None:
            return {"error": "pred_col must be provided for prediction drift analysis."}

        ref, cur = self._get_predictions()

        if ref.size == 0 or cur.size == 0:
            return {"error": "No valid prediction values found."}

        return {
            "summary_statistics": self._summary_stats(ref, cur),
            "distribution_shift": self._distribution_shift(ref, cur),
            "decile_shift": self._decile_shift(ref, cur),
        }
