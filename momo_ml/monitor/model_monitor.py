
from typing import Optional, Dict, Any
import pandas as pd

from momo_ml.monitor.data_drift import DataDriftDetector
from momo_ml.monitor.performance import PerformanceEvaluator
from momo_ml.monitor.prediction import PredictionDriftDetector


class ModelMonitor:
    """
    High-level orchestrator for model monitoring.
    Combines performance drift, data drift, and prediction drift detection.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference (baseline) dataset.
    cur_df : pd.DataFrame
        Current dataset to compare against.
    label_col : str, optional
        Column name of the ground truth label.
    pred_col : str, optional
        Column name of model predictions.
    """

    def __init__(
        self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        label_col: Optional[str] = None,
        pred_col: Optional[str] = None,
    ):
        self.ref_df = ref_df
        self.cur_df = cur_df
        self.label_col = label_col
        self.pred_col = pred_col

        self.performance = PerformanceEvaluator(
            ref_df, cur_df, label_col, pred_col
        )
        self.data_drift = DataDriftDetector(ref_df, cur_df)
        self.prediction_drift = PredictionDriftDetector(ref_df, cur_df, pred_col)

    def run_performance_drift(self) -> Dict[str, Any]:
        """Compute performance metrics comparing ref vs. cur."""
        if self.label_col is None or self.pred_col is None:
            return {"error": "Performance drift requires label_col and pred_col."}
        return self.performance.evaluate()

    def run_data_drift(self) -> Dict[str, Any]:
        """Compute feature-level data drift measures."""
        return self.data_drift.compute()

    def run_prediction_drift(self) -> Dict[str, Any]:
        """Compute distribution shift in model predictions."""
        if self.pred_col is None:
            return {"error": "Prediction drift requires pred_col."}
        return self.prediction_drift.compute()

    def run_all(self) -> Dict[str, Any]:
        """Execute the full monitoring pipeline."""
        result = {
            "performance_drift": self.run_performance_drift(),
            "data_drift": self.run_data_drift(),
            "prediction_drift": self.run_prediction_drift(),
        }
        return result
