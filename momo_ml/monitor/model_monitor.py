from typing import Optional, Dict, Any
import pandas as pd
import warnings

from momo_ml.monitor.data_drift import DataDriftDetector
from momo_ml.monitor.performance import PerformanceEvaluator
from momo_ml.monitor.prediction_drift import PredictionDriftDetector


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
    data_drift_kwargs : dict, optional
        Keyword arguments passed to DataDriftDetector constructor.
    performance_kwargs : dict, optional
        Keyword arguments passed to PerformanceEvaluator constructor.
    prediction_drift_kwargs : dict, optional
        Keyword arguments passed to PredictionDriftDetector constructor.

    Notes
    -----
    The detectors are instantiated lazily – only when the corresponding
    `run_*` method is called for the first time. This avoids unnecessary
    memory copies and computation if only a subset of analyses is needed.
    """

    def __init__(
        self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        label_col: Optional[str] = None,
        pred_col: Optional[str] = None,
        data_drift_kwargs: Optional[Dict[str, Any]] = None,
        performance_kwargs: Optional[Dict[str, Any]] = None,
        prediction_drift_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.ref_df = ref_df
        self.cur_df = cur_df
        self.label_col = label_col
        self.pred_col = pred_col

        # Store kwargs for each detector
        self.data_drift_kwargs = data_drift_kwargs or {}
        self.performance_kwargs = performance_kwargs or {}
        self.prediction_drift_kwargs = prediction_drift_kwargs or {}

        # Internal placeholders for lazy instantiation
        self._data_drift = None
        self._performance = None
        self._prediction_drift = None

    @property
    def data_drift(self) -> DataDriftDetector:
        """Lazy initializer for DataDriftDetector."""
        if self._data_drift is None:
            self._data_drift = DataDriftDetector(
                self.ref_df, self.cur_df, **self.data_drift_kwargs
            )
        return self._data_drift

    @property
    def performance(self) -> PerformanceEvaluator:
        """Lazy initializer for PerformanceEvaluator."""
        if self._performance is None:
            self._performance = PerformanceEvaluator(
                self.ref_df,
                self.cur_df,
                self.label_col,
                self.pred_col,
                **self.performance_kwargs,
            )
        return self._performance

    @property
    def prediction_drift(self) -> PredictionDriftDetector:
        """Lazy initializer for PredictionDriftDetector."""
        if self._prediction_drift is None:
            self._prediction_drift = PredictionDriftDetector(
                self.ref_df, self.cur_df, self.pred_col, **self.prediction_drift_kwargs
            )
        return self._prediction_drift

    def run_performance_drift(self) -> Dict[str, Any]:
        """
        Compute performance metrics comparing reference vs. current.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance metrics or an error message.
        """
        try:
            return self.performance.evaluate()
        except Exception as e:
            warnings.warn(f"Performance drift evaluation failed: {e}", RuntimeWarning)
            return {"error": f"Performance drift evaluation failed: {str(e)}"}

    def run_data_drift(self) -> Dict[str, Any]:
        """
        Compute feature-level data drift measures.

        Returns
        -------
        Dict[str, Any]
            Dictionary with drift results for numeric and categorical features,
            or an error message.
        """
        try:
            return self.data_drift.compute()
        except Exception as e:
            warnings.warn(f"Data drift computation failed: {e}", RuntimeWarning)
            return {"error": f"Data drift computation failed: {str(e)}"}

    def run_prediction_drift(self) -> Dict[str, Any]:
        """
        Compute distribution shift in model predictions.

        Returns
        -------
        Dict[str, Any]
            Dictionary with prediction drift statistics or an error message.
        """
        try:
            return self.prediction_drift.compute()
        except Exception as e:
            warnings.warn(f"Prediction drift computation failed: {e}", RuntimeWarning)
            return {"error": f"Prediction drift computation failed: {str(e)}"}

    def run_all(self) -> Dict[str, Any]:
        """
        Execute the full monitoring pipeline.

        Returns
        -------
        Dict[str, Any]
            A dictionary with keys 'performance_drift', 'data_drift',
            and 'prediction_drift', each containing the corresponding result
            or an error message.
        """
        return {
            "performance_drift": self.run_performance_drift(),
            "data_drift": self.run_data_drift(),
            "prediction_drift": self.run_prediction_drift(),
        }
