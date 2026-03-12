
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)


class PerformanceEvaluator:
    """
    Evaluate model performance drift between reference and current datasets.

    Supports:
    - Binary classification: AUC, Accuracy, Precision, Recall, F1
    - Regression: RMSE, MAE

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference dataset containing true labels and predictions.
    cur_df : pd.DataFrame
        Current dataset.
    label_col : str
        Column name of true labels.
    pred_col : str
        Column name of model predictions.
    """

    def __init__(
        self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        label_col: Optional[str],
        pred_col: Optional[str],
    ):
        self.ref_df = ref_df.copy()
        self.cur_df = cur_df.copy()
        self.label_col = label_col
        self.pred_col = pred_col

    # -------------------------------------------------------
    # Utility
    # -------------------------------------------------------
    def _is_classification(self) -> bool:
        """Detect classification vs regression based on label dtype."""
        y = self.ref_df[self.label_col]
        return pd.api.types.is_integer_dtype(y) or y.nunique() <= 20

    # -------------------------------------------------------
    # Classification metrics
    # -------------------------------------------------------
    def _classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:

        # Handle edge cases: AUC requires both positive and negative labels
        metrics = {}

        try:
            metrics["auc"] = roc_auc_score(y_true, y_pred)
        except Exception:
            metrics["auc"] = np.nan

        # Convert pred→binary label using 0.5 threshold
        y_hat = (y_pred >= 0.5).astype(int)

        metrics.update(
            {
                "accuracy": accuracy_score(y_true, y_hat),
                "precision": precision_score(y_true, y_hat, zero_division=0),
                "recall": recall_score(y_true, y_hat, zero_division=0),
                "f1": f1_score(y_true, y_hat, zero_division=0),
            }
        )
        return metrics

    # -------------------------------------------------------
    # Regression metrics
    # -------------------------------------------------------
    def _regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)

        return {
            "rmse": rmse,
            "mae": mae,
        }

    # -------------------------------------------------------
    # Public method
    # -------------------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        """
        Compute performance metrics for both ref_df and cur_df,
        and return a comparison.

        Returns
        -------
        Dict[str, Any]
        """
        if self.label_col is None or self.pred_col is None:
            return {"error": "label_col and pred_col must be provided."}

        # Extract arrays
        y_ref = self.ref_df[self.label_col].dropna().values
        p_ref = self.ref_df[self.pred_col].dropna().values

        y_cur = self.cur_df[self.label_col].dropna().values
        p_cur = self.cur_df[self.pred_col].dropna().values

        # Determine task type
        is_classif = self._is_classification()

        if is_classif:
            ref_metrics = self._classification_metrics(y_ref, p_ref)
            cur_metrics = self._classification_metrics(y_cur, p_cur)
        else:
            ref_metrics = self._regression_metrics(y_ref, p_ref)
            cur_metrics = self._regression_metrics(y_cur, p_cur)

        # Combine output
        return {
            "task_type": "classification" if is_classif else "regression",
            "reference": ref_metrics,
            "current": cur_metrics,
            "delta": {
                k: cur_metrics.get(k, np.nan) - ref_metrics.get(k, np.nan)
                for k in ref_metrics.keys()
            },
        }
