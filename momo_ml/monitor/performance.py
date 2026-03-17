from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from momo_ml.metrics.ks import compute_ks


class PerformanceEvaluator:
    """
    Evaluate model performance drift between reference and current datasets.

    Supports:
    - Binary & multiclass classification: AUC, Accuracy, Precision, Recall, F1, KS
    - Regression: RMSE, MAE, R2, SMAPE, P90/P95 Error, Huber Loss

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference dataset containing true labels and predictions.
    cur_df : pd.DataFrame
        Current dataset.
    label_col : str
        Column name of true labels. For binary classification, the labels can be
        any two distinct values; the larger value will be treated as the positive
        class, the smaller as the negative. However, to ensure correct calculation
        of precision/recall/F1 (which assume 0/1 labels), it is strongly recommended
        to convert your labels to 0 (negative) and 1 (positive) before passing data.
    pred_col : str
        Column name of model predictions.
    task_type : Optional[str]
        "classification" or "regression" or None (auto-detect)
    huber_delta : float
        Threshold for huber loss. Default 1.0
    smape_as_percentage : bool
        If True, returns SMAPE as percentage in [0, 200]. If False, returns ratio in [0, 2]. Default True.
    smape_epsilon : float
        Small value to avoid division by zero in SMAPE. Default 1e-8.
    """

    def __init__(
        self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        label_col: Optional[str],
        pred_col: Optional[str],
        task_type: Optional[str] = None,
        huber_delta: float = 1.0,
        smape_as_percentage: bool = True,
        smape_epsilon: float = 1e-8,
    ):
        self.ref_df = ref_df.copy()
        self.cur_df = cur_df.copy()
        self.label_col = label_col
        self.pred_col = pred_col
        self.task_type = (
            task_type  # "classification" or "regression" or None (auto-detect)
        )
        self.huber_delta = float(huber_delta)
        self.smape_as_percentage = bool(smape_as_percentage)
        self.smape_epsilon = float(smape_epsilon)

    # -------------------------------------------------------
    # Utility
    # -------------------------------------------------------
    def _get_classification_type(self) -> str:
        """Return 'binary' or 'multiclass' based on label column."""
        y = self.ref_df[self.label_col].dropna()
        n_unique = y.nunique()
        if n_unique == 2:
            return "binary"
        elif n_unique > 2:
            return "multiclass"
        else:
            return "unknown"  # can't train if only 1 class present

    def _is_classification(self) -> bool:
        """Detect classification vs regression based on label dtype."""
        y = self.ref_df[self.label_col].dropna()
        return y.nunique() <= 20

    # -------------------------------------------------------
    # Helpers for regression metrics
    # -------------------------------------------------------
    @staticmethod
    def _smape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-8,
        as_percentage: bool = True,
    ) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        Range: [0, 2] if ratio; [0, 200] if percentage.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denom = np.abs(y_true) + np.abs(y_pred) + epsilon
        smape = np.mean(2.0 * np.abs(y_pred - y_true) / denom)
        if as_percentage:
            smape *= 100.0
        return smape

    @staticmethod
    def _huber_loss(
        y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0
    ) -> float:
        """
        Huber loss with parameter delta.
        For residual r:
          if |r| <= delta: 0.5 * r^2
          else           : delta * (|r| - 0.5 * delta)
        Returns mean loss over samples.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        r = y_true - y_pred
        abs_r = np.abs(r)
        quad = 0.5 * (r**2)
        lin = delta * (abs_r - 0.5 * delta)
        loss = np.where(abs_r <= delta, quad, lin)
        return float(np.mean(loss))

    # -------------------------------------------------------
    # Classification metrics
    # -------------------------------------------------------
    def _classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str
    ) -> Dict[str, float]:
        metrics = {}

        if task_type == "binary":
            # AUC (need probability scores)
            try:
                metrics["auc"] = roc_auc_score(y_true, y_pred)
            except Exception:
                metrics["auc"] = np.nan
            # ----- KS statistic -----
            unique_labels = np.unique(y_true)
            if not (
                np.array_equal(unique_labels, [0, 1])
                or np.array_equal(unique_labels, [1, 0])
            ):
                warnings.warn(
                    f"Binary classification detected but labels are {unique_labels}. "
                    "For accurate precision/recall/F1 calculation, it is recommended to "
                    "convert labels to 0 (negative) and 1 (positive). The library will "
                    f"treat {max(unique_labels)} as positive and {min(unique_labels)} as negative."
                )
            if len(unique_labels) == 2:
                # assume pos label > neg label
                pos_label = max(unique_labels)
                neg_scores = y_pred[y_true != pos_label]
                pos_scores = y_pred[y_true == pos_label]
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    # call internal KS calculation method
                    ks_result = compute_ks(neg_scores, pos_scores, return_pvalue=False)
                    metrics["ks"] = ks_result["statistic"]
                else:
                    metrics["ks"] = np.nan
            else:
                metrics["ks"] = (
                    np.nan
                )  # non-binary classification: KS is not meaningful
            # binary (threshold at 0.5)
            y_hat = (y_pred >= 0.5).astype(int)

            metrics.update(
                {
                    "accuracy": accuracy_score(y_true, y_hat),
                    "precision": precision_score(y_true, y_hat, zero_division=0),
                    "recall": recall_score(y_true, y_hat, zero_division=0),
                    "f1": f1_score(y_true, y_hat, zero_division=0),
                }
            )

        elif task_type == "multiclass":
            # assume y_pred are predicted classes
            y_hat = y_pred.astype(int)

            metrics.update(
                {
                    "accuracy": accuracy_score(y_true, y_hat),
                    "precision_macro": precision_score(
                        y_true, y_hat, average="macro", zero_division=0
                    ),
                    "recall_macro": recall_score(
                        y_true, y_hat, average="macro", zero_division=0
                    ),
                    "f1_macro": f1_score(
                        y_true, y_hat, average="macro", zero_division=0
                    ),
                    # Optional: add micro or weighted averages
                    "precision_weighted": precision_score(
                        y_true, y_hat, average="weighted", zero_division=0
                    ),
                    "recall_weighted": recall_score(
                        y_true, y_hat, average="weighted", zero_division=0
                    ),
                    "f1_weighted": f1_score(
                        y_true, y_hat, average="weighted", zero_division=0
                    ),
                }
            )

        return metrics

    # -------------------------------------------------------
    # Regression metrics
    # -------------------------------------------------------
    def _regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        r2 = r2_score(y_true, y_pred)
        smape = self._smape(
            y_true,
            y_pred,
            epsilon=self.smape_epsilon,
            as_percentage=self.smape_as_percentage,
        )
        abs_err = np.abs(y_true - y_pred)
        p90_err = float(np.quantile(abs_err, 0.90))
        p95_err = float(np.quantile(abs_err, 0.95))
        huber = self._huber_loss(y_true, y_pred, delta=self.huber_delta)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "smape": smape,
            "p90_error": p90_err,
            "p95_error": p95_err,
            "huber_loss": huber,
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
        Dict[str, Any]  # on error: {"error": "..."}
        """
        # ---- param check ----
        if self.label_col is None or self.pred_col is None:
            return {"error": "label_col and pred_col must be provided."}

        missing = []
        # label needs to exist in both sides
        if self.label_col not in self.ref_df.columns:
            missing.append(f"ref_df.{self.label_col}")
        if self.label_col not in self.cur_df.columns:
            missing.append(f"cur_df.{self.label_col}")
        # pred needs to exist in both sides
        if self.pred_col not in self.ref_df.columns:
            missing.append(f"ref_df.{self.pred_col}")
        if self.pred_col not in self.cur_df.columns:
            missing.append(f"cur_df.{self.pred_col}")

        if missing:
            return {"error": f"Missing required columns: {', '.join(missing)}"}

        # ---- get values and drop NaN ----
        y_ref_s = self.ref_df[self.label_col].dropna()
        p_ref_s = self.ref_df[self.pred_col].dropna()
        y_cur_s = self.cur_df[self.label_col].dropna()
        p_cur_s = self.cur_df[self.pred_col].dropna()

        # if any of the series is empty after dropping NaN, we cannot compute metrics
        if y_ref_s.empty or p_ref_s.empty or y_cur_s.empty or p_cur_s.empty:
            return {"error": "No valid (non-NaN) values in label/pred columns."}

        y_ref = y_ref_s.values
        p_ref = p_ref_s.values
        y_cur = y_cur_s.values
        p_cur = p_cur_s.values

        # ---- validate job type ----
        is_classif = self._is_classification()

        # ---- determine task type ----
        if self.task_type is not None:
            if self.task_type not in ["classification", "regression"]:
                return {
                    "error": f"task_type must be 'classification' or 'regression', got {self.task_type}"
                }
            is_classif = self.task_type == "classification"
        else:
            is_classif = self._is_classification()

        # ---- calculate metrics ----
        if is_classif:
            classif_type = self._get_classification_type()
            ref_metrics = self._classification_metrics(y_ref, p_ref, classif_type)
            cur_metrics = self._classification_metrics(y_cur, p_cur, classif_type)
        else:
            # reg: y_pred and p_pred are both continuous
            ref_metrics = self._regression_metrics(y_ref, p_ref)
            cur_metrics = self._regression_metrics(y_cur, p_cur)

        # ---- calculate delta（current - reference）----
        keys = set(ref_metrics.keys()) | set(cur_metrics.keys())
        delta = {
            k: (cur_metrics.get(k, np.nan) - ref_metrics.get(k, np.nan)) for k in keys
        }

        return {
            "task_type": "classification" if is_classif else "regression",
            "classification_subtype": classif_type if is_classif else None,
            "reference": ref_metrics,
            "current": cur_metrics,
            "delta": delta,
        }
