"""
Low-level performance metric functions for model monitoring.

This module provides:
- Binary classification metrics (AUC, accuracy, precision, recall, F1)
- Regression metrics (RMSE, MAE)
- Automatic task detection (classification vs regression)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)


# ======================================================
# Utility: Detect task type
# ======================================================


def infer_task_type_from_labels(y: pd.Series, max_class_cardinality: int = 20) -> str:
    """
    Infer whether the task is classification or regression based on label characteristics.

    Rules:
    - If dtype is integer or number of unique values <= threshold → classification
    - Else → regression
    """
    y_clean = y.dropna()

    if y_clean.empty:
        return "regression"

    if pd.api.types.is_integer_dtype(y_clean):
        return "classification"

    if y_clean.nunique() <= max_class_cardinality:
        return "classification"

    return "regression"


# ======================================================
# Classification Metrics
# ======================================================


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0/1).
    y_pred_prob : array-like
        Predicted probabilities (0-1).
    threshold : float
        Threshold to binarize predictions.

    Returns
    -------
    Dict[str, float]
    """
    metrics = {}

    # AUC can fail if labels have only one class
    try:
        metrics["auc"] = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        metrics["auc"] = np.nan

    y_pred_cls = (y_pred_prob >= threshold).astype(int)

    metrics.update(
        {
            "accuracy": accuracy_score(y_true, y_pred_cls),
            "precision": precision_score(y_true, y_pred_cls, zero_division=0),
            "recall": recall_score(y_true, y_pred_cls, zero_division=0),
            "f1": f1_score(y_true, y_pred_cls, zero_division=0),
        }
    )

    return metrics


# ======================================================
# Regression Metrics
# ======================================================


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics (RMSE, MAE).

    Parameters
    ----------
    y_true : array-like
        Ground truth numeric labels.
    y_pred : array-like
        Predicted numeric output.

    Returns
    -------
    Dict[str, float]
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
    }


# ======================================================
# Public helper
# ======================================================


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    """
    Wrapper used by PerformanceEvaluator:
    Compute metrics based on a required task type ('classification' or 'regression').

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    task_type : str
        'classification' or 'regression'

    Returns
    -------
    Dict[str, Any]
    """
    if task_type == "classification":
        return compute_classification_metrics(y_true, y_pred)

    elif task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
