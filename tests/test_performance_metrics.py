# tests/test_performance_metrics.py

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from momo_ml.metrics.performance_metrics import (
    infer_task_type_from_labels,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_performance_metrics,
)


# ------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------

def test_classification_metrics_basic():
    # ground truth
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    # predicted probabilities
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4, 0.65, 0.3])

    m = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    # AUC should be in [0,1]
    assert 0.0 <= m["auc"] <= 1.0

    # Compute explicit expectations for accuracy at threshold 0.5
    y_hat = (y_prob >= 0.5).astype(int)
    acc = (y_hat == y_true).mean()
    assert_allclose(m["accuracy"], acc, rtol=1e-12, atol=1e-12)

    # Basic sanity checks: precision/recall/f1 in [0,1]
    for k in ["precision", "recall", "f1"]:
        assert 0.0 <= m[k] <= 1.0


def test_classification_metrics_single_class_auc_nan():
    # All positives -> AUC should fail and return NaN per our implementation
    y_true = np.ones(10, dtype=int)
    y_prob = np.linspace(0.2, 0.9, 10)

    m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert np.isnan(m["auc"])  # AUC undefined with single-class labels

    # Metrics that depend on threshold remain well-defined
    y_hat = (y_prob >= 0.5).astype(int)
    acc = (y_hat == y_true).mean()
    assert_allclose(m["accuracy"], acc, rtol=1e-12, atol=1e-12)
    # precision/recall/f1 should be finite due to zero_division=0
    for k in ["precision", "recall", "f1"]:
        assert np.isfinite(m[k])


def test_classification_threshold_effect():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.49, 0.6, 0.51, 0.2], dtype=float)

    m_05 = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    m_06 = compute_classification_metrics(y_true, y_prob, threshold=0.6)

    # Different thresholds should change discrete metrics
    assert m_05["accuracy"] != m_06["accuracy"] or \
           m_05["precision"] != m_06["precision"] or \
           m_05["recall"] != m_06["recall"] or \
           m_05["f1"] != m_06["f1"]

    # AUC should be threshold-independent
    assert_allclose(m_05["auc"], m_06["auc"], rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------
# Regression metrics
# ------------------------------------------------------------

def test_regression_metrics_basic():
    y_true = np.array([3.0, -1.0, 2.0, 7.0])
    y_pred = np.array([2.5, -0.5, 2.0, 8.0])

    m = compute_regression_metrics(y_true, y_pred)

    # Manual RMSE/MAE
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    assert_allclose(m["mae"], mae, rtol=1e-12, atol=1e-12)
    assert_allclose(m["rmse"], rmse, rtol=1e-12, atol=1e-12)


def test_regression_metrics_zero_error():
    y_true = np.array([1.5, 2.5, 3.5])
    y_pred = np.array([1.5, 2.5, 3.5])

    m = compute_regression_metrics(y_true, y_pred)
    assert_allclose(m["mae"], 0.0, rtol=1e-12, atol=1e-12)
    assert_allclose(m["rmse"], 0.0, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------
# Task type inference
# ------------------------------------------------------------

def test_infer_task_type_classification_integer_labels():
    y = pd.Series([0, 1, 1, 0, 1], dtype=int)
    t = infer_task_type_from_labels(y)
    assert t == "classification"


def test_infer_task_type_classification_low_cardinality():
    # even if float, few unique values should be classification
    y = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0], dtype=float)
    t = infer_task_type_from_labels(y, max_class_cardinality=5)
    assert t == "classification"


def test_infer_task_type_regression_high_cardinality():
    y = pd.Series(np.linspace(0, 1, 100))
    t = infer_task_type_from_labels(y, max_class_cardinality=10)
    assert t == "regression"


def test_infer_task_type_empty_defaults_to_regression():
    y = pd.Series([], dtype=float)
    t = infer_task_type_from_labels(y)
    assert t == "regression"


# ------------------------------------------------------------
# Unified wrapper routing
# ------------------------------------------------------------

def test_compute_performance_metrics_routing_classification():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.6, 0.2, 0.7])

    out = compute_performance_metrics(y_true, y_prob, task_type="classification")
    for k in ["auc", "accuracy", "precision", "recall", "f1"]:
        assert k in out


def test_compute_performance_metrics_routing_regression():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.9, 2.1, 2.9])

    out = compute_performance_metrics(y_true, y_pred, task_type="regression")
    for k in ["rmse", "mae"]:
        assert k in out


def test_compute_performance_metrics_invalid_task():
    with pytest.raises(ValueError):
        compute_performance_metrics(np.array([0, 1]), np.array([0.1, 0.9]), task_type="unknown")