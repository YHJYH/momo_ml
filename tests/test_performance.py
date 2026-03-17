"""
Unit tests for the PerformanceEvaluator class.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    root_mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    r2_score
)

from momo_ml.metrics.ks import compute_ks
from momo_ml.monitor.performance import PerformanceEvaluator


# ---------------------------
#  Helpers for expected values (mirror implementation)
# ---------------------------
def _smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8, as_percentage: bool = True) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / denom)
    if as_percentage:
        smape *= 100.0
    return float(smape)


def _huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r = y_true - y_pred
    abs_r = np.abs(r)
    quad = 0.5 * (r ** 2)
    lin = delta * (abs_r - 0.5 * delta)
    loss = np.where(abs_r <= delta, quad, lin)
    return float(np.mean(loss))

# ---------------------------
#  Fixtures
# ---------------------------
@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 100
    ref_df = pd.DataFrame({
        "label": np.random.randn(n),
        "pred": np.random.randn(n) + 0.1
    })
    cur_df = pd.DataFrame({
        "label": np.random.randn(n),
        "pred": np.random.randn(n) - 0.1
    })
    return ref_df, cur_df


@pytest.fixture
def binary_data_01():
    np.random.seed(42)
    n = 100
    y_ref = np.array([0]*50 + [1]*50)
    p_ref = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.6, 1, 50)])
    ref_df = pd.DataFrame({"label": y_ref, "pred": p_ref})

    y_cur = np.array([0]*50 + [1]*50)
    p_cur = np.concatenate([np.random.uniform(0, 0.5, 50), np.random.uniform(0.5, 1, 50)])
    cur_df = pd.DataFrame({"label": y_cur, "pred": p_cur})
    return ref_df, cur_df


@pytest.fixture
def binary_data_25():
    np.random.seed(42)
    y_ref = np.array([2]*50 + [5]*50)
    p_ref = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.6, 1, 50)])
    ref_df = pd.DataFrame({"label": y_ref, "pred": p_ref})

    y_cur = np.array([2]*50 + [5]*50)
    p_cur = np.concatenate([np.random.uniform(0, 0.5, 50), np.random.uniform(0.5, 1, 50)])
    cur_df = pd.DataFrame({"label": y_cur, "pred": p_cur})
    return ref_df, cur_df


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    y_ref = np.array([0]*50 + [1]*50 + [2]*50)
    p_ref = np.array([0]*40 + [1]*10 + [1]*40 + [2]*10 + [2]*40 + [0]*10)
    ref_df = pd.DataFrame({"label": y_ref, "pred": p_ref})

    y_cur = np.array([0]*50 + [1]*50 + [2]*50)
    p_cur = np.array([0]*35 + [1]*15 + [1]*35 + [2]*15 + [2]*35 + [0]*15)
    cur_df = pd.DataFrame({"label": y_cur, "pred": p_cur})
    return ref_df, cur_df


@pytest.fixture
def data_with_nans():
    ref_df = pd.DataFrame({
        "label": [0, 1, np.nan, 1, 0],
        "pred": [0.2, 0.8, 0.5, np.nan, 0.3]
    })
    cur_df = pd.DataFrame({
        "label": [1, 0, 1, np.nan, 0],
        "pred": [0.7, 0.4, np.nan, 0.6, 0.2]
    })
    return ref_df, cur_df


# ---------------------------
#  Utility method tests
# ---------------------------
def test_is_classification():
    df_int = pd.DataFrame({"label": [0, 1, 0, 1]})
    df_float = pd.DataFrame({"label": [0.0, 1.0, 0.5, 1.5]})
    df_many = pd.DataFrame({"label": list(range(25))})

    eval_int = PerformanceEvaluator(df_int, df_int, "label", "dummy")
    assert eval_int._is_classification() is True

    eval_float = PerformanceEvaluator(df_float, df_float, "label", "dummy")
    assert eval_float._is_classification() is True

    eval_many = PerformanceEvaluator(df_many, df_many, "label", "dummy")
    assert eval_many._is_classification() is False


def test_get_classification_type():
    df_binary = pd.DataFrame({"label": [0, 1, 0, 1]})
    df_multiclass = pd.DataFrame({"label": [0, 1, 2, 0, 1, 2]})
    df_single = pd.DataFrame({"label": [1, 1, 1]})

    eval_bin = PerformanceEvaluator(df_binary, df_binary, "label", "dummy")
    assert eval_bin._get_classification_type() == "binary"

    eval_multi = PerformanceEvaluator(df_multiclass, df_multiclass, "label", "dummy")
    assert eval_multi._get_classification_type() == "multiclass"

    eval_single = PerformanceEvaluator(df_single, df_single, "label", "dummy")
    assert eval_single._get_classification_type() == "unknown"


# ---------------------------
#  Error cases
# ---------------------------
def test_missing_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    evaluator = PerformanceEvaluator(df, df, "label", "pred")
    result = evaluator.evaluate()
    assert "error" in result
    assert "Missing required columns" in result["error"]


def test_empty_after_dropna():
    ref_empty = pd.DataFrame({"label": [np.nan, np.nan], "pred": [np.nan, np.nan]})
    cur_empty = pd.DataFrame({"label": [np.nan, np.nan], "pred": [np.nan, np.nan]})
    evaluator = PerformanceEvaluator(ref_empty, cur_empty, "label", "pred")
    result = evaluator.evaluate()
    assert "error" in result
    assert "No valid (non-NaN) values" in result["error"]


# ---------------------------
#  Regression tests
# ---------------------------
def test_regression_evaluate(regression_data):
    ref_df, cur_df = regression_data
    evaluator = PerformanceEvaluator(ref_df, cur_df, "label", "pred")
    result = evaluator.evaluate()

    assert result["task_type"] == "regression"
    assert result["classification_subtype"] is None

    y_ref = ref_df["label"].values
    p_ref = ref_df["pred"].values
    y_cur = cur_df["label"].values
    p_cur = cur_df["pred"].values

    expected_ref_rmse = root_mean_squared_error(y_ref, p_ref)
    expected_ref_mae = mean_absolute_error(y_ref, p_ref)
    expected_cur_rmse = root_mean_squared_error(y_cur, p_cur)
    expected_cur_mae = mean_absolute_error(y_cur, p_cur)

    np.testing.assert_almost_equal(result["reference"]["rmse"], expected_ref_rmse)
    np.testing.assert_almost_equal(result["reference"]["mae"], expected_ref_mae)
    np.testing.assert_almost_equal(result["current"]["rmse"], expected_cur_rmse)
    np.testing.assert_almost_equal(result["current"]["mae"], expected_cur_mae)

    assert result["delta"]["rmse"] == pytest.approx(expected_cur_rmse - expected_ref_rmse)
    assert result["delta"]["mae"] == pytest.approx(expected_cur_mae - expected_ref_mae)

    expected_ref_r2 = r2_score(y_ref, p_ref)
    expected_cur_r2 = r2_score(y_cur, p_cur)
    np.testing.assert_almost_equal(result["reference"]["r2"], expected_ref_r2)
    np.testing.assert_almost_equal(result["current"]["r2"], expected_cur_r2)
    assert result["delta"]["r2"] == pytest.approx(expected_cur_r2 - expected_ref_r2)

    expected_ref_smape = _smape(y_ref, p_ref, epsilon=1e-8, as_percentage=True)
    expected_cur_smape = _smape(y_cur, p_cur, epsilon=1e-8, as_percentage=True)
    np.testing.assert_almost_equal(result["reference"]["smape"], expected_ref_smape)
    np.testing.assert_almost_equal(result["current"]["smape"], expected_cur_smape)
    assert result["delta"]["smape"] == pytest.approx(expected_cur_smape - expected_ref_smape)

    abs_err_ref = np.abs(y_ref - p_ref)
    abs_err_cur = np.abs(y_cur - p_cur)
    expected_ref_p90 = float(np.quantile(abs_err_ref, 0.90))
    expected_ref_p95 = float(np.quantile(abs_err_ref, 0.95))
    expected_cur_p90 = float(np.quantile(abs_err_cur, 0.90))
    expected_cur_p95 = float(np.quantile(abs_err_cur, 0.95))

    np.testing.assert_almost_equal(result["reference"]["p90_error"], expected_ref_p90)
    np.testing.assert_almost_equal(result["reference"]["p95_error"], expected_ref_p95)
    np.testing.assert_almost_equal(result["current"]["p90_error"], expected_cur_p90)
    np.testing.assert_almost_equal(result["current"]["p95_error"], expected_cur_p95)

    assert result["delta"]["p90_error"] == pytest.approx(expected_cur_p90 - expected_ref_p90)
    assert result["delta"]["p95_error"] == pytest.approx(expected_cur_p95 - expected_ref_p95)

    expected_ref_huber = _huber_loss(y_ref, p_ref, delta=1.0)
    expected_cur_huber = _huber_loss(y_cur, p_cur, delta=1.0)
    np.testing.assert_almost_equal(result["reference"]["huber_loss"], expected_ref_huber)
    np.testing.assert_almost_equal(result["current"]["huber_loss"], expected_cur_huber)
    assert result["delta"]["huber_loss"] == pytest.approx(expected_cur_huber - expected_ref_huber)


def test_regression_smape_ratio_and_huber_delta(regression_data):
    """SMAPE as ratio (0~2) and Huber delta sensitivity."""
    ref_df, cur_df = regression_data

    # Use non-default configs
    evaluator = PerformanceEvaluator(
        ref_df, cur_df, "label", "pred",
        task_type="regression",
        smape_as_percentage=False,  # ratio in [0, 2]
        huber_delta=0.5,            # smaller delta -> more linear penalty for residuals > 0.5
    )
    result = evaluator.evaluate()

    y_ref = ref_df["label"].values
    p_ref = ref_df["pred"].values
    y_cur = cur_df["label"].values
    p_cur = cur_df["pred"].values

    # SMAPE ratio checks
    expected_ref_smape_ratio = _smape(y_ref, p_ref, epsilon=1e-8, as_percentage=False)
    expected_cur_smape_ratio = _smape(y_cur, p_cur, epsilon=1e-8, as_percentage=False)
    assert 0.0 <= result["reference"]["smape"] <= 2.0
    assert 0.0 <= result["current"]["smape"] <= 2.0
    np.testing.assert_almost_equal(result["reference"]["smape"], expected_ref_smape_ratio)
    np.testing.assert_almost_equal(result["current"]["smape"], expected_cur_smape_ratio)

    # Huber with smaller delta vs larger delta: smaller delta should generally yield
    # a larger (or equal) loss due to earlier linear transition.
    evaluator_larger_delta = PerformanceEvaluator(
        ref_df, cur_df, "label", "pred",
        task_type="regression",
        huber_delta=2.0,
        smape_as_percentage=False,
    )
    result_larger_delta = evaluator_larger_delta.evaluate()

    assert result["reference"]["huber_loss"] <= result_larger_delta["reference"]["huber_loss"] - 1e-12
    assert result["current"]["huber_loss"] <= result_larger_delta["current"]["huber_loss"] - 1e-12

# ---------------------------
#  Binary classification (labels 0/1)
# ---------------------------
def test_binary_evaluate_01(binary_data_01):
    ref_df, cur_df = binary_data_01
    evaluator = PerformanceEvaluator(ref_df, cur_df, "label", "pred")
    result = evaluator.evaluate()

    assert result["task_type"] == "classification"
    assert result["classification_subtype"] == "binary"

    y_ref = ref_df["label"].values
    p_ref = ref_df["pred"].values
    y_cur = cur_df["label"].values
    p_cur = cur_df["pred"].values

    # AUC
    expected_ref_auc = roc_auc_score(y_ref, p_ref)
    expected_cur_auc = roc_auc_score(y_cur, p_cur)

    # KS
    pos_label = 1
    ks_ref = compute_ks(p_ref[y_ref != pos_label], p_ref[y_ref == pos_label], return_pvalue=False)["statistic"]
    ks_cur = compute_ks(p_cur[y_cur != pos_label], p_cur[y_cur == pos_label], return_pvalue=False)["statistic"]

    # Threshold-based metrics
    y_hat_ref = (p_ref >= 0.5).astype(int)
    y_hat_cur = (p_cur >= 0.5).astype(int)

    expected_ref_acc = accuracy_score(y_ref, y_hat_ref)
    expected_cur_acc = accuracy_score(y_cur, y_hat_cur)
    expected_ref_prec = precision_score(y_ref, y_hat_ref, zero_division=0)
    expected_cur_prec = precision_score(y_cur, y_hat_cur, zero_division=0)
    expected_ref_rec = recall_score(y_ref, y_hat_ref, zero_division=0)
    expected_cur_rec = recall_score(y_cur, y_hat_cur, zero_division=0)
    expected_ref_f1 = f1_score(y_ref, y_hat_ref, zero_division=0)
    expected_cur_f1 = f1_score(y_cur, y_hat_cur, zero_division=0)

    np.testing.assert_almost_equal(result["reference"]["auc"], expected_ref_auc)
    np.testing.assert_almost_equal(result["reference"]["ks"], ks_ref)
    np.testing.assert_almost_equal(result["reference"]["accuracy"], expected_ref_acc)
    np.testing.assert_almost_equal(result["reference"]["precision"], expected_ref_prec)
    np.testing.assert_almost_equal(result["reference"]["recall"], expected_ref_rec)
    np.testing.assert_almost_equal(result["reference"]["f1"], expected_ref_f1)

    np.testing.assert_almost_equal(result["current"]["auc"], expected_cur_auc)
    np.testing.assert_almost_equal(result["current"]["ks"], ks_cur)
    np.testing.assert_almost_equal(result["current"]["accuracy"], expected_cur_acc)
    np.testing.assert_almost_equal(result["current"]["precision"], expected_cur_prec)
    np.testing.assert_almost_equal(result["current"]["recall"], expected_cur_rec)
    np.testing.assert_almost_equal(result["current"]["f1"], expected_cur_f1)

    assert result["delta"]["auc"] == pytest.approx(expected_cur_auc - expected_ref_auc)
    assert result["delta"]["ks"] == pytest.approx(ks_cur - ks_ref)


def test_binary_evaluate_non_01_labels(binary_data_25):
    """Non‑0/1 labels trigger a warning but are handled correctly (no ValueError)."""
    ref_df, cur_df = binary_data_25
    evaluator = PerformanceEvaluator(ref_df, cur_df, "label", "pred")

    # The evaluator should warn about the non‑standard labels
    with pytest.warns(UserWarning, match="Binary classification detected but labels are"):
        result = evaluator.evaluate()

    # No exception is raised; metrics are computed correctly.
    assert "error" not in result
    assert result["task_type"] == "classification"
    assert result["classification_subtype"] == "binary"
    # Check that essential metrics exist and are finite
    for key in ["auc", "ks", "accuracy", "precision", "recall", "f1"]:
        assert key in result["reference"]
        assert np.isfinite(result["reference"][key])
        assert key in result["current"]
        assert np.isfinite(result["current"][key])


# ---------------------------
#  Multiclass tests
# ---------------------------
def test_multiclass_evaluate(multiclass_data):
    ref_df, cur_df = multiclass_data
    evaluator = PerformanceEvaluator(ref_df, cur_df, "label", "pred")
    result = evaluator.evaluate()

    assert result["task_type"] == "classification"
    assert result["classification_subtype"] == "multiclass"

    y_ref = ref_df["label"].values
    p_ref = ref_df["pred"].values.astype(int)
    y_cur = cur_df["label"].values
    p_cur = cur_df["pred"].values.astype(int)

    # Accuracy
    expected_ref_acc = accuracy_score(y_ref, p_ref)
    expected_cur_acc = accuracy_score(y_cur, p_cur)

    # Macro averages
    expected_ref_prec_macro = precision_score(y_ref, p_ref, average="macro", zero_division=0)
    expected_cur_prec_macro = precision_score(y_cur, p_cur, average="macro", zero_division=0)
    expected_ref_rec_macro = recall_score(y_ref, p_ref, average="macro", zero_division=0)
    expected_cur_rec_macro = recall_score(y_cur, p_cur, average="macro", zero_division=0)
    expected_ref_f1_macro = f1_score(y_ref, p_ref, average="macro", zero_division=0)
    expected_cur_f1_macro = f1_score(y_cur, p_cur, average="macro", zero_division=0)

    # Weighted averages
    expected_ref_prec_weighted = precision_score(y_ref, p_ref, average="weighted", zero_division=0)
    expected_cur_prec_weighted = precision_score(y_cur, p_cur, average="weighted", zero_division=0)
    expected_ref_rec_weighted = recall_score(y_ref, p_ref, average="weighted", zero_division=0)
    expected_cur_rec_weighted = recall_score(y_cur, p_cur, average="weighted", zero_division=0)
    expected_ref_f1_weighted = f1_score(y_ref, p_ref, average="weighted", zero_division=0)
    expected_cur_f1_weighted = f1_score(y_cur, p_cur, average="weighted", zero_division=0)

    # Compare
    np.testing.assert_almost_equal(result["reference"]["accuracy"], expected_ref_acc)
    np.testing.assert_almost_equal(result["reference"]["precision_macro"], expected_ref_prec_macro)
    np.testing.assert_almost_equal(result["reference"]["recall_macro"], expected_ref_rec_macro)
    np.testing.assert_almost_equal(result["reference"]["f1_macro"], expected_ref_f1_macro)
    np.testing.assert_almost_equal(result["reference"]["precision_weighted"], expected_ref_prec_weighted)
    np.testing.assert_almost_equal(result["reference"]["recall_weighted"], expected_ref_rec_weighted)
    np.testing.assert_almost_equal(result["reference"]["f1_weighted"], expected_ref_f1_weighted)

    np.testing.assert_almost_equal(result["current"]["accuracy"], expected_cur_acc)
    np.testing.assert_almost_equal(result["current"]["precision_macro"], expected_cur_prec_macro)
    np.testing.assert_almost_equal(result["current"]["recall_macro"], expected_cur_rec_macro)
    np.testing.assert_almost_equal(result["current"]["f1_macro"], expected_cur_f1_macro)
    np.testing.assert_almost_equal(result["current"]["precision_weighted"], expected_cur_prec_weighted)
    np.testing.assert_almost_equal(result["current"]["recall_weighted"], expected_cur_rec_weighted)
    np.testing.assert_almost_equal(result["current"]["f1_weighted"], expected_cur_f1_weighted)

    # Deltas
    assert result["delta"]["accuracy"] == pytest.approx(expected_cur_acc - expected_ref_acc)


# ---------------------------
#  NaN handling
# ---------------------------
def test_nan_handling(data_with_nans):
    ref_df, cur_df = data_with_nans
    evaluator = PerformanceEvaluator(ref_df, cur_df, "label", "pred")
    result = evaluator.evaluate()

    assert "error" not in result
    assert result["task_type"] == "classification"
    assert result["classification_subtype"] == "binary"
    assert not np.isnan(result["reference"]["auc"])
    assert not np.isnan(result["current"]["auc"])
    assert not np.isnan(result["reference"]["ks"])
    assert not np.isnan(result["current"]["ks"])


def test_all_nan_after_dropna():
    ref_df = pd.DataFrame({"label": [1, 0, 1], "pred": [0.2, 0.8, 0.3]})
    cur_df = pd.DataFrame({"label": [np.nan, np.nan], "pred": [np.nan, np.nan]})
    evaluator = PerformanceEvaluator(ref_df, cur_df, "label", "pred")
    result = evaluator.evaluate()
    assert "error" in result
    assert "No valid (non-NaN) values" in result["error"]


# ---------------------------
#  KS edge cases
# ---------------------------
def test_ks_with_empty_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3])
    evaluator = PerformanceEvaluator(pd.DataFrame(), pd.DataFrame(), "label", "pred")
    metrics = evaluator._classification_metrics(y_true, y_pred, task_type="binary")
    assert np.isnan(metrics["ks"])
