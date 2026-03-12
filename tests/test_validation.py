
import numpy as np
import pandas as pd

from momo_ml.utils.validation import (
    ValidationReport,
    ValidationError,
    validate_monitor_inputs,
    validate_dataframe_basic,
    validate_required_columns,
    validate_missing_ratio,
    validate_unique_values,
    ensure_numeric_columns,
    assert_same_schema,
    assert_compatible_dtypes,
    infer_task_type,
    infer_task_type as infer_task_type_from_labels,
    validate_binary_labels,
    validate_prediction_probabilities,
)


# ---------------------------------------------------------
# Basic DataFrame validation
# ---------------------------------------------------------

def test_validate_dataframe_basic_ok():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    report = ValidationReport()
    validate_dataframe_basic(df, "df", report)
    assert report.ok


def test_validate_dataframe_basic_missing_columns():
    df = pd.DataFrame()
    report = ValidationReport()
    validate_dataframe_basic(df, "df", report)
    assert not report.ok
    assert "[df] has 0 columns." in "\n".join(report.errors)


def test_validate_dataframe_basic_mixed_types_warning():
    df = pd.DataFrame({"a": [1, "x", 2.5]})
    report = ValidationReport()
    validate_dataframe_basic(df, "df", report)
    # Should warn about mixed Python types, but not error
    assert report.ok
    assert any("mixed Python types" in w for w in report.warnings)


# ---------------------------------------------------------
# Required column tests
# ---------------------------------------------------------

def test_validate_required_columns_missing():
    df = pd.DataFrame({"a": [1, 2]})
    report = ValidationReport()
    validate_required_columns(df, ["a", "b"], name="df", report=report)
    assert not report.ok
    assert "missing columns" in report.errors[0]


def test_validate_required_columns_ok():
    df = pd.DataFrame({"a": [1]})
    report = ValidationReport()
    validate_required_columns(df, ["a"], name="df", report=report)
    assert report.ok


# ---------------------------------------------------------
# Missing ratio tests
# ---------------------------------------------------------

def test_validate_missing_ratio_warn_and_error():
    df = pd.DataFrame({
        "a": [1, None, None, None],  # 75% missing → warning threshold < ratio < error threshold
        "b": [None, None, None, None],  # 100% missing → error
    })
    report = ValidationReport()
    validate_missing_ratio(df, ["a", "b"], name="df", report=report,
                           warn_threshold=0.3, error_threshold=0.9)
    assert not report.ok  # because "b" missing ratio >= error_threshold
    assert any("missing ratio" in msg for msg in report.errors)


# ---------------------------------------------------------
# Unique value tests
# ---------------------------------------------------------

def test_validate_unique_values_warn():
    df = pd.DataFrame({"a": [1, 1, 1]})
    report = ValidationReport()
    validate_unique_values(df, ["a"], name="df", report=report)
    assert report.ok
    assert any("unique value" in w for w in report.warnings)


# ---------------------------------------------------------
# ensure_numeric_columns tests
# ---------------------------------------------------------

def test_ensure_numeric_columns_error():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    report = ValidationReport()
    ensure_numeric_columns(df, ["a"], name="df", report=report)
    assert not report.ok
    assert "non-numeric" in report.errors[0]


# ---------------------------------------------------------
# Schema tests (ref vs cur)
# ---------------------------------------------------------

def test_assert_same_schema_intersection():
    ref = pd.DataFrame({"a": [1], "b": [1]})
    cur = pd.DataFrame({"b": [1], "c": [1]})

    report = ValidationReport()
    features = assert_same_schema(ref, cur, cols=None, report=report)

    assert features == ["b"]
    assert report.ok


def test_assert_same_schema_explicit_missing():
    ref = pd.DataFrame({"a": [1], "b": [1]})
    cur = pd.DataFrame({"b": [1], "c": [1]})

    report = ValidationReport()
    features = assert_same_schema(ref, cur, cols=["a", "c"], report=report)

    assert not report.ok
    assert "missing columns" in "\n".join(report.errors)


def test_assert_compatible_dtypes_warn():
    ref = pd.DataFrame({"x": [1, 2, 3]})
    cur = pd.DataFrame({"x": ["1", "2", "3"]})
    report = ValidationReport()
    assert_compatible_dtypes(ref, cur, ["x"], report=report)
    assert any("dtype mismatch" in w for w in report.warnings)


# ---------------------------------------------------------
# Task type inference
# ---------------------------------------------------------

def test_infer_task_type_classification():
    y = pd.Series([0, 1, 1, 0])
    assert infer_task_type(y) == "classification"


def test_infer_task_type_regression():
    y = pd.Series([1.1, 2.2, 3.3, 4.4])
    assert infer_task_type(y) == "regression"


# ---------------------------------------------------------
# Binary label validator
# ---------------------------------------------------------

def test_validate_binary_labels_warn_non_binary():
    y = pd.Series([0, 1, 2])  # not binary
    report = ValidationReport()
    validate_binary_labels(y, name="y", report=report)
    assert any("not binary" in w for w in report.warnings)


# ---------------------------------------------------------
# Predicted probability validator
# ---------------------------------------------------------

def test_validate_prediction_probabilities_warn():
    p = pd.Series([-0.1, 0.2, 1.2])  # out of range
    report = ValidationReport()
    validate_prediction_probabilities(p, name="pred", report=report)
    assert any("outside [0,1]" in w for w in report.warnings)


# ---------------------------------------------------------
# validate_monitor_inputs E2E tests
# ---------------------------------------------------------

def test_validate_monitor_inputs_ok():
    ref_df = pd.DataFrame({
        "x": [1, 2, 3],
        "label": [0, 1, 1],
        "pred": [0.2, 0.8, 0.6],
    })
    cur_df = ref_df.copy()

    report = validate_monitor_inputs(
        ref_df, cur_df, label_col="label", pred_col="pred"
    )
    assert report.ok


def test_validate_monitor_inputs_missing_cols():
    ref_df = pd.DataFrame({"label": [0, 1, 1]})
    cur_df = pd.DataFrame({"wrong": [1, 2, 3]})

    report = validate_monitor_inputs(
        ref_df, cur_df, label_col="label", pred_col="pred"
    )

    assert not report.ok
    assert "missing columns" in "\n".join(report.errors)


def test_validate_monitor_inputs_feature_schema():
    ref_df = pd.DataFrame({
        "x": [1, 2, 3],
        "label": [0, 1, 1],
        "pred": [0.2, 0.8, 0.6],
    })
    cur_df = pd.DataFrame({
        "x": ["A", "B", "C"],   # dtype mismatch
        "label": [1, 0, 1],
        "pred": [0.3, 0.4, 0.5],
    })

    report = validate_monitor_inputs(
        ref_df, cur_df, label_col="label", pred_col="pred"
    )

    assert report.ok  # dtype mismatch is a WARN, not an ERROR
    assert any("dtype mismatch" in w for w in report.warnings)


def test_validate_monitor_inputs_missing_values():
    ref_df = pd.DataFrame({
        "x": [None, None, None],
        "label": [0, 1, 1],
        "pred": [0.1, 0.2, 0.3],
    })
    cur_df = ref_df.copy()

    report = validate_monitor_inputs(
        ref_df, cur_df, label_col="label", pred_col="pred"
    )
    # x missing ratio = 100% → should error
    assert not report.ok
    assert any("missing ratio" in e for e in report.errors)