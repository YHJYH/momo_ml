# momo_ml/tests/test_model_monitor.py

import pytest
import pandas as pd
from unittest.mock import Mock, patch, PropertyMock

from momo_ml.monitor.model_monitor import ModelMonitor


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_dataframes():
    """Create simple reference and current DataFrames for testing."""
    ref_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c'],
        'label': [0, 1, 0],
        'pred': [0.1, 0.9, 0.2]
    })
    cur_df = pd.DataFrame({
        'feature1': [4, 5, 6],
        'feature2': ['a', 'b', 'd'],
        'label': [1, 0, 1],
        'pred': [0.8, 0.3, 0.7]
    })
    return ref_df, cur_df


# ----------------------------------------------------------------------
# Tests for initialization and lazy properties
# ----------------------------------------------------------------------
def test_initialization_default(sample_dataframes):
    """Test that ModelMonitor can be created with minimal arguments."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df)

    assert monitor.ref_df is ref_df
    assert monitor.cur_df is cur_df
    assert monitor.label_col is None
    assert monitor.pred_col is None
    assert monitor.data_drift_kwargs == {}
    assert monitor.performance_kwargs == {}
    assert monitor.prediction_drift_kwargs == {}
    # Lazy placeholders should be None
    assert monitor._data_drift is None
    assert monitor._performance is None
    assert monitor._prediction_drift is None


def test_initialization_with_kwargs(sample_dataframes):
    """Test that constructor stores per-detector kwargs correctly."""
    ref_df, cur_df = sample_dataframes
    data_kwargs = {'features': ['feature1'], 'bins': 15}
    perf_kwargs = {'task_type': 'classification'}
    pred_kwargs = {'include_ks': True}

    monitor = ModelMonitor(
        ref_df, cur_df,
        label_col='label',
        pred_col='pred',
        data_drift_kwargs=data_kwargs,
        performance_kwargs=perf_kwargs,
        prediction_drift_kwargs=pred_kwargs
    )

    assert monitor.label_col == 'label'
    assert monitor.pred_col == 'pred'
    assert monitor.data_drift_kwargs == data_kwargs
    assert monitor.performance_kwargs == perf_kwargs
    assert monitor.prediction_drift_kwargs == pred_kwargs


@patch('momo_ml.monitor.model_monitor.DataDriftDetector')
@patch('momo_ml.monitor.model_monitor.PerformanceEvaluator')
@patch('momo_ml.monitor.model_monitor.PredictionDriftDetector')
def test_lazy_initialization(mock_pred_drift, mock_perf, mock_data_drift, sample_dataframes):
    """Verify that detectors are created only when first accessed."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df, label_col='label', pred_col='pred')

    # Initially no constructors called
    mock_data_drift.assert_not_called()
    mock_perf.assert_not_called()
    mock_pred_drift.assert_not_called()

    # Access data_drift property -> constructor should be called once
    dd = monitor.data_drift
    mock_data_drift.assert_called_once_with(ref_df, cur_df)
    mock_perf.assert_not_called()
    mock_pred_drift.assert_not_called()

    # Access again should not create new instance
    mock_data_drift.reset_mock()
    dd2 = monitor.data_drift
    mock_data_drift.assert_not_called()
    assert dd is dd2

    # Access performance property
    perf = monitor.performance
    mock_perf.assert_called_once_with(ref_df, cur_df, 'label', 'pred')
    mock_pred_drift.assert_not_called()

    # Access prediction_drift property
    pred = monitor.prediction_drift
    mock_pred_drift.assert_called_once_with(ref_df, cur_df, 'pred')


# ----------------------------------------------------------------------
# Tests for run_* methods (with mocked detectors)
# ----------------------------------------------------------------------
@patch('momo_ml.monitor.model_monitor.PerformanceEvaluator')
def test_run_performance_drift(mock_perf_class, sample_dataframes):
    """Test that run_performance_drift calls evaluate and returns result."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df, label_col='label', pred_col='pred')

    # Mock the evaluate method of the instance
    mock_perf_instance = Mock()
    expected_result = {'task_type': 'classification', 'reference': {'auc': 0.8}}
    mock_perf_instance.evaluate.return_value = expected_result
    # Make the property return our mock
    mock_perf_class.return_value = mock_perf_instance

    result = monitor.run_performance_drift()

    # Ensure the detector was created (lazy) and evaluate called
    mock_perf_class.assert_called_once_with(ref_df, cur_df, 'label', 'pred')
    mock_perf_instance.evaluate.assert_called_once()
    assert result == expected_result


@patch('momo_ml.monitor.model_monitor.DataDriftDetector')
def test_run_data_drift(mock_data_class, sample_dataframes):
    """Test run_data_drift calls compute and returns result."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df)

    mock_data_instance = Mock()
    expected_result = {'numeric_features': {}, 'categorical_features': {}}
    mock_data_instance.compute.return_value = expected_result
    mock_data_class.return_value = mock_data_instance

    result = monitor.run_data_drift()

    mock_data_class.assert_called_once_with(ref_df, cur_df)
    mock_data_instance.compute.assert_called_once()
    assert result == expected_result


@patch('momo_ml.monitor.model_monitor.PredictionDriftDetector')
def test_run_prediction_drift(mock_pred_class, sample_dataframes):
    """Test run_prediction_drift calls compute and returns result."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df, pred_col='pred')

    mock_pred_instance = Mock()
    expected_result = {'prediction_type': 'continuous', 'summary_statistics': {}}
    mock_pred_instance.compute.return_value = expected_result
    mock_pred_class.return_value = mock_pred_instance

    result = monitor.run_prediction_drift()

    mock_pred_class.assert_called_once_with(ref_df, cur_df, 'pred')
    mock_pred_instance.compute.assert_called_once()
    assert result == expected_result


@patch('momo_ml.monitor.model_monitor.ModelMonitor.run_performance_drift')
@patch('momo_ml.monitor.model_monitor.ModelMonitor.run_data_drift')
@patch('momo_ml.monitor.model_monitor.ModelMonitor.run_prediction_drift')
def test_run_all(mock_run_pred, mock_run_data, mock_run_perf, sample_dataframes):
    """Test that run_all calls all three run methods and combines results."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df, label_col='label', pred_col='pred')

    mock_run_perf.return_value = {'perf': 1}
    mock_run_data.return_value = {'data': 2}
    mock_run_pred.return_value = {'pred': 3}

    result = monitor.run_all()

    mock_run_perf.assert_called_once()
    mock_run_data.assert_called_once()
    mock_run_pred.assert_called_once()
    assert result == {
        'performance_drift': {'perf': 1},
        'data_drift': {'data': 2},
        'prediction_drift': {'pred': 3}
    }


# ----------------------------------------------------------------------
# Tests for error handling in run_* methods
# ----------------------------------------------------------------------
@patch('momo_ml.monitor.model_monitor.PerformanceEvaluator')
def test_run_performance_drift_exception(mock_perf_class, sample_dataframes):
    """Test that an exception in performance.evaluate is caught and returned as error."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df, label_col='label', pred_col='pred')

    mock_perf_instance = Mock()
    mock_perf_instance.evaluate.side_effect = ValueError("Something went wrong")
    mock_perf_class.return_value = mock_perf_instance

    with pytest.warns(RuntimeWarning, match="Performance drift evaluation failed"):
        result = monitor.run_performance_drift()

    assert "error" in result
    assert "Something went wrong" in result["error"]


@patch('momo_ml.monitor.model_monitor.DataDriftDetector')
def test_run_data_drift_exception(mock_data_class, sample_dataframes):
    """Test that an exception in data_drift.compute is caught and returned as error."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df)

    mock_data_instance = Mock()
    mock_data_instance.compute.side_effect = TypeError("Type error")
    mock_data_class.return_value = mock_data_instance

    with pytest.warns(RuntimeWarning, match="Data drift computation failed"):
        result = monitor.run_data_drift()

    assert "error" in result
    assert "Type error" in result["error"]


@patch('momo_ml.monitor.model_monitor.PredictionDriftDetector')
def test_run_prediction_drift_exception(mock_pred_class, sample_dataframes):
    """Test that an exception in prediction_drift.compute is caught and returned as error."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df, pred_col='pred')

    mock_pred_instance = Mock()
    mock_pred_instance.compute.side_effect = KeyError("Missing column")
    mock_pred_class.return_value = mock_pred_instance

    with pytest.warns(RuntimeWarning, match="Prediction drift computation failed"):
        result = monitor.run_prediction_drift()

    assert "error" in result
    assert "Missing column" in result["error"]


# ----------------------------------------------------------------------
# Additional test: ensure run_all aggregates even if one method fails
# ----------------------------------------------------------------------
@patch('momo_ml.monitor.model_monitor.ModelMonitor.run_performance_drift')
@patch('momo_ml.monitor.model_monitor.ModelMonitor.run_data_drift')
@patch('momo_ml.monitor.model_monitor.ModelMonitor.run_prediction_drift')
def test_run_all_with_errors(mock_run_pred, mock_run_data, mock_run_perf, sample_dataframes):
    """Verify that run_all returns combined dict even if some runs return errors."""
    ref_df, cur_df = sample_dataframes
    monitor = ModelMonitor(ref_df, cur_df)

    mock_run_perf.return_value = {'error': 'performance failed'}
    mock_run_data.return_value = {'data': 'ok'}
    mock_run_pred.return_value = {'error': 'prediction failed'}

    result = monitor.run_all()

    assert result['performance_drift'] == {'error': 'performance failed'}
    assert result['data_drift'] == {'data': 'ok'}
    assert result['prediction_drift'] == {'error': 'prediction failed'}