import pytest
import json
from pathlib import Path

from momo_ml.report.report_builder import ReportBuilder


@pytest.fixture
def dummy_results():
    """Provide a standard dummy monitoring result dictionary."""
    return {
        "performance_drift": {
            "task_type": "classification",
            "classification_subtype": "binary",
            "reference": {"auc": 0.95, "accuracy": 0.92, "f1": 0.91},
            "current": {"auc": 0.93, "accuracy": 0.90, "f1": 0.89},
            "delta": {"auc": -0.02, "accuracy": -0.02, "f1": -0.02},
        },
        "data_drift": {
            "numeric_features": {
                "age": {"psi": 0.12, "ks": {"statistic": 0.08}, "kl": 0.03, "js": 0.01, "wd": 1.2},
                "income": {"psi": 0.32, "ks": {"statistic": 0.15}, "kl": 0.11, "js": 0.05, "wd": 500},
            },
            "categorical_features": {
                "gender": {"psi": 0.05, "kl": 0.02, "js": 0.01, "wd": 0.03},
            },
            "incompatible_features": ["old_feature"],
        },
        "prediction_drift": {
            "prediction_type": "continuous",
            "summary_statistics": {
                "mean": {"reference": 0.5, "current": 0.52, "delta": 0.02},
                "std": {"reference": 0.1, "current": 0.12, "delta": 0.02},
                "min": {"reference": 0.0, "current": 0.0},
                "max": {"reference": 1.0, "current": 1.0},
                "q25": {"reference": 0.25, "current": 0.27},
                "q50": {"reference": 0.5, "current": 0.52},
                "q75": {"reference": 0.75, "current": 0.77},
            },
            "distribution_shift": {
                "psi": 0.08,
                "ks": {"statistic": 0.04, "pvalue": 0.2},
                "kl": 0.02,
                "js": 0.01,
                "l1_distance": 0.3,
                "l2_distance": 0.12,
            },
            "decile_shift": {
                "quantiles": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "ref_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "cur_values": [0.0, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 1.0],
                "delta": [0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.0],
            },
        },
    }


@pytest.fixture
def error_results():
    """Results with errors in each section."""
    return {
        "performance_drift": {"error": "Performance drift evaluation failed: missing labels"},
        "data_drift": {"error": "Data drift computation failed: incompatible types"},
        "prediction_drift": {"error": "Prediction drift computation failed: no predictions"},
    }


@pytest.fixture
def empty_results():
    """Empty results (no sections)."""
    return {}


def test_report_builder_initialization(dummy_results):
    """Test that ReportBuilder initializes correctly with and without thresholds."""
    builder = ReportBuilder(dummy_results)
    assert builder.results == dummy_results
    assert "psi" in builder.thresholds
    assert builder.thresholds["psi"]["low"] == 0.1

    custom_thresholds = {"psi": {"low": 0.05, "medium": 0.15}, "kl": {"low": 0.2}}
    builder2 = ReportBuilder(dummy_results, thresholds=custom_thresholds)
    assert builder2.thresholds["psi"]["low"] == 0.05
    assert builder2.thresholds["psi"]["medium"] == 0.15
    assert builder2.thresholds["kl"]["low"] == 0.2


def test_to_markdown_basic(dummy_results):
    """Test that to_markdown returns a string with expected sections."""
    builder = ReportBuilder(dummy_results)
    markdown = builder.to_markdown(title="Test Report", include_metadata=True)

    assert "Test Report" in markdown
    assert "Performance drift" in markdown
    assert "Data drift" in markdown
    assert "Prediction drift" in markdown

    # Performance table
    assert "auc" in markdown
    assert "0.9500" in markdown
    assert "0.9300" in markdown

    # Data drift numeric features
    assert "age" in markdown
    assert "🟡 0.1200" in markdown
    assert "0.0800" in markdown
    assert "🟢 0.0300" in markdown
    assert "🟢 0.0100" in markdown
    assert "1.2000" in markdown

    assert "income" in markdown
    assert "🔴 0.3200" in markdown
    assert "0.1500" in markdown
    assert "🟡 0.1100" in markdown
    assert "🟢 0.0500" in markdown
    assert "500.0000" in markdown

    # Data drift categorical features
    assert "gender" in markdown
    assert "🟢 0.0500" in markdown
    assert "🟢 0.0200" in markdown
    assert "🟢 0.0100" in markdown
    assert "0.0300" in markdown

    assert "old_feature" in markdown

    # Prediction drift continuous
    assert "mean" in markdown
    assert "0.5000" in markdown
    assert "0.5200" in markdown

    assert "psi" in markdown
    assert "🟢 0.0800" in markdown
    assert "KS (statistic)" in markdown
    assert "0.0400" in markdown
    assert "0.2000" in markdown

    assert "decile shift" in markdown.lower()
    assert "0.00" in markdown
    assert "0.0000" in markdown


def test_to_markdown_without_metadata(dummy_results):
    """Test that to_markdown can skip metadata."""
    builder = ReportBuilder(dummy_results)
    markdown = builder.to_markdown(include_metadata=False)
    assert "# Model Monitoring Report" not in markdown
    assert "Performance drift" in markdown


def test_to_markdown_with_custom_title(dummy_results):
    """Test custom title."""
    builder = ReportBuilder(dummy_results)
    markdown = builder.to_markdown(title="Custom Title")
    assert "Custom Title" in markdown


def test_to_markdown_with_errors(error_results):
    """Test Markdown generation when errors are present."""
    builder = ReportBuilder(error_results)
    markdown = builder.to_markdown()
    assert "Performance drift error" in markdown
    assert "Data drift error" in markdown
    assert "Prediction drift error" in markdown


def test_to_markdown_with_empty_results(empty_results):
    """Test Markdown generation with empty results."""
    builder = ReportBuilder(empty_results)
    markdown = builder.to_markdown()
    assert "No performance drift data available." in markdown
    assert "No data drift data available." in markdown
    assert "No prediction drift data available." in markdown


def test_to_json(dummy_results):
    """Test that to_json returns a valid JSON string."""
    builder = ReportBuilder(dummy_results)
    json_str = builder.to_json(indent=2)
    data = json.loads(json_str)
    assert "timestamp" in data
    assert "results" in data
    assert data["results"] == dummy_results


def test_save_markdown(dummy_results, tmp_path):
    """Test saving Markdown to a file."""
    builder = ReportBuilder(dummy_results)
    filepath = tmp_path / "report.md"
    builder.save_markdown(str(filepath), title="Saved Report")
    assert filepath.exists()
    content = filepath.read_text(encoding="utf-8")
    assert "Saved Report" in content
    assert "age" in content
    assert "🟡 0.1200" in content


def test_save_json(dummy_results, tmp_path):
    """Test saving JSON to a file."""
    builder = ReportBuilder(dummy_results)
    filepath = tmp_path / "report.json"
    builder.save_json(str(filepath), indent=2)
    assert filepath.exists()
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "timestamp" in data
    assert data["results"] == dummy_results


def test_custom_thresholds(dummy_results):
    """Test that custom thresholds affect risk badges."""
    # Default thresholds
    builder = ReportBuilder(dummy_results)
    markdown = builder.to_markdown()
    assert "🟡 0.1200" in markdown   # age PSI moderate
    assert "🔴 0.3200" in markdown   # income PSI high
    assert "🟢 0.0500" in markdown   # gender PSI low

    # Custom thresholds: PSI low <0.15, medium <0.35 => age low, income moderate
    custom = {"psi": {"low": 0.15, "medium": 0.35}}
    builder2 = ReportBuilder(dummy_results, thresholds=custom)
    markdown2 = builder2.to_markdown()
    assert "🟢 0.1200" in markdown2
    assert "🟡 0.3200" in markdown2
    assert "🟢 0.0500" in markdown2

    # Custom thresholds for KL
    custom_kl = {"kl": {"low": 0.05, "medium": 0.2}}
    builder3 = ReportBuilder(dummy_results, thresholds=custom_kl)
    markdown3 = builder3.to_markdown()
    assert "🟢 0.0300" in markdown3   # age KL low
    assert "🟡 0.1100" in markdown3   # income KL moderate


def test_numeric_feature_with_missing_metrics(dummy_results):
    """Test that missing metrics are handled gracefully."""
    results = dummy_results.copy()
    # Remove KL from age
    results["data_drift"]["numeric_features"]["age"].pop("kl")
    builder = ReportBuilder(results)
    markdown = builder.to_markdown()
    assert "age" in markdown
    assert "🟡 0.1200" in markdown
    assert "0.0800" in markdown
    assert "N/A" in markdown          # KL column shows N/A
    assert "🟢 0.0100" in markdown    # JS still has badge


def test_categorical_feature_with_missing_metrics(dummy_results):
    """Test that categorical feature handles missing metrics."""
    results = dummy_results.copy()
    # Remove JS from gender
    results["data_drift"]["categorical_features"]["gender"].pop("js")
    builder = ReportBuilder(results)
    markdown = builder.to_markdown()
    assert "gender" in markdown
    assert "🟢 0.0500" in markdown
    assert "🟢 0.0200" in markdown
    assert "N/A" in markdown          # JS column shows N/A
    assert "0.0300" in markdown


def test_prediction_drift_categorical_type():
    """Test handling of categorical prediction type."""
    results = {
        "prediction_drift": {
            "prediction_type": "categorical",
            "summary_statistics": {
                "categories": ["A", "B", "C"],
                "reference_proportions": {"A": 0.5, "B": 0.3, "C": 0.2},
                "current_proportions": {"A": 0.6, "B": 0.2, "C": 0.2},
                "delta_proportions": {"A": 0.1, "B": -0.1, "C": 0.0},
            },
            "distribution_shift": {"psi": 0.08, "kl": 0.05, "js": 0.02},
        }
    }
    builder = ReportBuilder(results)
    markdown = builder.to_markdown()
    assert "Category proportions" in markdown
    assert "A" in markdown
    assert "0.5000" in markdown
    assert "0.6000" in markdown
    assert "0.1000" in markdown
    assert "Distribution shift metrics" in markdown
    assert "psi" in markdown
    assert "🟢 0.0800" in markdown
    assert "kl" in markdown
    assert "🟢 0.0500" in markdown
    assert "js" in markdown
    assert "🟢 0.0200" in markdown


def test_no_incompatible_features():
    """Test that incompatible features list is omitted when empty."""
    results = {
        "data_drift": {
            "numeric_features": {"age": {"psi": 0.1}},
            "categorical_features": {},
            "incompatible_features": [],
        }
    }
    builder = ReportBuilder(results)
    markdown = builder.to_markdown()
    assert "⚠️ **Incompatible features" not in markdown