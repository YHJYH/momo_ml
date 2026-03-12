
import os
import re
import base64
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from momo_ml.report import ReportBuilder
from momo_ml.monitor import ModelMonitor
from momo_ml.utils.plotting import plot_distribution


# ---------------------------------------------------------
# Helper: minimal monitor output
# ---------------------------------------------------------

def _fake_monitor_output():
    return {
        "performance_drift": {
            "task_type": "classification",
            "reference": {"auc": 0.9},
            "current": {"auc": 0.85},
            "delta": {"auc": -0.05},
        },
        "data_drift": {
            "numeric_features": {"x1": {"psi": 0.1}},
            "categorical_features": {"cat": {"psi": 0.2}},
        },
        "prediction_drift": {
            "summary_statistics": {
                "mean": {"reference": 0.4, "current": 0.5, "delta": 0.1}
            },
            "distribution_shift": {"l1_distance": 0.3, "l2_distance": 0.2},
            "decile_shift": {
                "ref_deciles": [0] * 11,
                "cur_deciles": [0] * 11,
                "delta": [0] * 11,
            },
        },
    }


# ---------------------------------------------------------
# 1. HTML generation — basic correctness
# ---------------------------------------------------------

def test_report_builder_generates_html_string():
    monitor_out = _fake_monitor_output()
    rb = ReportBuilder(monitor_output=monitor_out, plots=None, title="Test Report")

    html = rb.to_html()

    assert isinstance(html, str)
    assert "<html>" in html
    assert "<h1>Test Report</h1>" in html
    assert "Performance Drift" in html
    assert "Data Drift" in html
    assert "Prediction Drift" in html


# ---------------------------------------------------------
# 2. HTML contains JSON-like pretty blocks
# ---------------------------------------------------------

def test_report_builder_contains_pretty_json_blocks():
    monitor_out = _fake_monitor_output()
    rb = ReportBuilder(monitor_output=monitor_out)

    html = rb.to_html()

    # Performance block should be inside <pre>
    assert "<pre" in html
    assert "performance_drift" not in html  # pretty JSON only shows values
    assert "auc" in html  # actual metrics appear


# ---------------------------------------------------------
# 3. Save HTML to file
# ---------------------------------------------------------

def test_report_builder_can_save_html(tmp_path):
    monitor_out = _fake_monitor_output()
    rb = ReportBuilder(monitor_output=monitor_out)

    out_path = tmp_path / "report.html"
    rb.save_html(str(out_path))

    assert out_path.exists()
    assert out_path.read_text().startswith("<html>")


# ---------------------------------------------------------
# 4. Plot embedding — base64 check
# ---------------------------------------------------------

def test_report_builder_embeds_plots():
    # Create two simple figures
    fig1 = Figure()
    ax1 = fig1.subplots()
    ax1.plot([0, 1], [0, 1])

    fig2 = Figure()
    ax2 = fig2.subplots()
    ax2.hist([1, 2, 3])

    plots = {
        "figline": fig1,
        "fighist": fig2,
    }

    rb = ReportBuilder(
        monitor_output=_fake_monitor_output(),
        plots=plots,
        title="Report with Plots"
    )

    html = rb.to_html()

    # Should contain base64-encoded PNG image tags
    assert "data:image/png;base64," in html

    # Extract base64 string and verify it can decode
    b64_matches = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    assert len(b64_matches) == 2  # we passed two figures

    # Try decoding the first one
    decoded = base64.b64decode(b64_matches[0])
    assert len(decoded) > 10  # not empty


# ---------------------------------------------------------
# 5. HTML sections remain intact even with no plots
# ---------------------------------------------------------

def test_report_builder_no_plots_still_has_sections():
    rb = ReportBuilder(
        monitor_output=_fake_monitor_output(),
        plots=None,
        title="No Plot Report"
    )

    html = rb.to_html()

    assert "Performance Drift" in html
    assert "Data Drift" in html
    assert "Prediction Drift" in html
    assert "Visualizations" not in html  # plot section should not appear


# ---------------------------------------------------------
# 6. Minimal integration with real ModelMonitor (light E2E)
# ---------------------------------------------------------

def test_report_builder_with_real_monitor_output():
    # Create small synthetic dataset
    ref_df = pd.DataFrame({
        "x1": np.random.rand(50),
        "label": np.random.randint(0, 2, 50),
        "pred": np.random.rand(50),
    })
    cur_df = pd.DataFrame({
        "x1": np.random.rand(50) + 0.1,
        "label": np.random.randint(0, 2, 50),
        "pred": np.random.rand(50),
    })

    monitor = ModelMonitor(
        ref_df=ref_df,
        cur_df=cur_df,
        label_col="label",
        pred_col="pred"
    )

    results = monitor.run_all()

    # Generate a small plot
    fig = plot_distribution(ref_df["pred"], cur_df["pred"])

    rb = ReportBuilder(
        monitor_output=results,
        plots={"pred_dist": fig},
        title="Full E2E Report"
    )

    html = rb.to_html()
    assert "<html>" in html
    assert "Performance Drift" in html
    assert "Data Drift" in html
    assert "Prediction Drift" in html
    assert "data:image/png;base64," in html
