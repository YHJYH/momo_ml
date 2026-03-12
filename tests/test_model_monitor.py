
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from momo_ml.monitor import ModelMonitor


# ------------------------------------------------------------
# 1. Basic E2E test: numeric + categorical + predictions
# ------------------------------------------------------------

def test_model_monitor_basic_e2e():
    np.random.seed(42)

    # Reference dataset (baseline)
    ref_df = pd.DataFrame({
        "x1": np.random.normal(0, 1, 500),
        "x2": np.random.normal(5, 2, 500),
        "cat": np.random.choice(["A", "B", "C"], size=500, p=[0.3, 0.5, 0.2]),
    })
    ref_df["label"] = (ref_df["x1"] + np.random.normal(0, 1, 500) > 0).astype(int)
    ref_df["pred"] = ref_df["label"] * 0.7 + np.random.normal(0.2, 0.1, 500)
    ref_df["pred"] = ref_df["pred"].clip(0, 1)

    # Current dataset (shifted)
    cur_df = pd.DataFrame({
        "x1": np.random.normal(0.3, 1.2, 500),
        "x2": np.random.normal(4.7, 2.1, 500),
        "cat": np.random.choice(["A", "B", "C"], size=500, p=[0.1, 0.7, 0.2]),
    })
    cur_df["label"] = (cur_df["x1"] + np.random.normal(0, 1, 500) > 0.1).astype(int)
    cur_df["pred"] = cur_df["label"] * 0.6 + np.random.normal(0.25, 0.15, 500)
    cur_df["pred"] = cur_df["pred"].clip(0, 1)

    monitor = ModelMonitor(
        ref_df=ref_df,
        cur_df=cur_df,
        label_col="label",
        pred_col="pred"
    )

    out = monitor.run_all()

    # Check top-level keys
    assert set(out.keys()) == {
        "performance_drift",
        "data_drift",
        "prediction_drift"
    }

    # Performance drift
    perf = out["performance_drift"]
    assert "reference" in perf and "current" in perf

    # Data drift
    dd = out["data_drift"]
    assert "numeric_features" in dd
    assert "categorical_features" in dd

    # Prediction drift
    pd_out = out["prediction_drift"]
    assert "summary_statistics" in pd_out
    assert "distribution_shift" in pd_out
    assert "decile_shift" in pd_out


# ------------------------------------------------------------
# 2. Missing prediction column → graceful error
# ------------------------------------------------------------

def test_model_monitor_missing_pred_column():
    ref_df = pd.DataFrame({
        "label": np.random.randint(0, 2, 100),
        "pred": np.random.rand(100),
        "x": np.random.rand(100)
    })
    cur_df = pd.DataFrame({
        "label": np.random.randint(0, 2, 100),
        "wrongcol": np.random.rand(100),
        "x": np.random.rand(100)
    })

    monitor = ModelMonitor(
        ref_df=ref_df,
        cur_df=cur_df,
        label_col="label",
        pred_col="pred"
    )
    out = monitor.run_all()

    # The prediction part should return an error message
    assert "error" in out["prediction_drift"]


# ------------------------------------------------------------
# 3. Missing label column → performance drift error
# ------------------------------------------------------------

def test_model_monitor_missing_label_column():
    ref_df = pd.DataFrame({
        "pred": np.random.rand(100),
        "x": np.random.rand(100)
    })
    cur_df = pd.DataFrame({
        "pred": np.random.rand(100),
        "x": np.random.rand(100)
    })

    monitor = ModelMonitor(
        ref_df=ref_df,
        cur_df=cur_df,
        label_col="label",
        pred_col="pred"
    )
    out = monitor.run_all()

    assert "error" in out["performance_drift"]


# ------------------------------------------------------------
# 4. E2E: identical distributions → drift near zero
# ------------------------------------------------------------

def test_model_monitor_identical_distributions():
    np.random.seed(123)
    ref_df = pd.DataFrame({
        "x1": np.random.normal(0, 1, 300),
        "cat": np.random.choice(["A", "B"], 300),
    })
    ref_df["label"] = (ref_df["x1"] > 0).astype(int)
    ref_df["pred"] = ref_df["label"] * 0.6 + np.random.normal(0.2, 0.1, 300)
    ref_df["pred"] = ref_df["pred"].clip(0, 1)

    cur_df = ref_df.copy()

    monitor = ModelMonitor(
        ref_df=ref_df,
        cur_df=cur_df,
        label_col="label",
        pred_col="pred"
    )

    out = monitor.run_all()

    # Performance drift deltas ~0
    deltas = out["performance_drift"]["delta"]
    for v in deltas.values():
        assert abs(v) < 1e-6

    # Numeric PSI ~0
    for f, v in out["data_drift"]["numeric_features"].items():
        assert v["psi"] < 1e-6

    # Categorical PSI ~0
    for f, v in out["data_drift"]["categorical_features"].items():
        assert v["psi"] < 1e-6

    # Prediction distribution shift ~0
    dist = out["prediction_drift"]["distribution_shift"]
    assert dist["l1_distance"] < 1e-6
    assert dist["l2_distance"] < 1e-6

    dec = out["prediction_drift"]["decile_shift"]
    assert_allclose(dec["delta"], np.zeros(11), atol=1e-6)


# ------------------------------------------------------------
# 5. E2E sanity check: output types are correct
# ------------------------------------------------------------

def test_model_monitor_output_types():
    np.random.seed(42)

    ref_df = pd.DataFrame({
        "x": np.random.rand(100),
        "label": np.random.randint(0, 2, 100),
        "pred": np.random.rand(100),
    })
    cur_df = pd.DataFrame({
        "x": np.random.rand(100),
        "label": np.random.randint(0, 2, 100),
        "pred": np.random.rand(100),
    })

    monitor = ModelMonitor(ref_df, cur_df, label_col="label", pred_col="pred")
    out = monitor.run_all()

    assert isinstance(out, dict)
    assert isinstance(out["performance_drift"], dict)
    assert isinstance(out["data_drift"], dict)
    assert isinstance(out["prediction_drift"], dict)