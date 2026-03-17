import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from momo_ml.monitor.prediction_drift import PredictionDriftDetector


# --------------------------------------------------------
# Helper: recursively check that all numeric values are finite
# --------------------------------------------------------
def is_finite_dict(d):
    for v in d.values():
        if isinstance(v, dict):
            is_finite_dict(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, (int, float)):
                    assert np.isfinite(item)
        elif isinstance(v, (int, float)):
            assert np.isfinite(v)
        # ignore other types (str, None, etc.)


# --------------------------------------------------------
# Basic prediction drift test (continuous)
# --------------------------------------------------------
def test_prediction_drift_basic():
    np.random.seed(0)

    ref_df = pd.DataFrame({
        "pred": np.random.rand(500)
    })
    cur_df = pd.DataFrame({
        "pred": np.random.rand(500) * 1.1  # slight shift
    })

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    assert "prediction_type" in out
    assert out["prediction_type"] == "continuous"
    assert "summary_statistics" in out
    assert "distribution_shift" in out
    assert "decile_shift" in out

    # summary statistics sanity
    stats = out["summary_statistics"]
    assert np.isfinite(stats["mean"]["reference"])
    assert np.isfinite(stats["std"]["current"])
    assert "q25" in stats
    assert "q50" in stats
    assert "q75" in stats

    # distribution shift keys (default: psi, l1, l2)
    dist = out["distribution_shift"]
    assert "l1_distance" in dist
    assert "l2_distance" in dist
    assert "psi" in dist
    assert dist["l1_distance"] >= 0
    assert dist["l2_distance"] >= 0
    assert np.isfinite(dist["psi"])

    # decile shift shape
    dec = out["decile_shift"]
    assert "quantiles" in dec
    assert "ref_values" in dec
    assert "cur_values" in dec
    assert "delta" in dec
    assert len(dec["ref_values"]) == 11
    assert len(dec["cur_values"]) == 11
    assert len(dec["delta"]) == 11
    assert_allclose(dec["quantiles"], np.linspace(0, 1, 11))


# --------------------------------------------------------
# Identical distribution → zero shift
# --------------------------------------------------------
def test_prediction_drift_identical():
    np.random.seed(42)

    preds = np.random.rand(400)
    ref_df = pd.DataFrame({"pred": preds})
    cur_df = pd.DataFrame({"pred": preds.copy()})

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    # L1 and L2 distances should be extremely small
    dist = out["distribution_shift"]
    assert dist["l1_distance"] < 1e-6
    assert dist["l2_distance"] < 1e-6
    assert dist["psi"] < 1e-6  # PSI should be near zero

    dec = out["decile_shift"]
    assert_allclose(dec["delta"], np.zeros(11), rtol=1e-6, atol=1e-6)


# --------------------------------------------------------
# Edge case: preds contain identical values
# --------------------------------------------------------
def test_prediction_constant_values():
    ref_df = pd.DataFrame({"pred": np.ones(200)})
    cur_df = pd.DataFrame({"pred": np.ones(200)})

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    # should not crash; distribution shift should be 0
    dist = out["distribution_shift"]
    assert "psi" in dist
    assert dist["psi"] == 0.0
    # l1/l2 may not be present if treated as categorical; check if present
    if "l1_distance" in dist:
        assert dist["l1_distance"] == 0.0
    if "l2_distance" in dist:
        assert dist["l2_distance"] == 0.0


# --------------------------------------------------------
# Edge case: cur shifted mean
# --------------------------------------------------------
def test_prediction_shifted_mean():
    np.random.seed(1)

    ref_df = pd.DataFrame({"pred": np.random.rand(500)})
    cur_df = pd.DataFrame({"pred": np.random.rand(500) + 0.2})  # shift mean

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    stats = out["summary_statistics"]
    # mean should change
    assert stats["mean"]["delta"] > 0.15

    # distribution shift positive
    assert out["distribution_shift"]["l1_distance"] > 0
    assert out["distribution_shift"]["psi"] > 0


# --------------------------------------------------------
# Error-handling: missing prediction column
# --------------------------------------------------------
def test_prediction_missing_column():
    ref_df = pd.DataFrame({"pred": np.random.rand(100)})
    cur_df = pd.DataFrame({"wrongcol": np.random.rand(100)})

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    assert "error" in out


# --------------------------------------------------------
# Error-handling: No valid prediction values
# --------------------------------------------------------
def test_prediction_empty_values():
    ref_df = pd.DataFrame({"pred": [np.nan] * 20})
    cur_df = pd.DataFrame({"pred": [np.nan] * 20})

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    assert "error" in out


# --------------------------------------------------------
# Structure test
# --------------------------------------------------------
def test_prediction_drift_structure():
    np.random.seed(123)

    ref_df = pd.DataFrame({"pred": np.random.rand(100)})
    cur_df = pd.DataFrame({"pred": np.random.rand(100)})

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    assert isinstance(out, dict)
    expected_keys = {"prediction_type", "summary_statistics", "distribution_shift", "decile_shift"}
    assert set(out.keys()) == expected_keys

    # check nested structure contains numbers (skip lists in decile_shift for finiteness check)
    is_finite_dict(out["distribution_shift"])
    is_finite_dict(out["summary_statistics"])
    # decile_shift contains lists of floats – check they are finite
    for v in out["decile_shift"].values():
        if isinstance(v, list):
            for x in v:
                if isinstance(x, float):
                    assert np.isfinite(x)


# --------------------------------------------------------
# Optional metrics: KS, KL, JS
# --------------------------------------------------------
def test_prediction_drift_optional_metrics():
    np.random.seed(42)
    ref_df = pd.DataFrame({"pred": np.random.rand(500)})
    cur_df = pd.DataFrame({"pred": np.random.rand(500) + 0.1})

    det = PredictionDriftDetector(
        ref_df, cur_df, pred_col="pred",
        include_ks=True,
        include_kl=True,
        include_js=True
    )
    out = det.compute()

    dist = out["distribution_shift"]
    assert "ks" in dist
    assert "kl" in dist
    assert "js" in dist
    assert np.isfinite(dist["ks"]["statistic"])
    # pvalue may be None if scipy not installed; we don't assert its presence
    assert np.isfinite(dist["kl"])
    assert np.isfinite(dist["js"])


# --------------------------------------------------------
# Categorical prediction tests
# --------------------------------------------------------
def test_categorical_prediction():
    # Discrete class labels (strings)
    ref_df = pd.DataFrame({"pred": ["A", "B", "A", "C", "B"] * 100})
    cur_df = pd.DataFrame({"pred": ["A", "B", "B", "C", "A"] * 100})  # slightly different proportions

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    assert out["prediction_type"] == "categorical"
    assert "summary_statistics" in out
    assert "distribution_shift" in out
    assert "decile_shift" not in out  # no decile for categorical

    # summary stats: should return proportions
    stats = out["summary_statistics"]
    assert "categories" in stats
    assert "reference_proportions" in stats
    assert "current_proportions" in stats
    assert "delta_proportions" in stats
    assert set(stats["categories"]) == {"A", "B", "C"}

    # distribution shift: should have psi (and possibly others if enabled)
    dist = out["distribution_shift"]
    assert "psi" in dist
    assert np.isfinite(dist["psi"])
    # KS not present for categorical
    assert "ks" not in dist

    # No histogram distances (since data is categorical, not numeric)
    assert "l1_distance" not in dist
    assert "l2_distance" not in dist


def test_categorical_with_kl_js():
    ref_df = pd.DataFrame({"pred": ["A", "B", "A", "C"] * 50})
    cur_df = pd.DataFrame({"pred": ["A", "B", "B", "C"] * 50})

    det = PredictionDriftDetector(
        ref_df, cur_df, pred_col="pred",
        include_kl=True,
        include_js=True
    )
    out = det.compute()

    dist = out["distribution_shift"]
    assert "kl" in dist
    assert "js" in dist
    assert np.isfinite(dist["kl"])
    assert np.isfinite(dist["js"])


# --------------------------------------------------------
# Edge case: mixed types (should be handled as categorical)
# --------------------------------------------------------
def test_mixed_type_prediction():
    # One dataset has numbers, other has strings – should be treated as categorical
    ref_df = pd.DataFrame({"pred": [1, 2, 1, 3] * 50})
    cur_df = pd.DataFrame({"pred": ["1", "2", "1", "3"] * 50})  # strings

    det = PredictionDriftDetector(ref_df, cur_df, pred_col="pred")
    out = det.compute()

    assert out["prediction_type"] == "categorical"
    # PSI should still compute (psi.py handles mixed types via object conversion)
    assert "psi" in out["distribution_shift"]
    assert np.isfinite(out["distribution_shift"]["psi"])