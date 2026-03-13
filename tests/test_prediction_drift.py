
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from momo_ml.monitor.prediction_drift import PredictionDriftDetector


# --------------------------------------------------------
# Helper
# --------------------------------------------------------

def is_finite_dict(d):
    for v in d.values():
        if isinstance(v, dict):
            is_finite_dict(v)
        else:
            assert np.isfinite(v)


# --------------------------------------------------------
# Basic prediction drift test
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

    assert "summary_statistics" in out
    assert "distribution_shift" in out
    assert "decile_shift" in out

    # summary statistics sanity
    stats = out["summary_statistics"]
    assert np.isfinite(stats["mean"]["reference"])
    assert np.isfinite(stats["std"]["current"])

    # distribution shift keys
    dist = out["distribution_shift"]
    assert "l1_distance" in dist
    assert "l2_distance" in dist
    assert dist["l1_distance"] >= 0
    assert dist["l2_distance"] >= 0

    # decile shift shape
    deciles = out["decile_shift"]
    assert len(deciles["ref_deciles"]) == 11
    assert len(deciles["cur_deciles"]) == 11
    assert len(deciles["delta"]) == 11


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
    assert dist["l1_distance"] == 0.0
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
    assert set(out.keys()) == {
        "summary_statistics",
        "distribution_shift",
        "decile_shift"
    }

    # check nested structure contains numbers
    is_finite_dict(out["distribution_shift"])
    is_finite_dict(out["summary_statistics"])