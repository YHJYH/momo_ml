
import pytest
import math
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from momo_ml.monitor.data_drift import DataDriftDetector
from momo_ml.metrics.js import compute_js


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def assert_is_finite_number(x):
    assert isinstance(x, float)
    assert np.isfinite(x)


# --------------------------------------------------------
# Numeric Drift Tests
# --------------------------------------------------------

def test_numeric_drift_basic():
    np.random.seed(0)

    ref_df = pd.DataFrame({
        "x1": np.random.normal(0, 1, 500),
        "x2": np.random.normal(5, 1, 500),
    })

    cur_df = pd.DataFrame({
        "x1": np.random.normal(0.2, 1.2, 500),
        "x2": np.random.normal(4.8, 1.1, 500),
    })

    det = DataDriftDetector(ref_df, cur_df)
    out = det.compute_numeric_drift()

    # Both features should appear
    assert "x1" in out
    assert "x2" in out
    assert "psi" in out["x1"]
    assert "kl" in out["x1"]

    # PSI values should be finite floats
    assert_is_finite_number(out["x1"]["psi"])
    assert_is_finite_number(out["x2"]["psi"])
    assert_is_finite_number(out["x1"]["kl"])
    assert_is_finite_number(out["x2"]["kl"])

    # numeric_features should be detected correctly
    assert set(det.numeric_features) == {"x1", "x2"}


# --------------------------------------------------------
# Categorical Drift Tests
# --------------------------------------------------------

def test_categorical_drift_basic():
    np.random.seed(0)

    ref_df = pd.DataFrame({
        "cat": np.random.choice(["A", "B", "C"], size=300, p=[0.3, 0.5, 0.2]),
    })
    cur_df = pd.DataFrame({
        "cat": np.random.choice(["A", "B", "C"], size=300, p=[0.1, 0.7, 0.2]),
    })

    det = DataDriftDetector(ref_df, cur_df)
    out = det.compute_categorical_drift()

    assert "cat" in out
    assert "psi" in out["cat"]
    assert_is_finite_number(out["cat"]["psi"])
    assert "kl" in out["cat"]
    assert_is_finite_number(out["cat"]["kl"])

    # categorical_features detected correctly
    assert det.categorical_features == ["cat"]

def test_categorical_kl_basic():
    ref_df = pd.DataFrame({"cat": ["A", "B", "A", "C"] * 25})
    cur_df = pd.DataFrame({"cat": ["A", "B", "B", "C"] * 25})

    det = DataDriftDetector(ref_df, cur_df)
    out = det.compute_categorical_drift()

    assert "cat" in out
    assert "psi" in out["cat"]
    assert "kl" in out["cat"]
    assert_is_finite_number(out["cat"]["psi"])
    assert_is_finite_number(out["cat"]["kl"])


# --------------------------------------------------------
# Mixed type test (numeric + categorical)
# --------------------------------------------------------

def test_mixed_numeric_and_categorical():
    np.random.seed(42)

    ref_df = pd.DataFrame({
        "num": np.random.normal(0, 1, 400),
        "cat": np.random.choice(["X", "Y"], size=400, p=[0.6, 0.4]),
    })
    cur_df = pd.DataFrame({
        "num": np.random.normal(0.3, 1.1, 400),
        "cat": np.random.choice(["X", "Y"], size=400, p=[0.2, 0.8]),
    })

    det = DataDriftDetector(ref_df, cur_df)
    out = det.compute()

    assert set(out.keys()) == {"numeric_features", "categorical_features", "incompatible_features"}

    # Check numeric part
    num_out = out["numeric_features"]
    assert "num" in num_out
    assert_is_finite_number(num_out["num"]["psi"])
    assert_is_finite_number(num_out["num"]["kl"])

    # Check categorical part
    cat_out = out["categorical_features"]
    assert "cat" in cat_out
    assert_is_finite_number(cat_out["cat"]["psi"])
    assert_is_finite_number(cat_out["cat"]["kl"])

    # Check internal lists
    assert det.numeric_features == ["num"]
    assert det.categorical_features == ["cat"]

def test_incompatible_feature_handling():
    ref_df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "cat": ["a", "b", "c"]
    })
    cur_df = pd.DataFrame({
        "num": ["1", "2", "3"],  # strings instead of floats → incompatible
        "cat": ["a", "b", "c"]    # both are object type → categorical feature
    })

    det = DataDriftDetector(ref_df, cur_df, features=["num", "cat"])

    assert det.numeric_features == []           # no numeric features
    assert det.categorical_features == ["cat"]  # cat is categorical
    assert det.incompatible_features == ["num"] # num is incompatible

    out = det.compute()
    assert "num" not in out["numeric_features"]
    assert "num" not in out["categorical_features"]
    assert "cat" in out["categorical_features"]
    assert out["incompatible_features"] == ["num"]


# --------------------------------------------------------
# Edge cases: identical distributions → PSI = 0 (or close)
# --------------------------------------------------------

def test_identical_numeric_distributions():
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 300)})
    cur = ref.copy()

    det = DataDriftDetector(ref, cur)
    out = det.compute_numeric_drift()

    psi_val = out["x"]["psi"]
    assert psi_val < 1e-6  # should be near zero
    kl_val = out["x"]["kl"]
    assert kl_val < 1e-6  # should also be near zero

def test_identical_categorical_distributions():
    ref = pd.DataFrame({"cat": ["A", "B", "A", "C"] * 50})
    cur = ref.copy()

    det = DataDriftDetector(ref, cur)
    out = det.compute_categorical_drift()

    psi_val = out["cat"]["psi"]
    assert psi_val < 1e-6  # PSI ~ 0
    kl_val = out["cat"]["kl"]
    assert kl_val < 1e-6  # KL ~ 0

# --------------------------------------------------------
# KL parameters passing
# --------------------------------------------------------

def test_kl_parameters():
    ref_df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
    cur_df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})

    det = DataDriftDetector(
        ref_df, cur_df,
        kl_buckets=5,
        kl_base="2",
        kl_epsilon=1e-10,
        kl_handle_outside="clip"
    )

    assert det.kl_buckets == 5
    assert det.kl_base == "2"
    assert det.kl_epsilon == 1e-10
    assert det.kl_handle_outside == "clip"

    out = det.compute_numeric_drift()
    assert "kl" in out["x"]
    assert_is_finite_number(out["x"]["kl"])

# --------------------------------------------------------
# KS-specific tests
# --------------------------------------------------------

def _assert_ks_result_struct(ks_obj):
    assert isinstance(ks_obj, dict)
    assert set(ks_obj.keys()) == {"statistic", "pvalue", "n_ref", "n_cur"}
    assert isinstance(ks_obj["statistic"], float) or (np.isnan(ks_obj["statistic"]))
    # pvalue be float or None
    assert (ks_obj["pvalue"] is None) or isinstance(ks_obj["pvalue"], float)
    assert isinstance(ks_obj["n_ref"], int)
    assert isinstance(ks_obj["n_cur"], int)
    assert ks_obj["n_ref"] >= 0
    assert ks_obj["n_cur"] >= 0

def test_numeric_drift_contains_ks_and_is_finite():
    np.random.seed(123)
    ref_df = pd.DataFrame({
        "x": np.random.normal(0, 1, 400),
        "y": np.random.normal(5, 2, 400),
    })
    cur_df = pd.DataFrame({
        "x": np.random.normal(0.2, 1.3, 400),
        "y": np.random.normal(5.5, 2.2, 400),
    })
    det = DataDriftDetector(ref_df, cur_df)
    out = det.compute_numeric_drift()

    assert "ks" in out["x"]
    assert "ks" in out["y"]

    _assert_ks_result_struct(out["x"]["ks"])
    _assert_ks_result_struct(out["y"]["ks"])

    assert np.isfinite(out["x"]["ks"]["statistic"])
    assert np.isfinite(out["y"]["ks"]["statistic"])

def test_ks_statistic_increases_with_stronger_shift():
    np.random.seed(7)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 600)})
    # light drift
    cur_small = pd.DataFrame({"x": np.random.normal(0.2, 1.0, 600)})
    # heavy drift
    cur_large = pd.DataFrame({"x": np.random.normal(1.0, 1.0, 600)})

    det_small = DataDriftDetector(ref, cur_small)
    det_large = DataDriftDetector(ref, cur_large)

    ks_small = det_small.compute_numeric_drift()["x"]["ks"]["statistic"]
    ks_large = det_large.compute_numeric_drift()["x"]["ks"]["statistic"]

    assert np.isfinite(ks_small) and np.isfinite(ks_large)
    assert ks_large > ks_small

def test_ks_identical_distributions_near_zero():
    np.random.seed(21)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
    cur = ref.copy()

    det = DataDriftDetector(ref, cur)
    ks = det.compute_numeric_drift()["x"]["ks"]["statistic"]

    # same distribution should yield KS statistic near zero
    assert ks < 0.05

def test_ks_not_present_for_categorical_features():
    np.random.seed(0)
    ref = pd.DataFrame({"cat": np.random.choice(["A", "B", "C"], size=300)})
    cur = pd.DataFrame({"cat": np.random.choice(["A", "B", "C"], size=300)})

    det = DataDriftDetector(ref, cur)
    cat_out = det.compute_categorical_drift()

    assert "cat" in cat_out
    # no ks in categorical
    assert "ks" not in cat_out["cat"]

def test_ks_handles_nans_and_empty_after_dropna():
    # scene 1: one side has all NaN → KS statistic should be NaN and p-value None
    ref = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    cur = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    det = DataDriftDetector(ref, cur)
    ks_obj = det.compute_numeric_drift()["x"]["ks"]
    _assert_ks_result_struct(ks_obj)
    assert np.isnan(ks_obj["statistic"])
    assert ks_obj["pvalue"] is None
    assert ks_obj["n_ref"] == 0 or ks_obj["n_cur"] == 0

    # scene 2: both sides have some NaN but also some valid data → KS should compute on valid data only
    ref2 = pd.DataFrame({"x": [np.nan, 0.0, 1.0, 2.0, np.nan]})
    cur2 = pd.DataFrame({"x": [np.nan, 0.5, 1.5, np.nan, 2.5]})

    det2 = DataDriftDetector(ref2, cur2)
    ks_obj2 = det2.compute_numeric_drift()["x"]["ks"]
    _assert_ks_result_struct(ks_obj2)
    assert ks_obj2["n_ref"] > 0 and ks_obj2["n_cur"] > 0
    assert np.isfinite(ks_obj2["statistic"])

def test_compute_structure_includes_ks_for_numeric():
    np.random.seed(1234)
    ref = pd.DataFrame({
        "x": np.random.normal(0, 1, 80),
        "cat": np.random.choice(["A", "B"], size=80),
    })
    cur = pd.DataFrame({
        "x": np.random.normal(0.3, 1.2, 80),
        "cat": np.random.choice(["A", "B"], size=80),
    })

    det = DataDriftDetector(ref, cur)
    out = det.compute()

    assert "x" in out["numeric_features"]
    assert "ks" in out["numeric_features"]["x"]
    _assert_ks_result_struct(out["numeric_features"]["x"]["ks"])

    assert "cat" in out["categorical_features"]
    assert "ks" not in out["categorical_features"]["cat"]

# --------------------------------------------------------
# JS divergence tests (standalone + optional integration)
# --------------------------------------------------------
# ---------- Standalone compute_js tests ----------

def test_js_numeric_basic():
    np.random.seed(123)
    ref = np.random.normal(0, 1, 500)
    cur = np.random.normal(0.2, 1.1, 500)
    val = compute_js(ref, cur, buckets=10, base="2", epsilon=1e-12, handle_outside="clip")
    assert_is_finite_number(val)
    # JS >= 0
    assert val >= 0.0

def test_js_categorical_basic():
    ref = ["A", "B", "A", "C"] * 50
    cur = ["A", "B", "B", "C"] * 50
    val = compute_js(ref, cur, base="2")
    assert_is_finite_number(val)
    assert val >= 0.0

def test_js_identical_distributions_near_zero_numeric():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 400)
    cur = ref.copy()
    # base=2 JS ∈ [0, 1]
    val = compute_js(ref, cur, base="2")
    assert val < 1e-6

def test_js_identical_distributions_near_zero_categorical():
    ref = ["A", "B", "A", "C"] * 40
    cur = ref.copy()
    val = compute_js(ref, cur, base="2")
    assert val < 1e-6

def test_js_strength_monotonicity_numeric():
    np.random.seed(7)
    ref = np.random.normal(0, 1, 600)
    cur_small = np.random.normal(0.2, 1.0, 600)
    cur_large = np.random.normal(1.0, 1.0, 600)
    js_small = compute_js(ref, cur_small, base="2")
    js_large = compute_js(ref, cur_large, base="2")
    assert js_small >= 0 and js_large >= 0
    assert js_large > js_small  # high drift high js

def test_js_parameters_passing_and_bases():
    np.random.seed(99)
    ref = np.random.normal(0, 1, 300)
    cur = np.random.normal(0.1, 1.2, 300)

    # all non-negative
    val_e = compute_js(ref, cur, base="e", buckets=8, epsilon=1e-10, handle_outside="clip")
    val_2 = compute_js(ref, cur, base="2", buckets=8, epsilon=1e-10, handle_outside="clip")
    val_10 = compute_js(ref, cur, base="10", buckets=8, epsilon=1e-10, handle_outside="clip")

    assert_is_finite_number(val_e)
    assert_is_finite_number(val_2)
    assert_is_finite_number(val_10)
    assert val_e >= 0 and val_2 >= 0 and val_10 >= 0

def test_js_handles_nans_and_empty_after_dropna():
    # one side has all NaN
    ref = [np.nan, np.nan, np.nan]
    cur = [1.0, 2.0, 3.0]
    val = compute_js(ref, cur)
    assert math.isnan(val)

    # both sides have nan but still valid sample
    ref2 = [np.nan, 0.0, 1.0, np.nan, 2.0]
    cur2 = [np.nan, 0.5, 1.5, 2.5, np.nan]
    val2 = compute_js(ref2, cur2)
    assert_is_finite_number(val2)

def _has_js_in_detector(det: DataDriftDetector, out: dict, feature: str, is_numeric: bool) -> bool:
    try:
        if is_numeric:
            return "js" in out["numeric_features"].get(feature, {})
        else:
            return "js" in out["categorical_features"].get(feature, {})
    except Exception:
        return False

@pytest.mark.parametrize("base", ["e", "2"])
def test_detector_js_numeric_optional(base):
    np.random.seed(1234)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 200)})
    cur = pd.DataFrame({"x": np.random.normal(0.3, 1.1, 200)})

    det = DataDriftDetector(ref, cur, kl_buckets=6, kl_base=base, kl_epsilon=1e-10, kl_handle_outside="clip")
    out = det.compute()

    has_js = _has_js_in_detector(det, out, "x", is_numeric=True)
    if not has_js:
        pytest.skip("JS is not integrated in DataDriftDetector numeric outputs; skipping.")
    else:
        val = out["numeric_features"]["x"]["js"]
        assert_is_finite_number(val)
        assert val >= 0.0

def test_detector_js_categorical_optional():
    ref = pd.DataFrame({"cat": ["A", "B", "A", "C"] * 40})
    cur = pd.DataFrame({"cat": ["A", "B", "B", "C"] * 40})

    det = DataDriftDetector(ref, cur)
    out = det.compute()

    has_js = _has_js_in_detector(det, out, "cat", is_numeric=False)
    if not has_js:
        pytest.skip("JS is not integrated in DataDriftDetector categorical outputs; skipping.")
    else:
        val = out["categorical_features"]["cat"]["js"]
        assert_is_finite_number(val)
        assert val >= 0.0

def test_detector_js_identical_near_zero_optional():
    np.random.seed(2024)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 300), "cat": np.random.choice(["A","B"], size=300)})
    cur = ref.copy()

    det = DataDriftDetector(ref, cur, kl_base="2")
    out = det.compute()

    # numerical JS
    has_js_num = _has_js_in_detector(det, out, "x", is_numeric=True)
    if has_js_num:
        js_num = out["numeric_features"]["x"]["js"]
        assert js_num < 1e-6
    # categorical JS
    has_js_cat = _has_js_in_detector(det, out, "cat", is_numeric=False)
    if has_js_cat:
        js_cat = out["categorical_features"]["cat"]["js"]
        assert js_cat < 1e-6
    if not (has_js_num or has_js_cat):
        pytest.skip("JS not integrated in DataDriftDetector outputs; skipping.")

# --------------------------------------------------------
# Wasserstein Distance (WD) tests
# --------------------------------------------------------

def test_wd_numeric_basic():
    np.random.seed(123)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": np.random.normal(0.2, 1.2, 500)})

    det = DataDriftDetector(ref, cur)
    out = det.compute_numeric_drift()

    assert "x" in out
    assert "wd" in out["x"]
    val = out["x"]["wd"]
    assert_is_finite_number(val)
    assert val >= 0.0


def test_wd_categorical_basic():
    ref = pd.DataFrame({"cat": ["A", "B", "A", "C"] * 40})
    cur = pd.DataFrame({"cat": ["A", "B", "B", "C"] * 40})

    det = DataDriftDetector(ref, cur)
    out = det.compute_categorical_drift()

    assert "cat" in out
    assert "wd" in out["cat"]
    val = out["cat"]["wd"]
    assert_is_finite_number(val)
    assert val >= 0.0


def test_wd_identical_numeric_near_zero():
    np.random.seed(42)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 400)})
    cur = ref.copy()

    det = DataDriftDetector(ref, cur)
    wd_val = det.compute_numeric_drift()["x"]["wd"]

    assert wd_val < 1e-6  # identical distributions → WD ≈ 0


def test_wd_identical_categorical_near_zero():
    ref = pd.DataFrame({"cat": ["A", "B", "A", "C"] * 50})
    cur = ref.copy()

    det = DataDriftDetector(ref, cur)
    wd_val = det.compute_categorical_drift()["cat"]["wd"]

    assert wd_val < 1e-6


def test_wd_strength_monotonicity_numeric():
    np.random.seed(777)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 600)})
    cur_small = pd.DataFrame({"x": np.random.normal(0.2, 1, 600)})
    cur_large = pd.DataFrame({"x": np.random.normal(1.2, 1, 600)})

    det_small = DataDriftDetector(ref, cur_small)
    det_large = DataDriftDetector(ref, cur_large)

    wd_small = det_small.compute_numeric_drift()["x"]["wd"]
    wd_large = det_large.compute_numeric_drift()["x"]["wd"]

    assert wd_small >= 0 and wd_large >= 0
    assert wd_large > wd_small  # stronger shift → WD larger


def test_wd_handles_nans_and_empty():
    ref = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    cur = pd.DataFrame({"x": [1, 2, 3]})

    det = DataDriftDetector(ref, cur)
    wd_val = det.compute_numeric_drift()["x"]["wd"]

    assert np.isnan(wd_val)

    # partially NaN but with valid samples
    ref2 = pd.DataFrame({"x": [np.nan, 0.0, 1.0, np.nan]})
    cur2 = pd.DataFrame({"x": [np.nan, 1.0, 2.0, np.nan]})

    det2 = DataDriftDetector(ref2, cur2)
    wd_val2 = det2.compute_numeric_drift()["x"]["wd"]

    assert_is_finite_number(wd_val2)


def test_compute_structure_includes_wd():
    np.random.seed(1234)
    ref = pd.DataFrame({
        "x": np.random.normal(0, 1, 100),
        "cat": np.random.choice(["A", "B"], size=100)
    })
    cur = pd.DataFrame({
        "x": np.random.normal(0.4, 1.3, 100),
        "cat": np.random.choice(["A", "B"], size=100)
    })

    det = DataDriftDetector(ref, cur)
    out = det.compute()

    # numeric includes wd
    assert "wd" in out["numeric_features"]["x"]
    assert_is_finite_number(out["numeric_features"]["x"]["wd"])

    # categorical includes wd
    assert "wd" in out["categorical_features"]["cat"]
    assert_is_finite_number(out["categorical_features"]["cat"]["wd"])

# --------------------------------------------------------
# Robustness: categorical column containing mixed types
# --------------------------------------------------------

def test_mixed_type_categorical_is_handled():
    ref = pd.DataFrame({"cat": ["A", "B", 1, 2, "C"]})
    cur = pd.DataFrame({"cat": ["B", 2, "C", 1, "A"]})

    det = DataDriftDetector(ref, cur)
    out = det.compute_categorical_drift()

    psi_val = out["cat"]["psi"]
    assert_is_finite_number(psi_val)
    kl_val = out["cat"]["kl"]
    assert_is_finite_number(kl_val)


# --------------------------------------------------------
# Check overall structure of compute()
# --------------------------------------------------------

def test_compute_structure():
    ref = pd.DataFrame({
        "x": np.random.normal(0, 1, 50),
        "cat": np.random.choice(["A", "B"], size=50)
    })
    cur = pd.DataFrame({
        "x": np.random.normal(0.1, 1.2, 50),
        "cat": np.random.choice(["A", "B"], size=50)
    })

    det = DataDriftDetector(ref, cur)
    out = det.compute()

    assert "numeric_features" in out
    assert "categorical_features" in out

    # numeric part structure
    assert isinstance(out["numeric_features"], dict)
    for k, v in out["numeric_features"].items():
        assert "psi" in v
        assert "kl" in v

    # categorical part structure
    assert isinstance(out["categorical_features"], dict)
    for k, v in out["categorical_features"].items():
        assert "psi" in v
        assert "kl" in v

def test_empty_features_warning():
    ref_df = pd.DataFrame({"a": [1, 2, 3]})
    cur_df = pd.DataFrame({"b": [4, 5, 6]})  # no common features with ref_df

    with pytest.warns(UserWarning, match="No common features found"):
        det = DataDriftDetector(ref_df, cur_df)

    out = det.compute()
    assert out["numeric_features"] == {}
    assert out["categorical_features"] == {}
    assert out["incompatible_features"] == []

def test_compute_structure():
    ref = pd.DataFrame({
        "x": np.random.normal(0, 1, 50),
        "cat": np.random.choice(["A", "B"], size=50)
    })
    cur = pd.DataFrame({
        "x": np.random.normal(0.1, 1.2, 50),
        "cat": np.random.choice(["A", "B"], size=50)
    })

    det = DataDriftDetector(ref, cur)
    out = det.compute()

    assert set(out.keys()) == {"numeric_features", "categorical_features", "incompatible_features"}
