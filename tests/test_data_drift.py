
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from momo_ml.monitor.data_drift import DataDriftDetector
from momo_ml.metrics.psi import compute_psi


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

    # PSI values should be finite floats
    assert_is_finite_number(out["x1"]["psi"])
    assert_is_finite_number(out["x2"]["psi"])

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
    psi_val = out["cat"]["psi"]
    assert_is_finite_number(psi_val)

    # categorical_features detected correctly
    assert det.categorical_features == ["cat"]


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

    assert set(out.keys()) == {"numeric_features", "categorical_features"}

    # Check numeric part
    num_out = out["numeric_features"]
    assert "num" in num_out
    assert_is_finite_number(num_out["num"]["psi"])

    # Check categorical part
    cat_out = out["categorical_features"]
    assert "cat" in cat_out
    assert_is_finite_number(cat_out["cat"]["psi"])

    # Check internal lists
    assert det.numeric_features == ["num"]
    assert det.categorical_features == ["cat"]


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


def test_identical_categorical_distributions():
    ref = pd.DataFrame({"cat": ["A", "B", "A", "C"] * 50})
    cur = ref.copy()

    det = DataDriftDetector(ref, cur)
    out = det.compute_categorical_drift()

    psi_val = out["cat"]["psi"]
    assert psi_val < 1e-6  # PSI ~ 0


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

    # categorical part structure
    assert isinstance(out["categorical_features"], dict)
    for k, v in out["categorical_features"].items():
        assert "psi" in v