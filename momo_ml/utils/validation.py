from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
import pandas as pd

# ============================================================
# Exceptions & Data Structures
# ============================================================


class ValidationError(ValueError):
    """Raised when hard validation fails (non-recoverable)."""


@dataclass
class ValidationReport:
    """Container for validation outcomes."""

    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, key: str, value: Any) -> None:
        self.info[key] = value

    def raise_if_error(self) -> None:
        if not self.ok:
            raise ValidationError("\n".join(self.errors))


# ============================================================
# Utilities
# ============================================================

_NUMERIC_KINDS = ("i", "u", "f")  # int, unsigned, float


def is_numeric_series(s: pd.Series) -> bool:
    try:
        return np.asarray(s.dropna()).dtype.kind in _NUMERIC_KINDS
    except Exception:
        return False


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(max(denominator, 1))


# ============================================================
# Core Validators (single DataFrame)
# ============================================================


def validate_dataframe_basic(
    df: pd.DataFrame, name: str, report: ValidationReport
) -> None:
    """Basic sanity checks for a dataframe."""
    if not isinstance(df, pd.DataFrame):
        report.add_error(f"[{name}] must be a pandas.DataFrame, got: {type(df)}")
        return

    if df.shape[0] == 0:
        report.add_error(f"[{name}] has 0 rows.")
    if df.shape[1] == 0:
        report.add_error(f"[{name}] has 0 columns.")

    # Duplicate columns
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        report.add_error(f"[{name}] has duplicated columns: {dup_cols}")

    # Index monotonicity not required, but warn for bad practice
    if not df.index.is_unique:
        report.add_warning(
            f"[{name}] index is not unique; downstream merges may behave unexpectedly."
        )

    # Mixed types in single column can be problematic; flag common cases
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        # mixture of numbers and objects often signals dirty data
        if sample.map(type).nunique() > 1 and not is_numeric_series(sample):
            report.add_warning(f"[{name}] column '{col}' has mixed Python types.")


def validate_required_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    *,
    name: str,
    report: ValidationReport,
) -> None:
    required = list(required or [])
    if not required:
        return
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add_error(f"[{name}] missing columns: {missing}")


def validate_missing_ratio(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    name: str,
    report: ValidationReport,
    warn_threshold: float = 0.3,
    error_threshold: float = 0.9,
) -> None:
    """Check missing ratio per column; warn/error by thresholds."""
    cols = [c for c in cols if c in df.columns]
    n = len(df)
    stats = {}
    for c in cols:
        miss = int(df[c].isna().sum())
        ratio = _ratio(miss, n)
        stats[c] = {"missing": miss, "ratio": ratio}
        if ratio >= error_threshold:
            report.add_error(
                f"[{name}] column '{c}' missing ratio {ratio:.2%} >= {error_threshold:.0%}."
            )
        elif ratio >= warn_threshold:
            report.add_warning(
                f"[{name}] column '{c}' missing ratio {ratio:.2%} >= {warn_threshold:.0%}."
            )
    report.add_info(f"{name}.missing_ratio", stats)


def validate_unique_values(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    name: str,
    report: ValidationReport,
    min_unique_warn: int = 1,
) -> None:
    """Warn if a column has too few unique values (may indicate constant)."""
    cols = [c for c in cols if c in df.columns]
    stats = {}
    for c in cols:
        nunique = int(df[c].nunique(dropna=True))
        stats[c] = {"nunique": nunique}
        if nunique <= min_unique_warn:
            report.add_warning(
                f"[{name}] column '{c}' has only {nunique} unique value(s)."
            )
    report.add_info(f"{name}.nunique", stats)


def ensure_numeric_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    name: str,
    report: ValidationReport,
) -> None:
    """Error if any given columns are not numeric-like."""
    cols = [c for c in cols if c in df.columns]
    non_numeric = [c for c in cols if not is_numeric_series(df[c])]
    if non_numeric:
        report.add_error(
            f"[{name}] expected numeric columns, found non-numeric: {non_numeric}"
        )


# ============================================================
# Cross-DataFrame Validators (ref vs cur)
# ============================================================


def assert_same_schema(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    cols: Optional[Iterable[str]],
    *,
    report: ValidationReport,
) -> List[str]:
    """
    Ensure ref and cur share the same set of columns to be compared.
    Returns the intersection (effective feature list).
    """
    if cols is None:
        common = list(ref_df.columns.intersection(cur_df.columns))
        if not common:
            report.add_error("[schema] no common columns between ref and cur.")
        return common

    cols = list(cols)
    missing_ref = [c for c in cols if c not in ref_df.columns]
    missing_cur = [c for c in cols if c not in cur_df.columns]
    if missing_ref:
        report.add_error(f"[schema] ref_df missing columns: {missing_ref}")
    if missing_cur:
        report.add_error(f"[schema] cur_df missing columns: {missing_cur}")
    return [c for c in cols if (c in ref_df.columns and c in cur_df.columns)]


def assert_compatible_dtypes(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    cols: Iterable[str],
    *,
    report: ValidationReport,
) -> None:
    """Warn if dtypes differ across ref/cur for the same column."""
    diffs = {}
    for c in cols:
        t_ref = str(ref_df[c].dtype)
        t_cur = str(cur_df[c].dtype)
        if t_ref != t_cur:
            diffs[c] = {"ref": t_ref, "cur": t_cur}
    if diffs:
        report.add_warning(f"[schema] dtype mismatch across ref/cur: {diffs}")


# ============================================================
# Task-specific Validators (label/pred)
# ============================================================


def infer_task_type(ref_labels: pd.Series) -> str:
    """
    Heuristic to infer 'classification' vs 'regression' from reference labels.
    - If integer-like or few unique values (<= max_class_cardinality) ⇒ classification
    - Else ⇒ regression
    """
    s = ref_labels.dropna()
    if s.empty:
        # Default to regression when unknown
        return "regression"

    if pd.api.types.is_integer_dtype(s):
        return "classification"

    return "regression"


def validate_binary_labels(
    y: pd.Series,
    *,
    name: str,
    report: ValidationReport,
) -> None:
    """Ensure labels are binary (0/1 or two unique values)."""
    s = y.dropna()
    uniq = sorted(s.unique().tolist())
    if len(uniq) != 2:
        report.add_warning(f"[{name}] labels are not binary: unique={uniq}")


def validate_prediction_probabilities(
    p: pd.Series,
    *,
    name: str,
    report: ValidationReport,
    tol: float = 1e-9,
) -> None:
    """Warn if predicted probabilities are outside [0,1] (with tolerance)."""
    s = pd.to_numeric(p, errors="coerce").dropna()
    below = (s < -tol).sum()
    above = (s > 1 + tol).sum()
    if below > 0 or above > 0:
        report.add_warning(
            f"[{name}] {below}+{above} prediction(s) fall outside [0,1]."
        )


# ============================================================
# High-level entry for ModelMonitor
# ============================================================


def validate_monitor_inputs(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    *,
    label_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    feature_cols: Optional[Iterable[str]] = None,
    missing_warn: float = 0.3,
    missing_error: float = 0.9,
) -> ValidationReport:
    """
    Comprehensive validation for ModelMonitor inputs.
    Returns a ValidationReport; caller may raise on error.

    Parameters
    ----------
    ref_df, cur_df : pd.DataFrame
        Reference and current datasets.
    label_col : Optional[str]
        Ground truth column.
    pred_col : Optional[str]
        Prediction column.
    feature_cols : Optional[Iterable[str]]
        Explicit features to compare. If None, use intersection.
    missing_warn : float
        Column-level missing ratio threshold to warn.
    missing_error : float
        Column-level missing ratio threshold to error.
    """
    report = ValidationReport()

    # Basic checks
    validate_dataframe_basic(ref_df, "ref_df", report)
    validate_dataframe_basic(cur_df, "cur_df", report)
    if not report.ok:
        return report

    # Required cols existence
    required = [c for c in [label_col, pred_col] if c]
    validate_required_columns(ref_df, required, name="ref_df", report=report)
    validate_required_columns(cur_df, required, name="cur_df", report=report)

    # Missing ratio / nunique diagnostics for required cols
    if required:
        validate_missing_ratio(
            ref_df,
            required,
            name="ref_df",
            report=report,
            warn_threshold=missing_warn,
            error_threshold=missing_error,
        )
        validate_missing_ratio(
            cur_df,
            required,
            name="cur_df",
            report=report,
            warn_threshold=missing_warn,
            error_threshold=missing_error,
        )
        validate_unique_values(ref_df, required, name="ref_df", report=report)
        validate_unique_values(cur_df, required, name="cur_df", report=report)

    # Feature set & schema alignment
    effective_features = assert_same_schema(ref_df, cur_df, feature_cols, report=report)
    # Remove label/pred from feature set if accidentally included
    effective_features = [
        c for c in effective_features if c not in (label_col, pred_col)
    ]
    report.add_info("features.effective", effective_features)

    assert_compatible_dtypes(ref_df, cur_df, effective_features, report=report)

    # Light feature diagnostics
    if effective_features:
        validate_missing_ratio(
            ref_df,
            effective_features,
            name="ref_df.features",
            report=report,
            warn_threshold=missing_warn,
            error_threshold=missing_error,
        )
        validate_missing_ratio(
            cur_df,
            effective_features,
            name="cur_df.features",
            report=report,
            warn_threshold=missing_warn,
            error_threshold=missing_error,
        )
        validate_unique_values(
            ref_df, effective_features, name="ref_df.features", report=report
        )
        validate_unique_values(
            cur_df, effective_features, name="cur_df.features", report=report
        )

    # Task-specific checks (if provided)
    if label_col and label_col in ref_df.columns and label_col in cur_df.columns:
        task = infer_task_type(ref_df[label_col])
        report.add_info("task_type_inferred", task)

        if task == "classification":
            validate_binary_labels(ref_df[label_col], name="ref_df", report=report)
            # pred checks for classification
            if pred_col and pred_col in ref_df.columns and pred_col in cur_df.columns:
                validate_prediction_probabilities(
                    ref_df[pred_col], name="ref_df", report=report
                )
                validate_prediction_probabilities(
                    cur_df[pred_col], name="cur_df", report=report
                )

    return report
