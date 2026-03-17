from typing import Dict, Any, List, Literal
import pandas as pd
import warnings

from momo_ml.metrics.psi import compute_psi

from momo_ml.metrics.ks import compute_ks
from momo_ml.metrics.kl import compute_kl


class DataDriftDetector:
    """
    Detects data drift between reference and current datasets.
    Supports numeric and categorical features.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference (baseline) dataset.
    cur_df : pd.DataFrame
        Current dataset to compare.
    features : List[str], optional
        If None, the intersection of columns in both datasets will be used.
    """

    def __init__(
        self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        features: List[str] = None,
        kl_buckets: int = 10,
        kl_base: Literal["e", "2", "10"] = "e",
        kl_epsilon: float = 1e-12,
        kl_handle_outside: str = "ignore",
    ):
        self.ref_df = ref_df.copy()
        self.cur_df = cur_df.copy()

        if features is None:
            # Only compare the common columns
            self.features = list(set(ref_df.columns) & set(cur_df.columns))
        else:
            seen = set()
            features = [f for f in features if not (f in seen or seen.add(f))]
            valid = [f for f in features if f in ref_df.columns and f in cur_df.columns]
            missing = [f for f in features if f not in ref_df.columns or f not in cur_df.columns]

            if missing:
                msg = (
                    "The following features are missing in one of the datasets "
                    "and will be skipped: " + ", ".join(map(str, missing))
                )
                warnings.warn(msg)
            self.features = valid

        if not self.features:
            warnings.warn(
                "No common features found between reference and current datasets. No drift will be computed."
            )

        self.kl_buckets = kl_buckets
        self.kl_base = kl_base
        self.kl_epsilon = kl_epsilon
        self.kl_handle_outside = kl_handle_outside

        # Split features by dtype
        self.numeric_features = []
        self.categorical_features = []
        self.incompatible_features = []
        for f in self.features:
            ref_is_num = pd.api.types.is_numeric_dtype(ref_df[f])
            cur_is_num = pd.api.types.is_numeric_dtype(cur_df[f])

            if ref_is_num and cur_is_num:
                self.numeric_features.append(f)
            elif not ref_is_num and not cur_is_num:
                self.categorical_features.append(f)
            else:
                # Incompatible types between ref and cur for this feature; skip it with a warning
                warnings.warn(
                    f"Feature '{f}' has incompatible types: ref type = {ref_df[f].dtype}, "
                    f"cur type = {cur_df[f].dtype}. It will be excluded from drift detection."
                )
                self.incompatible_features.append(f)

    def compute_feature_psi(self, feature: str) -> float:
        """Compute PSI for a single feature."""
        return compute_psi(
            self.ref_df[feature].values,
            self.cur_df[feature].values,
        )

    def compute_feature_kl(self, feature: str) -> float:
        """Compute KL divergence for a single feature."""
        return compute_kl(
            self.ref_df[feature].values,
            self.cur_df[feature].values,
            buckets=self.kl_buckets,
            base=self.kl_base,
            epsilon=self.kl_epsilon,
            handle_outside=self.kl_handle_outside,
        )
    
    def compute_feature_ks(self, feature: str) -> Dict[str, Any]:
        """Compute KS statistic for a single numeric feature."""
        return compute_ks(
            self.ref_df[feature].values,
            self.cur_df[feature].values,
            return_pvalue=True,
        )

    def compute_numeric_drift(self) -> Dict[str, Any]:
        """Compute drift for all numeric features."""
        results = {}
        for feat in self.numeric_features:
            results[feat] = {
                "psi": self.compute_feature_psi(feat),
                "kl": self.compute_feature_kl(feat),
                "ks": self.compute_feature_ks(feat),
            }
        return results

    def compute_categorical_drift(self) -> Dict[str, Any]:
        """Compute drift for categorical features."""
        results = {}
        for feat in self.categorical_features:
            results[feat] = {
                "psi": self.compute_feature_psi(feat),
                "kl": self.compute_feature_kl(feat),
            }
        return results

    def compute(self) -> Dict[str, Any]:
        """
        Execute full data drift detection, separated into:
        - numeric feature drift
        - categorical feature drift
        """
        return {
            "numeric_features": self.compute_numeric_drift(),
            "categorical_features": self.compute_categorical_drift(),
            "incompatible_features": self.incompatible_features,
        }
