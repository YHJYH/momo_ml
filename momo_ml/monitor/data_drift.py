from typing import Dict, Any, List
import pandas as pd

from momo_ml.metrics.psi import compute_psi

# from momo_ml.metrics.ks import compute_ks       # (next step)
# from momo_ml.metrics.kl import compute_kl       # (next step)


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
    ):
        self.ref_df = ref_df.copy()
        self.cur_df = cur_df.copy()

        if features is None:
            # Only compare the common columns
            self.features = list(set(ref_df.columns) & set(cur_df.columns))
        else:
            self.features = features

        # Split features by dtype
        self.numeric_features = [
            f for f in self.features if pd.api.types.is_numeric_dtype(ref_df[f])
        ]
        self.categorical_features = [
            f for f in self.features if f not in self.numeric_features
        ]

    def compute_feature_psi(self, feature: str) -> float:
        """Compute PSI for a single feature."""
        return compute_psi(
            self.ref_df[feature].values,
            self.cur_df[feature].values,
        )

    def compute_numeric_drift(self) -> Dict[str, Any]:
        """Compute drift for all numeric features."""
        results = {}
        for feat in self.numeric_features:
            results[feat] = {
                "psi": self.compute_feature_psi(feat),
                # "ks": compute_ks(...),   # to be added
                # "kl": compute_kl(...),   # to be added
            }
        return results

    def compute_categorical_drift(self) -> Dict[str, Any]:
        """Compute drift for categorical features (PSI-based)."""
        results = {}
        for feat in self.categorical_features:
            results[feat] = {
                "psi": self.compute_feature_psi(feat)
                # (Categorical drift often uses PSI only)
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
        }
