"""
report_builder.py

Generates human-readable reports from model monitoring results.
Supports Markdown and JSON output formats.
"""

import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import warnings


class ReportBuilder:
    """
    Generate monitoring reports from the output of ModelMonitor.run_all().

    Parameters
    ----------
    monitor_results : dict
        The dictionary returned by ModelMonitor.run_all(), containing keys:
        'performance_drift', 'data_drift', 'prediction_drift'.
    thresholds : dict, optional
        User-defined thresholds for highlighting drift severity.
        Default thresholds:
        {
            "psi": {"low": 0.1, "medium": 0.25},   # PSI: <0.1 no drift, 0.1-0.25 moderate, >0.25 severe
            "ks": {"low": 0.1, "medium": 0.2},      # KS statistic
            "kl": {"low": 0.1, "medium": 0.5},      # KL divergence
            "js": {"low": 0.1, "medium": 0.5},      # JS divergence
            "wd": {},                               # Wasserstein distance - no default thresholds
        }
    """

    def __init__(self, monitor_results: Dict[str, Any], thresholds: Optional[Dict[str, Any]] = None):
        self.results = monitor_results
        self._set_default_thresholds()
        if thresholds:
            self._update_thresholds(thresholds)

    def _set_default_thresholds(self) -> None:
        """Set default drift thresholds for various metrics."""
        self.thresholds = {
            "psi": {"low": 0.1, "medium": 0.25},
            "ks": {"low": 0.1, "medium": 0.2},
            "kl": {"low": 0.1, "medium": 0.5},
            "js": {"low": 0.1, "medium": 0.5},
            # No default for Wasserstein distance – it's scale-dependent.
        }

    def _update_thresholds(self, user_thresholds: Dict[str, Any]) -> None:
        """Update thresholds with user-provided values."""
        for metric, thresh in user_thresholds.items():
            if metric in self.thresholds:
                self.thresholds[metric].update(thresh)
            else:
                self.thresholds[metric] = thresh

    # ------------------------------------------------------------------
    # Helper methods for formatting
    # ------------------------------------------------------------------
    def _risk_badge(self, metric_name: str, value: float) -> str:
        """
        Return a Markdown badge (or emoji) indicating drift severity.
        Only works for metrics with defined thresholds.
        """
        thresholds = self.thresholds.get(metric_name)
        if not thresholds:
            return ""

        low = thresholds.get("low")
        medium = thresholds.get("medium")

        if low is None and medium is None:
            return ""

        if low is not None and value < low:
            return "🟢"  # low risk
        elif medium is not None and value < medium:
            return "🟡"  # moderate risk
        else:
            return "🔴"  # high risk

    def _format_performance_section(self) -> str:
        """Format performance drift results into Markdown."""
        perf = self.results.get("performance_drift", {})
        if not perf:
            return "No performance drift data available.\n\n"

        if "error" in perf:
            return f"⚠️ **Performance drift error:** {perf['error']}\n\n"

        task = perf.get("task_type", "unknown")
        subtype = perf.get("classification_subtype")
        ref_metrics = perf.get("reference", {})
        cur_metrics = perf.get("current", {})
        delta = perf.get("delta", {})

        lines = [f"### Performance drift ({task})"]
        if subtype:
            lines.append(f"*Subtype: {subtype}*")

        # Build table
        lines.append("| Metric | Reference | Current | Delta |")
        lines.append("|--------|-----------|---------|-------|")
        for metric in sorted(set(ref_metrics.keys()) | set(cur_metrics.keys())):
            ref_val = ref_metrics.get(metric, "N/A")
            cur_val = cur_metrics.get(metric, "N/A")
            delta_val = delta.get(metric, "N/A")
            # Format numbers with 4 decimal places if they are numeric
            if isinstance(ref_val, (int, float)):
                ref_val = f"{ref_val:.4f}"
            if isinstance(cur_val, (int, float)):
                cur_val = f"{cur_val:.4f}"
            if isinstance(delta_val, (int, float)):
                delta_val = f"{delta_val:.4f}"
            lines.append(f"| {metric} | {ref_val} | {cur_val} | {delta_val} |")
        lines.append("")  # blank line after table
        return "\n".join(lines)

    def _format_data_drift_section(self) -> str:
        """Format data drift results into Markdown."""
        data = self.results.get("data_drift", {})
        if not data:
            return "No data drift data available.\n\n"

        if "error" in data:
            return f"⚠️ **Data drift error:** {data['error']}\n\n"

        numeric = data.get("numeric_features", {})
        categorical = data.get("categorical_features", {})
        incompatible = data.get("incompatible_features", [])

        lines = ["### Data drift"]

        if incompatible:
            lines.append(f"⚠️ **Incompatible features (skipped):** {', '.join(incompatible)}")
            lines.append("")

        # Numeric features
        if numeric:
            lines.append("#### Numeric features")
            lines.append("| Feature | PSI | KS | KL | JS | WD |")
            lines.append("|---------|-----|----|----|----|----|")
            for feat, metrics in numeric.items():
                psi = metrics.get("psi", "N/A")
                ks = metrics.get("ks", {}).get("statistic", "N/A")
                kl = metrics.get("kl", "N/A")
                js = metrics.get("js", "N/A")
                wd = metrics.get("wd", "N/A")

                # Store numeric values for badge computation
                psi_val = None
                if isinstance(psi, (int, float)):
                    psi_val = psi
                    psi = f"{psi:.4f}"

                if isinstance(ks, (int, float)):
                    ks = f"{ks:.4f}"

                kl_val = None
                if isinstance(kl, (int, float)):
                    kl_val = kl
                    kl = f"{kl:.4f}"

                js_val = None
                if isinstance(js, (int, float)):
                    js_val = js
                    js = f"{js:.4f}"

                if isinstance(wd, (int, float)):
                    wd = f"{wd:.4f}"

                psi_badge = self._risk_badge("psi", psi_val) if psi_val is not None else ""
                kl_badge = self._risk_badge("kl", kl_val) if kl_val is not None else ""
                js_badge = self._risk_badge("js", js_val) if js_val is not None else ""

                lines.append(f"| {feat} | {psi_badge} {psi} | {ks} | {kl_badge} {kl} | {js_badge} {js} | {wd} |")
            lines.append("")

        # Categorical features
        if categorical:
            lines.append("#### Categorical features")
            lines.append("| Feature | PSI | KL | JS | WD |")
            lines.append("|---------|-----|----|----|----|")
            for feat, metrics in categorical.items():
                psi = metrics.get("psi", "N/A")
                kl = metrics.get("kl", "N/A")
                js = metrics.get("js", "N/A")
                wd = metrics.get("wd", "N/A")

                psi_val = None
                if isinstance(psi, (int, float)):
                    psi_val = psi
                    psi = f"{psi:.4f}"

                kl_val = None
                if isinstance(kl, (int, float)):
                    kl_val = kl
                    kl = f"{kl:.4f}"

                js_val = None
                if isinstance(js, (int, float)):
                    js_val = js
                    js = f"{js:.4f}"

                if isinstance(wd, (int, float)):
                    wd = f"{wd:.4f}"

                psi_badge = self._risk_badge("psi", psi_val) if psi_val is not None else ""
                kl_badge = self._risk_badge("kl", kl_val) if kl_val is not None else ""
                js_badge = self._risk_badge("js", js_val) if js_val is not None else ""

                lines.append(f"| {feat} | {psi_badge} {psi} | {kl_badge} {kl} | {js_badge} {js} | {wd} |")
            lines.append("")

        return "\n".join(lines)

    def _format_prediction_drift_section(self) -> str:
        """Format prediction drift results into Markdown."""
        pred = self.results.get("prediction_drift", {})
        if not pred:
            return "No prediction drift data available.\n\n"

        if "error" in pred:
            return f"⚠️ **Prediction drift error:** {pred['error']}\n\n"

        pred_type = pred.get("prediction_type", "unknown")
        summary = pred.get("summary_statistics", {})
        dist_shift = pred.get("distribution_shift", {})
        decile_shift = pred.get("decile_shift")

        lines = [f"### Prediction drift ({pred_type})"]

        # Summary statistics
        if pred_type == "continuous":
            lines.append("#### Summary statistics")
            lines.append("| Statistic | Reference | Current | Delta |")
            lines.append("|-----------|-----------|---------|-------|")
            for stat in ["mean", "std", "min", "max", "q25", "q50", "q75"]:
                if stat in summary:
                    ref = summary[stat].get("reference", "N/A")
                    cur = summary[stat].get("current", "N/A")
                    delta = summary[stat].get("delta", "N/A")
                    if isinstance(ref, (int, float)):
                        ref = f"{ref:.4f}"
                    if isinstance(cur, (int, float)):
                        cur = f"{cur:.4f}"
                    if isinstance(delta, (int, float)):
                        delta = f"{delta:.4f}"
                    lines.append(f"| {stat} | {ref} | {cur} | {delta} |")
            lines.append("")
        else:  # categorical
            lines.append("#### Category proportions")
            categories = summary.get("categories", [])
            ref_props = summary.get("reference_proportions", {})
            cur_props = summary.get("current_proportions", {})
            delta_props = summary.get("delta_proportions", {})
            lines.append("| Category | Reference | Current | Delta |")
            lines.append("|----------|-----------|---------|-------|")
            for cat in categories:
                ref = ref_props.get(cat, 0)
                cur = cur_props.get(cat, 0)
                delta = delta_props.get(cat, 0)
                lines.append(f"| {cat} | {ref:.4f} | {cur:.4f} | {delta:.4f} |")
            lines.append("")

        # Distribution shift metrics
        if dist_shift:
            lines.append("#### Distribution shift metrics")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in dist_shift.items():
                if metric == "ks":
                    if isinstance(value, dict):
                        stat = value.get("statistic", "N/A")
                        pval = value.get("pvalue", "N/A")
                        if isinstance(stat, (int, float)):
                            stat = f"{stat:.4f}"
                        if isinstance(pval, (int, float)):
                            pval = f"{pval:.4f}"
                        lines.append(f"| KS (statistic) | {stat} |")
                        lines.append(f"| KS (p-value) | {pval} |")
                    else:
                        lines.append(f"| KS | {value} |")
                else:
                    # For metrics like psi, kl, js, l1_distance, l2_distance
                    if isinstance(value, (int, float)):
                        numeric_val = value
                        display_val = f"{value:.4f}"
                        # Add risk badge if applicable
                        badge = ""
                        if metric in ["psi", "kl", "js"]:
                            badge = self._risk_badge(metric, numeric_val)
                        lines.append(f"| {metric} | {badge} {display_val} |")
                    else:
                        lines.append(f"| {metric} | {value} |")

        # Decile shift (continuous only)
        if decile_shift:
            lines.append("#### Decile shift (quantiles)")
            quantiles = decile_shift.get("quantiles", [])
            ref_vals = decile_shift.get("ref_values", [])
            cur_vals = decile_shift.get("cur_values", [])
            delta_vals = decile_shift.get("delta", [])

            lines.append("| Quantile | Reference | Current | Delta |")
            lines.append("|----------|-----------|---------|-------|")
            for q, r, c, d in zip(quantiles, ref_vals, cur_vals, delta_vals):
                lines.append(f"| {q:.2f} | {r:.4f} | {c:.4f} | {d:.4f} |")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def to_markdown(self, title: Optional[str] = None, include_metadata: bool = True) -> str:
        """
        Generate a Markdown string summarizing the monitoring results.

        Parameters
        ----------
        title : str, optional
            Title of the report. If not provided, a default title with timestamp is used.
        include_metadata : bool, default True
            Whether to include generation timestamp and other metadata.

        Returns
        -------
        str
            Markdown formatted report.
        """
        lines = []
        if include_metadata:
            if title is None:
                title = f"Model Monitoring Report – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            lines.append(f"# {title}")
            lines.append("")

        lines.append(self._format_performance_section())
        lines.append(self._format_data_drift_section())
        lines.append(self._format_prediction_drift_section())

        return "\n".join(lines)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Serialize the monitoring results to a JSON string.

        Parameters
        ----------
        indent : int, optional
            Indentation level for pretty printing. If None, compact representation.

        Returns
        -------
        str
            JSON string.
        """
        # Include a timestamp and maybe a version field
        output = {
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
        }
        return json.dumps(output, indent=indent, default=str)

    def save_markdown(self, filepath: str, **kwargs) -> None:
        """
        Save the Markdown report to a file.

        Parameters
        ----------
        filepath : str
            Path to output .md file.
        **kwargs
            Additional arguments passed to to_markdown().
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_markdown(**kwargs))

    def save_json(self, filepath: str, **kwargs) -> None:
        """
        Save the JSON report to a file.

        Parameters
        ----------
        filepath : str
            Path to output .json file.
        **kwargs
            Additional arguments passed to to_json().
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(**kwargs))
