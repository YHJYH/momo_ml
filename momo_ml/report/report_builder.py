# momo_ml/report/report_builder.py

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from matplotlib.figure import Figure


# =====================================================================
# Utility functions
# =====================================================================


def _fig_to_base64(fig: Figure) -> str:
    """Convert a matplotlib Figure into base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _escape_html(text: str) -> str:
    """Basic HTML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# =====================================================================
# ReportBuilder
# =====================================================================


@dataclass
class ReportBuilder:
    """
    Assemble HTML monitoring reports from:
    - performance drift results
    - data drift results
    - prediction drift results
    - plots (optional)

    Parameters
    ----------
    monitor_output : Dict[str, Any]
        The result from ModelMonitor.run_all().
    plots : Optional[Dict[str, Figure]]
        Optional dict of plot figures to embed in the report.
    title : str
        Report title shown at the top.
    """

    monitor_output: Dict[str, Any]
    plots: Optional[Dict[str, Figure]] = None
    title: str = "Model Monitoring Report"

    # -----------------------------------------------------------------
    # Render sections
    # -----------------------------------------------------------------

    def _render_section_header(self, text: str) -> str:
        return f"<h2 style='margin-top:30px;'>{_escape_html(text)}</h2>\n"

    def _render_json_block(self, data: Any) -> str:
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return f"<pre style='background:#f5f5f5;padding:10px;border-radius:4px;'>{_escape_html(pretty)}</pre>"

    def _render_image(self, fig: Figure, caption: Optional[str] = None) -> str:
        src = _fig_to_base64(fig)
        cap_html = (
            f"<div style='text-align:center;margin-top:5px;color:#555;'>{_escape_html(caption)}</div>"
            if caption
            else ""
        )
        return f"<div style='margin:15px 0;'>" f"{src}" f"{cap_html}" f"</div>"

    # -----------------------------------------------------------------
    # Rendering core sections
    # -----------------------------------------------------------------

    def _render_performance_drift(self) -> str:
        perf = self.monitor_output.get("performance_drift", {})
        html = self._render_section_header("Performance Drift")
        html += self._render_json_block(perf)
        return html

    def _render_data_drift(self) -> str:
        drift = self.monitor_output.get("data_drift", {})
        html = self._render_section_header("Data Drift")
        html += self._render_json_block(drift)
        return html

    def _render_prediction_drift(self) -> str:
        pred = self.monitor_output.get("prediction_drift", {})
        html = self._render_section_header("Prediction Drift")
        html += self._render_json_block(pred)
        return html

    def _render_plots(self) -> str:
        if not self.plots:
            return ""

        html = self._render_section_header("Visualizations")

        for name, fig in self.plots.items():
            html += self._render_image(fig, caption=name)

        return html

    # -----------------------------------------------------------------
    # Assemble full HTML
    # -----------------------------------------------------------------

    def to_html(self) -> str:
        """Return full HTML report as a string."""
        html_parts: List[str] = []
        html_parts.append("<html><head>")
        html_parts.append(f"<title>{_escape_html(self.title)}</title>")
        html_parts.append(
            """
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                       padding: 20px; line-height: 1.6; }
                h1 { text-align:center; margin-bottom:30px; }
                h2 { color:#2d6cdf; }
            </style>
        """
        )
        html_parts.append("</head><body>")

        html_parts.append(f"<h1>{_escape_html(self.title)}</h1>")

        # Sections
        html_parts.append(self._render_performance_drift())
        html_parts.append(self._render_data_drift())
        html_parts.append(self._render_prediction_drift())
        html_parts.append(self._render_plots())

        html_parts.append("</body></html>")
        return "".join(html_parts)

    # -----------------------------------------------------------------
    # File I/O
    # -----------------------------------------------------------------

    def save_html(self, path: str) -> None:
        """Write HTML report to file."""
        html = self.to_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    # -----------------------------------------------------------------
    # (Optional) PDF export stub (future implementation)
    # -----------------------------------------------------------------

    def to_pdf(self, path: str) -> None:
        """
        Placeholder interface for PDF export.
        You may later implement via:
        - WeasyPrint
        - wkhtmltopdf
        - headless Chromium

        Currently, raises NotImplementedError.
        """
        raise NotImplementedError("PDF export is not implemented yet.")
