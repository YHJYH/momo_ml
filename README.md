[![PyPI version](https://img.shields.io/pypi/v/momo-ml)](https://pypi.org/project/momo-ml/)
![Python Versions](https://img.shields.io/pypi/pyversions/momo-ml)
![License](https://img.shields.io/pypi/l/momo-ml)

# momo-ml

**momo-ml** (**MO**del **MO**nitoring for **ML**) is a production‑oriented library for systematically, scalably, and automatically monitoring model quality.  
It covers key monitoring dimensions such as **performance drift**, **data drift**, and **prediction drift**, and also supports automatic generation of visual reports.

This project is suitable for data science, ML engineering, MLOps, and other scenarios – it can be integrated into various monitoring and governance workflows after model deployment.

---

## 📌 Features

### **1. Performance Drift**
Monitor changes in model prediction performance over different time windows:
- Metrics: AUC / F1 / Precision / Recall / RMSE, etc.
- Compare reference window vs. current window.
- Rolling window trend analysis.

### **2. Data Drift**
Detect shifts in the stability of input data distributions:
- Population Stability Index (PSI)
- KL Divergence, KS Test
- Statistical changes in numerical features (mean / var / quantile shift)
- Distribution drift in categorical features (frequency shift)

### **3. Prediction Drift**
Monitor abnormal model output behavior:
- Changes in output distribution
- Binning stability (e.g., deciles shift)
- Prediction differences across groups/segments

### **4. Automated Reporting**
Unified report generation capabilities:
- Automatically generate drift charts (matplotlib / plotly)
- One‑click HTML or PDF report generation
- Extensible storage or push to custom dashboards

---

## 🔧 Installation

```bash
pip install momo-ml