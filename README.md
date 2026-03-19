[![PyPI version](https://img.shields.io/pypi/v/momo-ml)](https://pypi.org/project/momo-ml/)
[![Python Versions](https://img.shields.io/pypi/pyversions/momo-ml)](https://pypi.org/project/momo-ml/)
[![License](https://img.shields.io/pypi/l/momo-ml)](https://github.com/YHJYH/momo_ml/blob/main/LICENSE)

# momo-ml

**momo-ml** (**MO**del **MO**nitoring for **ML**) is a production‑oriented library for systematically, scalably, and automatically monitoring model quality.  
It covers key monitoring dimensions such as **model performance drift**, **data drift**, and **prediction drift**, and also supports automatic generation of visual reports.

This project is suitable for data science, ML engineering, MLOps, and other scenarios – it can be integrated into various monitoring and governance workflows after model deployment.

---

## 📌 Features

### **1. Performance Drift**
Evaluate how model performance changes over time by comparing a **reference dataset** with a **current dataset**.  
Supports both classification and regression tasks with a full suite of metrics, enabling detailed drift detection across time windows.

Key capabilities include:
- Classification metrics such as AUC, Accuracy, Precision, Recall, F1, and KS (supports binary & multiclass).
- Regression metrics including RMSE, MAE, R², SMAPE, P90/P95 Error, and Huber Loss.
- Automatic task‑type detection (classification vs. regression) based on label/prediction structure.
- Flexible configuration for label and prediction columns.

Use this to monitor whether a model’s predictive quality remains stable in production.

---

### **2. Data Drift**
Detect how input feature distributions shift between a **reference** and a **current** dataset.  
Works for both numeric and categorical features, with multiple statistical measures to quantify distributional changes.

Key capabilities include:
- For numeric features: PSI, KL divergence, JS divergence, KS statistic, and Wasserstein distance.
- For categorical features: PSI, KL divergence, JS divergence, and Wasserstein distance.
- Automatic separation of numeric/categorical fields.
- Safe handling of incompatible feature types with clear warnings.

This module helps identify upstream data issues such as schema drift, feature instability, or gradual population changes.

---

### **3. Prediction Drift**
Monitor changes in the distribution of model predictions, independent of labels.  
Useful for real‑time systems, unlabeled production environments, and early anomaly detection.

Key capabilities include:
- Summary statistics for numeric predictions (mean, std, quantiles) and categorical predictions (proportion changes).
- Drift metrics including PSI, KL, JS (all prediction types) and KS for continuous predictions.
- Histogram‑based L1/L2 distance and quantile/decile shift for continuous outputs.
- Automatic distinction between numeric and categorical prediction behavior using a configurable threshold.

Ideal for detecting unexpected output shifts before they escalate into performance degradation.


### **4. Automated Reporting**
Unified report generation capabilities:
- Automatically generate drift charts (matplotlib / plotly)
- One‑click HTML or PDF report generation
- Extensible storage or push to custom dashboards

---

## 🔧 Installation

```bash
pip install momo-ml
