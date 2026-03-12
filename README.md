# momo-ml

**momo-ml** 是一个用于生产环境的 **模型监控（MOdel MOnitoring for ML）** 工具库，旨在提供系统化、可扩展、可自动化的模型质量监测能力。  
库覆盖模型性能漂移（performance drift）、数据漂移（data drift）、预测值漂移（prediction drift）等关键监控维度，同时支持自动生成可视化报表。

本项目适用于数据科学、ML 工程、MLOps 等场景，可集成至模型上线后的各类监控与治理流程。

---

## 📌 Features

### **1. Performance Drift**
监测模型在不同时间窗口的预测性能变化：
- AUC / F1 / Precision / Recall / RMSE 等指标
- Reference window vs current window 对比
- Rolling window 趋势分析

### **2. Data Drift**
检测输入数据分布稳定性：
- Population Stability Index (PSI)
- KL Divergence、KS Test
- 数值特征统计变化（mean / var / quantile shift）
- 类别特征分布漂移（frequency shift）

### **3. Prediction Drift**
监测模型输出行为是否异常：
- 输出分布变化
- 分箱稳定性（如 deciles shift）
- 不同群体/分段之间的预测差异

### **4. Automated Reporting**
提供统一报表生成能力：
- 自动生成 Drift 图表（matplotlib / plotly）
- 一键生成 HTML 或 PDF 报告
- 可扩展存储或推送到自定义 dashboard

---

## 🔧 Installation

```bash
pip install momo-ml


momo-ml/
├── momo_ml/
│   ├── __init__.py
│   ├── monitor/
│   │   ├── __init__.py
│   │   ├── model_monitor.py        # 主监控逻辑入口
│   │   ├── drift_detector.py       # drift 检测引擎
│   │   ├── performance.py          # 性能指标相关
│   │   ├── prediction.py           # 预测漂移检测
│   │   └── data_drift.py           # 数据漂移检测
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── psi.py                  # PSI
│   │   ├── kl.py                   # KL Divergence
│   │   ├── ks.py                   # KS Test
│   │   └── performance_metrics.py  # AUC/Precision/F1...
│   │
│   ├── report/
│   │   ├── __init__.py
│   │   ├── report_builder.py       # HTML / PDF 生成
│   │   ├── html_template.html      # HTML 模板
│   │   └── pdf_template.html       # PDF 模板（基于 HTML 转换）
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── plotting.py             # 所有图表方法
│   │   └── validation.py           # 输入校验
│
├── tests/
│   ├── test_data_drift.py
│   ├── test_model_monitor.py
│   └── test_performance_metrics.py
│   └── test_prediction_drift.py
│
├── README.md
├── pyproject.toml
└── setup.cfg
