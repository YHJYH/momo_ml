# momo-ml

**momo-ml** 是一个用于生产环境的 **模型监控（Model Monitoring）** 工具库，旨在提供系统化、可扩展、可自动化的模型质量监测能力。  
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
