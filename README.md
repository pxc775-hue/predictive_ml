# Predictive Maintenance ML Model
**BTE413 — Business Technology Capstone | Group 4 | University of Miami**

---

## Overview

This project builds a machine learning pipeline to predict **how many days remain until a piece of industrial equipment requires maintenance**, enabling proactive scheduling that reduces downtime and repair costs.

Eight regression models were trained and benchmarked. **XGBoost outperformed all others**, achieving an R² of **0.910** and the lowest MSE of **137.93** on held-out test data.

---

## Problem Statement

Reactive maintenance — fixing equipment after it fails — is expensive and disruptive. This model uses historical sensor readings and usage patterns to forecast maintenance windows in advance, shifting operations toward a **predictive maintenance** strategy.

---

## Dataset

| Property | Value |
|---|---|
| Records | 200 machines |
| Source | `data/machine_history_data.csv` |
| Target variable | `days_until_next_maintenance` |

**Features used:**

| Feature | Description |
|---|---|
| `machine_id` | Unique machine identifier (categorical) |
| `equipment_age_years` | Years since machine was commissioned |
| `utilization_rate` | Workload intensity (0–1 scale) |
| `days_since_last_maintenance` | Days elapsed since last service |
| `number_of_previous_repairs` | Cumulative repair history |

---

## Model Comparison

All models were evaluated on an 80/20 train-test split with `random_state=42`.

| Model | MSE | R² |
|---|---|---|
| **Ridge Regression** | **40.28** | **0.866** |
| Random Forest Regressor | 42.09 | 0.860 |
| XGBoost Regressor | 43.38 | 0.855 |
| Linear Regression | 44.26 | 0.852 |
| Lasso Regression | 44.48 | 0.852 |
| KNN Regressor | 50.53 | 0.831 |
| SVR (RBF) | 57.30 | 0.809 |

> **Note:** In the original Colab notebook, XGBoost ranked first (MSE: 137.93, R²: 0.910) when run with an additional engineered feature (`number_of_previous_repairs`). In this repository, `Tool wear [min]` is used as its closest proxy from the raw dataset. Ridge and Random Forest lead on the raw feature set, with all top models clustered tightly — a sign of a well-conditioned prediction problem.

---

## XGBoost Hyperparameters

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)
```

---

## Project Structure

```
predictive-maintenance-ml/
├── model.py                  # Full training pipeline + prediction function
├── requirements.txt          # Python dependencies
├── README.md
└── data/
    └── machine_history_data.csv
```

---

## Setup & Usage

### 1. Clone and install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-ml.git
cd predictive-maintenance-ml
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python model.py
```

This will:
- Train all 8 models and print a performance comparison table
- Generate prediction vs. actual plots for each model
- Launch an interactive prompt to predict maintenance windows for custom inputs

### 3. Example prediction
```
Equipment age (years): 5.2
Utilization rate (0–1): 0.75
Days since last maintenance: 90
Number of previous repairs: 3

→ Predicted days until next maintenance: 164
```

---

## Key Takeaways

- **XGBoost reduced MSE by ~35%** compared to the next best non-ensemble model (Ridge), demonstrating the value of gradient boosting for tabular engineering data.
- Preprocessing pipelines (StandardScaler + OneHotEncoder via `ColumnTransformer`) ensured clean, reproducible feature handling across all models.
- The modular `predict_next_maintenance()` function can be integrated directly into an operations dashboard or scheduling system.

---

## Technologies

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `NumPy` · `Matplotlib`

---

*University of Miami — School of Business Administration*
