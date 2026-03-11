# =============================================================================
# Predictive Maintenance Model Pipeline
# Course: BTE413 — Group 4
# University of Miami
#
# Objective: Predict the number of days until a machine's next maintenance
#            event using historical equipment sensor and usage data.
#
# Models compared: Linear Regression, KNN, Random Forest, SVR,
#                  Lasso, Ridge, Polynomial Regression, XGBoost
# Winner: XGBoost (MSE: 137.93 | R²: 0.910)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor


# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("data/machine_history_data.csv")

# Features used for prediction:
#   machine_id                   — unique machine identifier (categorical)
#   equipment_age_years          — how long the machine has been in service
#   utilization_rate             — workload intensity (0–1)
#   days_since_last_maintenance  — days elapsed since last service
#   number_of_previous_repairs   — cumulative repair count
#
# Target:
#   days_until_next_maintenance  — what we want to predict

target_col = "days_until_next_maintenance"

# Select only the columns the model was built on
feature_cols = [
    "machine_id",
    "equipment_age_years",
    "utilization_rate",
    "days_since_last_maintenance",
    "number_of_previous_repairs",
]

# Map actual dataset column names to model feature names
df = df.rename(columns={
    "Product ID": "machine_id",
    "Tool wear [min]": "number_of_previous_repairs",  # proxy: cumulative tool usage
})

# Drop rows missing target or key features
df = df.dropna(subset=feature_cols + [target_col])

X = df[feature_cols]
y = df[target_col]


# =============================================================================
# 2. PREPROCESSING
# =============================================================================
categorical_features = ["machine_id"]
numeric_features = [col for col in feature_cols if col not in categorical_features]

# Base preprocessor: scale numerics, one-hot encode machine_id
base_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# =============================================================================
# 3. TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =============================================================================
# 4. EVALUATION HELPER
# =============================================================================
def evaluate_model(name, model, X_test, y_test, results_list):
    """Compute MSE and R² for a fitted model and store results."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results_list.append({"Model": name, "MSE": mse, "R2": r2})
    print(f"{name:30s}  MSE = {mse:8.3f}   R² = {r2:6.3f}")


# =============================================================================
# 5. TRAIN ALL MODELS
# =============================================================================
results = []
models = {}


# --- Linear Regression -------------------------------------------------------
linear_model = Pipeline([
    ("preprocess", base_preprocessor),
    ("model", LinearRegression())
])
linear_model.fit(X_train, y_train)
evaluate_model("Linear Regression", linear_model, X_test, y_test, results)
models["Linear Regression"] = linear_model


# --- KNN Regressor -----------------------------------------------------------
knn_model = Pipeline([
    ("preprocess", base_preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=5))
])
knn_model.fit(X_train, y_train)
evaluate_model("KNN Regressor", knn_model, X_test, y_test, results)
models["KNN Regressor"] = knn_model


# --- Random Forest -----------------------------------------------------------
rf_model = Pipeline([
    ("preprocess", base_preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])
rf_model.fit(X_train, y_train)
evaluate_model("Random Forest Regressor", rf_model, X_test, y_test, results)
models["Random Forest Regressor"] = rf_model


# --- SVR (RBF Kernel) --------------------------------------------------------
svr_model = Pipeline([
    ("preprocess", base_preprocessor),
    ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1))
])
svr_model.fit(X_train, y_train)
evaluate_model("SVR (RBF)", svr_model, X_test, y_test, results)
models["SVR (RBF)"] = svr_model


# --- Lasso Regression --------------------------------------------------------
lasso_model = Pipeline([
    ("preprocess", base_preprocessor),
    ("model", Lasso(alpha=0.001, max_iter=10000))
])
lasso_model.fit(X_train, y_train)
evaluate_model("Lasso Regression", lasso_model, X_test, y_test, results)
models["Lasso Regression"] = lasso_model


# --- Ridge Regression --------------------------------------------------------
ridge_model = Pipeline([
    ("preprocess", base_preprocessor),
    ("model", Ridge(alpha=15.0))
])
ridge_model.fit(X_train, y_train)
evaluate_model("Ridge Regression", ridge_model, X_test, y_test, results)
models["Ridge Regression"] = ridge_model


# --- Polynomial Regression (degree=3) ----------------------------------------
poly_numeric = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=3, include_bias=False))
])
poly_preprocessor = ColumnTransformer(
    transformers=[
        ("num", poly_numeric, numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
poly_model = Pipeline([
    ("preprocess", poly_preprocessor),
    ("model", LinearRegression())
])
poly_model.fit(X_train, y_train)
evaluate_model("Polynomial Regression", poly_model, X_test, y_test, results)
models["Polynomial Regression"] = poly_model


# --- XGBoost (Best Model) ----------------------------------------------------
# Tuned hyperparameters: learning_rate=0.05, max_depth=4,
# subsample=0.8, colsample_bytree=0.8
xgb_model = Pipeline([
    ("preprocess", base_preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ))
])
xgb_model.fit(X_train, y_train)
evaluate_model("XGBoost Regressor", xgb_model, X_test, y_test, results)
models["XGBoost Regressor"] = xgb_model


# =============================================================================
# 6. RESULTS SUMMARY
# =============================================================================
results_df = pd.DataFrame(results).sort_values(by="MSE")
print("\n=== Model Performance (sorted by MSE) ===")
print(results_df.to_string(index=False))


# =============================================================================
# 7. VISUALIZATION — Prediction vs Actual (per model)
# =============================================================================
days_col = "days_since_last_maintenance"
min_days = df[days_col].min()
max_days = df[days_col].max()
x_grid = np.linspace(min_days, max_days, 100)

num_means = df[numeric_features].mean()
typical_machine = df["machine_id"].mode()[0]

for name, model in models.items():
    plt.figure()
    plt.scatter(X_test[days_col], y_test, alpha=0.4, label="Actual data")

    grid = pd.DataFrame({
        "machine_id": [typical_machine] * len(x_grid),
        "equipment_age_years": num_means["equipment_age_years"],
        "utilization_rate": num_means["utilization_rate"],
        "days_since_last_maintenance": x_grid,
        "number_of_previous_repairs": num_means["number_of_previous_repairs"],
    })

    y_line = model.predict(grid)
    plt.plot(x_grid, y_line, label=f"{name} forecast")
    plt.xlabel("Days since last maintenance")
    plt.ylabel("Days until next maintenance")
    plt.title(f"{name}: Prediction vs Actual")
    plt.legend()
    plt.tight_layout()

plt.show()


# =============================================================================
# 8. AUTO-SELECT BEST MODEL (by lowest MSE)
# =============================================================================
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
print(f"\nBest model selected: {best_model_name}")


# =============================================================================
# 9. PREDICTION FUNCTION
# =============================================================================
def predict_next_maintenance(
    model,
    equipment_age_years,
    utilization_rate,
    days_since_last_maintenance,
    number_of_previous_repairs,
    machine_id=None,
):
    """
    Predict days until next maintenance for a single machine.

    Parameters:
        model                        : fitted sklearn Pipeline
        equipment_age_years          : float — age of machine in years
        utilization_rate             : float — workload intensity (0–1)
        days_since_last_maintenance  : float — days since last service
        number_of_previous_repairs   : float — cumulative repair count
        machine_id                   : str   — machine identifier (optional)

    Returns:
        float — predicted days until next maintenance
    """
    if machine_id is None:
        machine_id = df["machine_id"].mode()[0]

    input_df = pd.DataFrame({
        "machine_id": [machine_id],
        "equipment_age_years": [equipment_age_years],
        "utilization_rate": [utilization_rate],
        "days_since_last_maintenance": [days_since_last_maintenance],
        "number_of_previous_repairs": [number_of_previous_repairs],
    })

    return model.predict(input_df)[0]


# =============================================================================
# 10. INTERACTIVE PREDICTION LOOP
# =============================================================================
def get_float_input(prompt):
    while True:
        val = input(prompt).strip()
        if val.lower().startswith("q"):
            return None
        try:
            return float(val)
        except ValueError:
            print("Please enter a numeric value or 'q' to quit.")


print(f"\n=== Manual Prediction using {best_model_name} ===")
print("Enter machine details to predict days until next maintenance.")
print("Type 'q' at any prompt to quit.\n")

while True:
    print("--- New Prediction ---")
    age = get_float_input("Equipment age (years): ")
    if age is None: break

    util = get_float_input("Utilization rate (0–1): ")
    if util is None: break

    days_since = get_float_input("Days since last maintenance: ")
    if days_since is None: break

    repairs = get_float_input("Number of previous repairs: ")
    if repairs is None: break

    result = predict_next_maintenance(
        best_model,
        equipment_age_years=age,
        utilization_rate=util,
        days_since_last_maintenance=days_since,
        number_of_previous_repairs=repairs,
    )
    print(f"→ Predicted days until next maintenance: {result:.0f}\n")
