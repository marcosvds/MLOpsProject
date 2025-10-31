"""
Training script for Rent Price Prediction (XGBoost + MLflow).
Performs small grid search, logs all runs to MLflow, and saves the best model.
"""

# === Environment and libraries ===
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import pickle

# === Step 1: Load environment variables ===
load_dotenv() 

# === Step 2: Configure MLflow tracking ===
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("rent-price-xgb")

# === Step 3: Load preprocessed dataset ===
train = pd.read_csv("data/train_preprocessed.csv")

# === Step 4: Split features and target ===
X = train.drop(columns=["y"], errors="ignore")
y = np.sqrt(train["y"])  # square-root transform for stability

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# === Step 5: Define small parameter grid for search ===
param_grid = [
    {"max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 0.5},
    {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 1.0, "reg_lambda": 1.0},
]

best_mae = float("inf")
best_model = None
best_params = None
best_run_id = None

# === Step 6: Run grid search with MLflow tracking ===
for params in param_grid:
    xgb_params = {
        "objective": "reg:squarederror",
        "reg_alpha": 0.1,
        "seed": 42,
        "verbosity": 0,
        **params
    }

    run_name = f"xgb_md{params['max_depth']}_lr{params['learning_rate']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(xgb_params)

        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # === Evaluate MAE on original (€) scale ===
        y_pred_val = model.predict(dval) ** 2
        mae_val = mean_absolute_error((y_val ** 2), y_pred_val)
        mlflow.log_metric("val_MAE", mae_val)

        # === Log model to MLflow registry ===
        signature = infer_signature(X_val, model.predict(dval))
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="rent-xgb-model"
        )

        # Track the best model
        if mae_val < best_mae:
            best_mae = mae_val
            best_model = model
            best_params = xgb_params
            best_run_id = mlflow.active_run().info.run_id

# === Step 7: Display best run info ===
print("\nTraining complete!")
print(f"Best MAE (€): {best_mae:.2f}")
print(f"Best parameters: {best_params}")
print(f"Logged MLflow run ID: {best_run_id}")

# === Step 8: Save best model locally ===
with open("models/xgb_best.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model saved at models/xgb_best.pkl")
