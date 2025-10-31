"""
Prediction script for Rent Price Model (XGBoost).
Loads the trained model and generates predictions on the processed test set,
rounding results to the nearest €50 for realistic outputs.
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# === Step 1: Load processed test data ===
test = pd.read_csv("data/test_preprocessed.csv")

# === Step 2: Load trained model ===
with open("models/xgb_best.pkl", "rb") as f:
    model = pickle.load(f)

# === Step 3: Prepare data for prediction ===
X_test = test.drop(columns=["y"], errors="ignore")
dtest = xgb.DMatrix(X_test)

# === Step 4: Predict and inverse transform ===
y_pred_test = model.predict(dtest) ** 2  # inverse √y → y

# === Step 5: Round predictions to nearest €50 ===
test["y_predicted"] = (y_pred_test // 50).round() * 50

# === Step 6: Save outputs ===
test[["y_predicted"]].to_csv("data/test_predictions.csv", index=False)

print("✅ Predictions saved to data/test_predictions.csv")
print(test[["y_predicted"]].head())
