from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("ml/regional_data/regional_traffic.csv")
PREDICTIONS_PATH = Path(
    "ml/regional_model/regional_timestamped_test_predictions.csv"
)
OUTPUT_DIR = Path("ml/regional_model")

# Selected on validation data only.
THRESHOLD = 10.0
TRAIN_END = pd.Timestamp("2024-08-24")

print("Loading training history and test predictions...")
history = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
pred = pd.read_csv(
    PREDICTIONS_PATH,
    parse_dates=["forecast_timestamp"],
)

history["segment_id"] = history["segment_id"].astype(str)
pred["segment_id"] = pred["segment_id"].astype(str)

# Compute road activity from training data only.
train = history[history["timestamp"] < TRAIN_END]
road_means = train.groupby("segment_id")["probe_count"].mean()

pred["training_mean_probe_count"] = pred["segment_id"].map(road_means)

if pred["training_mean_probe_count"].isna().any():
    raise ValueError("Some test roads have no training-period mean.")

if pred[
    ["actual_probe_count", "naive_prediction", "predicted_probe_count"]
].isna().any().any():
    raise ValueError("Missing values found in test predictions.")

actual = pred["actual_probe_count"].to_numpy()
naive = pred["naive_prediction"].to_numpy()
lstm = pred["predicted_probe_count"].to_numpy()

use_lstm = pred["training_mean_probe_count"].to_numpy() >= THRESHOLD
hybrid = np.where(use_lstm, lstm, naive)

def report(name, values):
    mae = mean_absolute_error(actual, values)
    rmse = np.sqrt(mean_squared_error(actual, values))
    return {
        "method": name,
        "MAE": mae,
        "RMSE": rmse,
    }

results = pd.DataFrame([
    report("Naive-only", naive),
    report("LSTM-only", lstm),
    report(f"Hybrid (threshold={THRESHOLD:g})", hybrid),
])

naive_mae = results.loc[results["method"] == "Naive-only", "MAE"].iloc[0]
lstm_mae = results.loc[results["method"] == "LSTM-only", "MAE"].iloc[0]
hybrid_mae = results.loc[
    results["method"] == f"Hybrid (threshold={THRESHOLD:g})", "MAE"
].iloc[0]

results["MAE improvement vs naive (%)"] = (
    100 * (naive_mae - results["MAE"]) / naive_mae
)
results["MAE improvement vs LSTM (%)"] = (
    100 * (lstm_mae - results["MAE"]) / lstm_mae
)

summary_path = OUTPUT_DIR / "regional_hybrid_test_summary.csv"
results.to_csv(summary_path, index=False)

pred["hybrid_prediction"] = hybrid
pred["forecast_method"] = np.where(use_lstm, "LSTM", "Naive")
pred_path = OUTPUT_DIR / "regional_hybrid_test_predictions.csv"
pred.to_csv(pred_path, index=False)

print("\n===== UNTOUCHED TEST RESULTS =====")
print("Test period:", pred["forecast_timestamp"].min(), "to",
      pred["forecast_timestamp"].max())
print("Predictions:", len(pred))
print("Road segments:", pred["segment_id"].nunique())
print("Threshold selected on validation:", THRESHOLD)
print("Roads using LSTM:", pred.loc[use_lstm, "segment_id"].nunique())
print("Roads using naive:", pred.loc[~use_lstm, "segment_id"].nunique())
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSaved summary:", summary_path)
print("Saved predictions:", pred_path)
print("Original test predictions and model were not modified.")
