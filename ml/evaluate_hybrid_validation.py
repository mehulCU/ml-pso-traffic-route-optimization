from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("ml/regional_data/regional_traffic.csv")
MODEL_PATH = Path("ml/regional_model/regional_lstm.keras")
SCALER_PATH = Path("ml/regional_model/regional_scaler.pkl")
OUTPUT_DIR = Path("ml/regional_model")

LOOKBACK = 24
TRAIN_END = pd.Timestamp("2024-08-24")
VAL_END = pd.Timestamp("2024-08-26")

FEATURES = [
    "probe_count",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "speed_limit",
    "frc",
    "distance",
]

# Thresholds are evaluated using validation data only.
THRESHOLDS = [0, 1, 5, 10, 20, 30, 50, 75, 100, np.inf]


def metrics(actual, predicted):
    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted)),
    )


print("Loading regional data...", flush=True)
df = pd.read_csv(DATA_PATH, parse_dates=["date", "timestamp"])
df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

print("Loading existing LSTM and scaler...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Calculate road activity using training-period observations only.
train_mask = df["timestamp"] < TRAIN_END
train_means = (
    df.loc[train_mask]
    .groupby("segment_id")["probe_count"]
    .mean()
    .rename("training_mean_probe_count")
)

# Apply the existing training scaler; do not fit a new scaler.
scaled = scaler.transform(df[FEATURES])
scaled_df = df[["segment_id", "timestamp"]].copy()

for i, feature in enumerate(FEATURES):
    scaled_df[feature] = scaled[:, i]

scaled_df["actual_probe_count"] = df["probe_count"].to_numpy()

records = []
windows = []

print("Preparing validation sequences...", flush=True)

for segment_id, group in scaled_df.groupby("segment_id", sort=False):
    group = group.sort_values("timestamp").reset_index(drop=True)

    values = group[FEATURES].to_numpy(dtype=np.float32)
    actual = group["actual_probe_count"].to_numpy(dtype=np.float32)
    timestamps = group["timestamp"]

    for target_idx in range(LOOKBACK, len(group)):
        target_time = timestamps.iloc[target_idx]

        if TRAIN_END <= target_time < VAL_END:
            windows.append(values[target_idx - LOOKBACK:target_idx])
            records.append({
                "segment_id": str(segment_id),
                "forecast_timestamp": target_time,
                "actual_probe_count": float(actual[target_idx]),
                "naive_prediction": float(actual[target_idx - 1]),
                "training_mean_probe_count": float(
                    train_means.loc[segment_id]
                ),
            })

if not records:
    raise ValueError("No validation predictions were prepared.")

result = pd.DataFrame(records)

print(f"Predicting {len(result):,} validation observations...", flush=True)
pred_scaled = model.predict(
    np.asarray(windows, dtype=np.float32),
    batch_size=512,
    verbose=1,
).ravel()

probe_idx = FEATURES.index("probe_count")
result["lstm_prediction"] = (
    pred_scaled * scaler.scale_[probe_idx] + scaler.mean_[probe_idx]
)
result["lstm_prediction"] = result["lstm_prediction"].clip(lower=0)

actual = result["actual_probe_count"].to_numpy()
naive = result["naive_prediction"].to_numpy()
lstm = result["lstm_prediction"].to_numpy()

# Save individual predictions for reproducibility.
pred_path = OUTPUT_DIR / "regional_validation_predictions.csv"
result.to_csv(pred_path, index=False)

# Compare all methods on exactly the same validation observations.
rows = []

for name, prediction in [
    ("naive_only", naive),
    ("lstm_only", lstm),
]:
    mae, rmse = metrics(actual, prediction)
    rows.append({
        "method": name,
        "threshold": np.nan,
        "MAE": mae,
        "RMSE": rmse,
        "lstm_road_count": (
            0 if name == "naive_only"
            else result["segment_id"].nunique()
        ),
    })

for threshold in THRESHOLDS:
    use_lstm = (
        result["training_mean_probe_count"].to_numpy() >= threshold
    )
    hybrid = np.where(use_lstm, lstm, naive)
    mae, rmse = metrics(actual, hybrid)

    rows.append({
        "method": "hybrid",
        "threshold": threshold,
        "MAE": mae,
        "RMSE": rmse,
        "lstm_road_count": result.loc[
            result["training_mean_probe_count"] >= threshold,
            "segment_id",
        ].nunique(),
    })

summary = pd.DataFrame(rows)
summary_path = OUTPUT_DIR / "regional_hybrid_validation_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n===== VALIDATION RESULTS: AUG 24–25, 2024 =====")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

hybrids = summary[summary["method"] == "hybrid"]
best = hybrids.loc[hybrids["MAE"].idxmin()]
naive_mae = summary.loc[
    summary["method"] == "naive_only", "MAE"
].iloc[0]
lstm_mae = summary.loc[
    summary["method"] == "lstm_only", "MAE"
].iloc[0]

print("\n===== BEST HYBRID ON VALIDATION MAE =====")
print("Threshold:", best["threshold"])
print("Validation MAE:", round(best["MAE"], 4))
print("Validation RMSE:", round(best["RMSE"], 4))
print("Roads using LSTM:", int(best["lstm_road_count"]))
print(f"Naive-only MAE: {naive_mae:.4f}")
print(f"LSTM-only MAE:  {lstm_mae:.4f}")

print("\nSaved predictions:", pred_path)
print("Saved comparison:", summary_path)
print("Existing model and test prediction files were not modified.")
