
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from pathlib import Path

DATA_PATH = "ml/regional_data/regional_traffic.csv"
MODEL_PATH = "ml/regional_model/regional_lstm.keras"
SCALER_PATH = "ml/regional_model/regional_scaler.pkl"
OUTPUT_PATH = "ml/regional_model/next_hour_forecasts.csv"

LOOKBACK = 24

FEATURES = [
    "probe_count",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "speed_limit",
    "frc",
    "distance",
]

# Load trained artifacts
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date", "timestamp"]
)
df = df.sort_values(["segment_id", "timestamp"])

# Forecast one hour after the latest available observation
latest_timestamp = df["timestamp"].max()
forecast_timestamp = latest_timestamp + pd.Timedelta(hours=1)

print("Latest observation:", latest_timestamp)
print("Forecast timestamp:", forecast_timestamp)

forecasts = []

for segment_id, group in df.groupby("segment_id", sort=False):
    group = group.sort_values("timestamp")

    # Take the most recent 24 observations
    recent = group.tail(LOOKBACK)

    if len(recent) != LOOKBACK:
        print(f"Skipping {segment_id}: insufficient history")
        continue

    # Ensure observations are consecutive hourly records
    time_differences = recent["timestamp"].diff().dropna()

    if not (time_differences == pd.Timedelta(hours=1)).all():
        print(f"Skipping {segment_id}: missing hourly observations")
        continue

    # Apply the same scaler used during training
    X = scaler.transform(recent[FEATURES])
    X = X.astype(np.float32).reshape(1, LOOKBACK, len(FEATURES))

    # Predict standardized probe count
    pred_scaled = float(model.predict(X, verbose=0)[0, 0])

    # Convert back to original probe-count units
    probe_index = FEATURES.index("probe_count")
    predicted_count = (
        pred_scaled * scaler.scale_[probe_index]
        + scaler.mean_[probe_index]
    )

    # Probe counts cannot be negative
    predicted_count = max(0.0, predicted_count)

    forecasts.append({
        "segment_id": str(segment_id),
        "forecast_timestamp": forecast_timestamp,
        "predicted_probe_count": predicted_count,
    })

result = pd.DataFrame(forecasts)

result.to_csv(OUTPUT_PATH, index=False)

print("\nForecasts generated:", len(result))
print("Unique road segments:", result["segment_id"].nunique())

if not result.empty:
    print("\nForecast summary:")
    print(result["predicted_probe_count"].describe())

print("\nSaved:", OUTPUT_PATH)
