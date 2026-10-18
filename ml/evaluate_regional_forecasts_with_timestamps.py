import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("ml/regional_data/regional_traffic.csv")
MODEL_PATH = Path("ml/regional_model/regional_lstm.keras")
SCALER_PATH = Path("ml/regional_model/regional_scaler.pkl")
OUTPUT_PATH = Path(
    "ml/regional_model/regional_timestamped_test_predictions.csv"
)

LOOKBACK = 24
TEST_START = pd.Timestamp("2024-08-26 00:00:00")
BATCH_SIZE = 4096

FEATURES = [
    "probe_count",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "speed_limit",
    "frc",
    "distance",
]


def main():
    print("Loading dataset...", flush=True)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
        dtype={"segment_id": str},
    )

    df = df.sort_values(
        ["segment_id", "timestamp"]
    ).reset_index(drop=True)

    print("Loading trained model and scaler...", flush=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    probe_idx = FEATURES.index("probe_count")
    output_columns = [
        "segment_id",
        "forecast_origin",
        "forecast_timestamp",
        "actual_probe_count",
        "naive_prediction",
        "predicted_probe_count",
    ]

    # Process one road segment at a time to limit memory usage.
    result_parts = []
    total_predictions = 0

    print("Preparing and predicting test sequences...", flush=True)

    for road_number, (segment_id, group) in enumerate(
        df.groupby("segment_id", sort=False), start=1
    ):
        group = group.sort_values("timestamp").reset_index(drop=True)

        timestamps = group["timestamp"].to_numpy()
        actual = group["probe_count"].to_numpy(dtype=np.float32)
        values = group[FEATURES].to_numpy(dtype=np.float64)

        scaled = scaler.transform(
            pd.DataFrame(values, columns=FEATURES)
        ).astype(np.float32)

        # Sliding windows: each prediction uses the previous 24 hours.
        if len(group) <= LOOKBACK:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(
            scaled, window_shape=LOOKBACK, axis=0
        )

        # Shape from sliding_window_view is (samples, features, lookback).
        windows = np.moveaxis(windows, -1, 1)

        # Window ending at t predicts the observation at t.
        target_indices = np.arange(LOOKBACK, len(group))

        # Require consecutive hourly observations throughout each window.
        valid = timestamps[target_indices] >= np.datetime64(TEST_START)

        valid &= (
            timestamps[target_indices]
            - timestamps[target_indices - 1]
            == np.timedelta64(1, "h")
        )

        for i in range(LOOKBACK, len(group)):
            if not valid[i - LOOKBACK]:
                continue

            start = i - LOOKBACK
            if not np.all(
                np.diff(timestamps[start:i])
                == np.timedelta64(1, "h")
            ):
                valid[i - LOOKBACK] = False

        selected_indices = target_indices[valid]

        if len(selected_indices) == 0:
            continue

        X = windows[selected_indices - LOOKBACK]

        predictions_scaled = model.predict(
            X, batch_size=BATCH_SIZE, verbose=0
        ).reshape(-1)

        predictions = (
            predictions_scaled * scaler.scale_[probe_idx]
            + scaler.mean_[probe_idx]
        )
        predictions = np.maximum(predictions, 0)

        part = pd.DataFrame({
            "segment_id": segment_id,
            "forecast_origin": timestamps[selected_indices - 1],
            "forecast_timestamp": timestamps[selected_indices],
            "actual_probe_count": actual[selected_indices],
            "naive_prediction": actual[selected_indices - 1],
            "predicted_probe_count": predictions,
        })

        result_parts.append(part)
        total_predictions += len(part)

        if road_number % 100 == 0:
            print(
                f"Processed {road_number} roads; "
                f"{total_predictions:,} predictions prepared.",
                flush=True,
            )

    if not result_parts:
        raise ValueError("No test predictions were generated.")

    result = pd.concat(result_parts, ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)

    lstm_mae = mean_absolute_error(
        result["actual_probe_count"],
        result["predicted_probe_count"],
    )
    lstm_rmse = np.sqrt(mean_squared_error(
        result["actual_probe_count"],
        result["predicted_probe_count"],
    ))

    naive_mae = mean_absolute_error(
        result["actual_probe_count"],
        result["naive_prediction"],
    )
    naive_rmse = np.sqrt(mean_squared_error(
        result["actual_probe_count"],
        result["naive_prediction"],
    ))

    print("\n========== TIMESTAMPED TEST RESULTS ==========")
    print(f"Predictions: {len(result):,}")
    print(f"Unique roads: {result['segment_id'].nunique():,}")
    print(
        "Unique target hours:",
        result["forecast_timestamp"].nunique(),
    )
    print(f"LSTM MAE:  {lstm_mae:.4f}")
    print(f"LSTM RMSE: {lstm_rmse:.4f}")
    print(f"Naive MAE: {naive_mae:.4f}")
    print(f"Naive RMSE: {naive_rmse:.4f}")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()