
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = "ml/regional_data/regional_traffic.csv"
OUTPUT_DIR = Path("ml/regional_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date", "timestamp"])
df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

# Chronological split
train_end = pd.Timestamp("2024-08-24")
val_end = pd.Timestamp("2024-08-26")

train_mask = df["timestamp"] < train_end
val_mask = (
    (df["timestamp"] >= train_end)
    & (df["timestamp"] < val_end)
)
test_mask = df["timestamp"] >= val_end

print("Rows:", len(df))
print("Roads:", df["segment_id"].nunique())
print("Training rows:", train_mask.sum())
print("Validation rows:", val_mask.sum())
print("Test rows:", test_mask.sum())

# Fit scaler on training observations only
scaler = StandardScaler()
scaler.fit(df.loc[train_mask, FEATURES])

scaled = scaler.transform(df[FEATURES])
df_scaled = df[["segment_id", "timestamp"]].copy()

for i, feature in enumerate(FEATURES):
    df_scaled[feature] = scaled[:, i]

# Preserve original probe counts for evaluation
df_scaled["actual_probe_count"] = df["probe_count"].to_numpy()

# Create sequences independently per road
X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []
naive_test = []

for segment_id, group in df_scaled.groupby("segment_id", sort=False):
    group = group.sort_values("timestamp").reset_index(drop=True)

    values = group[FEATURES].to_numpy(dtype=np.float32)
    actual = group["actual_probe_count"].to_numpy(dtype=np.float32)
    timestamps = group["timestamp"]

    for target_idx in range(LOOKBACK, len(group)):
        target_time = timestamps.iloc[target_idx]
        x = values[target_idx - LOOKBACK:target_idx]
        y = values[target_idx, 0]

        if target_time < train_end:
            X_train.append(x)
            y_train.append(y)

        elif target_time < val_end:
            X_val.append(x)
            y_val.append(y)

        else:
            X_test.append(x)
            y_test.append(y)
            # Previous hour's observed count is the naive forecast
            naive_test.append(actual[target_idx - 1])

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)
X_val = np.asarray(X_val, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)
naive_test = np.asarray(naive_test, dtype=np.float32)

print("\nSequence shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# Build model
model = Sequential([
    Input(shape=(LOOKBACK, len(FEATURES))),
    LSTM(64),
    Dense(1),
])

model.compile(optimizer="adam", loss="mse")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1,
)

# Predict in scaled space, then invert probe-count scaling
pred_scaled = model.predict(X_test, batch_size=512, verbose=1).ravel()

probe_scaler_index = FEATURES.index("probe_count")
probe_mean = scaler.mean_[probe_scaler_index]
probe_scale = scaler.scale_[probe_scaler_index]

predictions = pred_scaled * probe_scale + probe_mean
actuals = y_test * probe_scale + probe_mean

# Metrics
lstm_mae = mean_absolute_error(actuals, predictions)
lstm_rmse = np.sqrt(mean_squared_error(actuals, predictions))

naive_mae = mean_absolute_error(actuals, naive_test)
naive_rmse = np.sqrt(mean_squared_error(actuals, naive_test))

print("\n========== REGIONAL MODEL RESULTS ==========")
print(f"Test roads: {df['segment_id'].nunique()}")
print(f"LSTM MAE:  {lstm_mae:.4f}")
print(f"LSTM RMSE: {lstm_rmse:.4f}")
print(f"Naive MAE: {naive_mae:.4f}")
print(f"Naive RMSE: {naive_rmse:.4f}")

print(
    f"MAE improvement: "
    f"{(naive_mae - lstm_mae) / naive_mae * 100:.2f}%"
)
print(
    f"RMSE improvement: "
    f"{(naive_rmse - lstm_rmse) / naive_rmse * 100:.2f}%"
)

# Save artifacts separately from the existing model
model.save(OUTPUT_DIR / "regional_lstm.keras")

import joblib
joblib.dump(scaler, OUTPUT_DIR / "regional_scaler.pkl")

results = pd.DataFrame({
    "actual_probe_count": actuals,
    "lstm_prediction": predictions,
    "naive_prediction": naive_test,
})

results.to_csv(OUTPUT_DIR / "regional_test_predictions.csv", index=False)

print("\nSaved model and evaluation artifacts to:", OUTPUT_DIR)
