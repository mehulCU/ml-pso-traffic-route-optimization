from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


# Resolve paths relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "regional_model" / "regional_lstm.keras"
SCALER_PATH = PROJECT_ROOT / "ml" / "regional_model" / "regional_scaler.pkl"

FEATURES = [
    "probe_count",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "speed_limit",
    "frc",
    "distance",
]
LOOKBACK_HOURS = 24


# Load the existing trained model and scaler once.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)


def predict_next_probe_count(history: list[dict]) -> float:
    """Predict the next probe count from 24 hourly observations."""

    if len(history) != LOOKBACK_HOURS:
        raise ValueError(
            f"Exactly {LOOKBACK_HOURS} hourly observations are required."
        )

    rows = []

    for index, observation in enumerate(history):
        missing = [feature for feature in FEATURES if feature not in observation]

        if missing:
            raise ValueError(
                f"Observation {index} is missing features: {missing}"
            )

        try:
            row = [float(observation[feature]) for feature in FEATURES]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Observation {index} contains a non-numeric feature."
            ) from exc

        if not np.all(np.isfinite(row)):
            raise ValueError(
                f"Observation {index} contains a non-finite value."
            )

        rows.append(row)

    # Shape: (24, 7)
    feature_array = np.asarray(rows, dtype=np.float32)

    # Scale the same seven features used during training.
    scaled_features = scaler.transform(feature_array)

    # Shape: (1, 24, 7) — one prediction sequence.
    model_input = scaled_features[np.newaxis, :, :]

    scaled_prediction = model.predict(model_input, verbose=0)

    # Inverse-transform only the probe_count feature.
    # StandardScaler inverse transform for feature 0:
    # original = scaled * scale_[0] + mean_[0]
    prediction = (
        float(scaled_prediction[0, 0]) * float(scaler.scale_[0])
        + float(scaler.mean_[0])
    )

    # Probe counts cannot be negative.
    return max(0.0, prediction)
