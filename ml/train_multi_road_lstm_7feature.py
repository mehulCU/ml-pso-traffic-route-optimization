import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_multi_road_data_7feature import prepare_data


# Load data
(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    probe_scaler,
    road_scaler
) = prepare_data()


print("-----------------------------------")
print("TRAINING 7-FEATURE MULTI-ROAD LSTM")
print("-----------------------------------")

print("\nTraining data:", X_train.shape)
print("Validation data:", X_validation.shape)
print("Test data:", X_test.shape)


# Model
model = Sequential([
    Input(shape=(24, 7)),
    LSTM(64),
    Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mse"
)


model.summary()


# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)


# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation
    ),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# Predict
predictions_scaled = model.predict(X_test)


# Convert back to probe counts
predictions = probe_scaler.inverse_transform(
    predictions_scaled
)

actual = probe_scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# Metrics
mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


print("\n-----------------------------------")
print("7-FEATURE MULTI-ROAD LSTM RESULTS")
print("-----------------------------------")

print("\nMAE :", mae)
print("RMSE:", rmse)

print(
    "\nNumber of test predictions:",
    len(actual)
)


print("\nFirst 10 predictions:")

for i in range(10):

    print(
        f"Actual: {actual[i][0]:.2f} "
        f"| Predicted: {predictions[i][0]:.2f}"
    )