import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_multi_road_data import prepare_data


# -----------------------------------
# 1. LOAD DATA
# -----------------------------------

(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    probe_scaler,
    lag_scaler,
    road_scaler
) = prepare_data()


print("-----------------------------------")
print("TRAINING ENHANCED MULTI-ROAD LSTM")
print("-----------------------------------")

print("\nTraining data:", X_train.shape)
print("Validation data:", X_validation.shape)
print("Test data:", X_test.shape)


# -----------------------------------
# 2. BUILD MODEL
# -----------------------------------

model = Sequential([
    Input(shape=(24, 9)),
    LSTM(64),
    Dense(1)
])


# -----------------------------------
# 3. COMPILE
# -----------------------------------

model.compile(
    optimizer="adam",
    loss="mse"
)


model.summary()


# -----------------------------------
# 4. EARLY STOPPING
# -----------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)


# -----------------------------------
# 5. TRAIN
# -----------------------------------

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


# -----------------------------------
# 6. PREDICTIONS
# -----------------------------------

predictions_scaled = model.predict(
    X_test
)


# -----------------------------------
# 7. INVERSE SCALE
# -----------------------------------

predictions = probe_scaler.inverse_transform(
    predictions_scaled
)

actual = probe_scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# -----------------------------------
# 8. METRICS
# -----------------------------------

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


# -----------------------------------
# 9. RESULTS
# -----------------------------------

print("\n-----------------------------------")
print("ENHANCED MULTI-ROAD LSTM RESULTS")
print("-----------------------------------")

print("\nMAE :", mae)
print("RMSE:", rmse)

print(
    "\nNumber of test predictions:",
    len(actual)
)


# -----------------------------------
# 10. SAMPLE PREDICTIONS
# -----------------------------------

print("\nFirst 10 predictions:")

for i in range(10):

    print(
        f"Actual: {actual[i][0]:.2f} "
        f"| Predicted: {predictions[i][0]:.2f}"
    )