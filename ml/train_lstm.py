import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

from prepare_lstm_data import prepare_data


# Prepare data
(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    scaler
) = prepare_data()


# Build model
model = Sequential([
    Input(shape=(24, 4)),
    LSTM(50),
    Dense(1)
])


# Compile model
model.compile(
    optimizer="adam",
    loss="mse"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)
# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

predictions_scaled = model.predict(X_test)

# Convert predictions back to original probe-count scale
predictions = scaler.inverse_transform(
    predictions_scaled
)

actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# -----------------------------------
# Calculate evaluation metrics
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


print("\n-----------------------------------")
print("LSTM TEST RESULTS")
print("-----------------------------------")

print("MAE :", mae)
print("RMSE:", rmse)