import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("traffic_all.csv")

df["date"] = pd.to_datetime(df["date"])


# --------------------------------------------------
# 2. Select the same road used by the LSTM
# --------------------------------------------------

segment_id = -13560179880493

df = df[df["segment_id"] == segment_id].copy()

df = df.sort_values(
    ["date", "hour"]
).reset_index(drop=True)


# --------------------------------------------------
# 3. Create previous-hour prediction
# --------------------------------------------------
# The prediction for hour t is simply:
#
# prediction(t) = probe_count(t - 1)
#
# Because we create this BEFORE selecting the test
# period, the first test prediction can correctly use
# the final hour of the previous day.
# --------------------------------------------------

df["prediction"] = df["probe_count"].shift(1)


# --------------------------------------------------
# 4. Define test period
# --------------------------------------------------

test_start = pd.Timestamp("2024-08-26")
test_end = pd.Timestamp("2024-08-30")

test = df[
    (df["date"] >= test_start) &
    (df["date"] <= test_end)
].copy()


# --------------------------------------------------
# 5. Remove rows where previous-hour value is missing
# --------------------------------------------------

test = test.dropna(
    subset=["prediction"]
)


# --------------------------------------------------
# 6. Calculate MAE
# --------------------------------------------------

mae = mean_absolute_error(
    test["probe_count"],
    test["prediction"]
)


# --------------------------------------------------
# 7. Calculate RMSE
# --------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        test["probe_count"],
        test["prediction"]
    )
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("-----------------------------------")
print("BASELINE TEST RESULTS")
print("-----------------------------------")

print("MAE :", mae)
print("RMSE:", rmse)

print(
    "\nNumber of test predictions:",
    len(test)
)


# --------------------------------------------------
# 9. Show a few predictions for verification
# --------------------------------------------------

print("\nFirst 10 predictions:")

print(
    test[
        [
            "date",
            "hour",
            "probe_count",
            "prediction"
        ]
    ].head(10).to_string(index=False)
)