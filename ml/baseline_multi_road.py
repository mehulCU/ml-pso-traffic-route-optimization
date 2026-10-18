import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1. Load traffic data
# -----------------------------
df = pd.read_csv("traffic_all.csv")

df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# 2. Load the same 100 roads
# -----------------------------
top_segments = pd.read_csv("ml/top_100_segments_train_only.csv")

segment_ids = top_segments["segment_id"].tolist()

df = df[df["segment_id"].isin(segment_ids)].copy()

# Sort separately by road and time
df = df.sort_values(
    ["segment_id", "date", "hour"]
).reset_index(drop=True)

# -----------------------------
# 3. Naive prediction
# -----------------------------
# For each road:
# next hour prediction = previous hour's actual probe count

df["prediction"] = (
    df.groupby("segment_id")["probe_count"]
      .shift(1)
)

# -----------------------------
# 4. Select the same test period
# -----------------------------
test_start = pd.Timestamp("2024-08-26")
test_end = pd.Timestamp("2024-08-30")

test = df[
    (df["date"] >= test_start) &
    (df["date"] <= test_end)
].copy()

# Remove the first unavailable prediction if any
test = test.dropna(subset=["prediction"])

# -----------------------------
# 5. Calculate metrics
# -----------------------------
mae = mean_absolute_error(
    test["probe_count"],
    test["prediction"]
)

rmse = np.sqrt(
    mean_squared_error(
        test["probe_count"],
        test["prediction"]
    )
)

# -----------------------------
# 6. Print results
# -----------------------------
print("-----------------------------------")
print("100-ROAD NAIVE BASELINE")
print("-----------------------------------")

print("MAE :", mae)
print("RMSE:", rmse)

print("\nNumber of predictions:", len(test))
print("Number of roads:", test["segment_id"].nunique())

print("\nExpected predictions: 12,000")

print("\nFirst 10 predictions:")
print(
    test[
        [
            "segment_id",
            "date",
            "hour",
            "probe_count",
            "prediction"
        ]
    ].head(10).to_string(index=False)
)
