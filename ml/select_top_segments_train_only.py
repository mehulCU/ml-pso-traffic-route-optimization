import pandas as pd

print("-----------------------------------")
print("SELECTING TOP 100 ROADS")
print("TRAINING PERIOD ONLY")
print("-----------------------------------")

df = pd.read_csv("traffic_all.csv")

df["date"] = pd.to_datetime(df["date"])

# Use ONLY training period
train = df[
    df["date"] <= pd.Timestamp("2024-08-23")
].copy()

# Calculate average probe count per road using training data only
top_segments = (
    train.groupby("segment_id")["probe_count"]
    .mean()
    .sort_values(ascending=False)
    .head(100)
    .reset_index()
)

top_segments.columns = [
    "segment_id",
    "train_mean_probe_count"
]

# Save
top_segments.to_csv(
    "ml/top_100_segments_train_only.csv",
    index=False
)

print("\nTop 100 roads selected.")
print("Number of roads:", len(top_segments))

print("\nTop 20:")
print(top_segments.head(20).to_string(index=False))

print("\nLowest selected road:")
print(top_segments.tail(1).to_string(index=False))

print("\nSaved to:")
print("ml/top_100_segments_train_only.csv")
