import pandas as pd
import numpy as np

df = pd.read_csv("traffic_all.csv")

# Select one road
df = df[df["segment_id"] == -13560179880493]

# Sort chronologically
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "hour"]).reset_index(drop=True)

values = df["probe_count"].values

time_steps = 3

X = []
y = []

for i in range(len(values) - time_steps):
    X.append(values[i:i + time_steps])
    y.append(values[i + time_steps])

X = np.array(X)
y = np.array(y)

# Add feature dimension for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst sequence:")
print(X[0])

print("\nFirst target:")
print(y[0])

print("\nSecond sequence:")
print(X[1])

print("\nSecond target:")
print(y[1])
