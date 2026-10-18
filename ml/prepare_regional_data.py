
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_PATH = "traffic_all.csv"
REGION_PATH = "ml/candidate_region_edges.csv"
OUTPUT_DIR = Path("ml/regional_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load regional road IDs
region = pd.read_csv(REGION_PATH)
region_ids = set(region["segment_id"].astype(str))

print("Regional road segments:", len(region_ids))

# Load traffic data
traffic = pd.read_csv(
    DATA_PATH,
    dtype={"segment_id": str},
    parse_dates=["date"]
)

traffic["segment_id"] = traffic["segment_id"].str.strip()
traffic = traffic[traffic["segment_id"].isin(region_ids)].copy()

print("Filtered traffic rows:", len(traffic))
print("Unique roads found:", traffic["segment_id"].nunique())

# Verify the expected historical coverage
coverage = traffic.groupby("segment_id").size()

print("Roads with 480 observations:", (coverage == 480).sum())
print("Roads with unexpected observations:", (coverage != 480).sum())

# Add time features
traffic["timestamp"] = (
    traffic["date"] + pd.to_timedelta(traffic["hour"], unit="h")
)

traffic = traffic.sort_values(
    ["segment_id", "timestamp"]
).reset_index(drop=True)

traffic["hour_sin"] = np.sin(2 * np.pi * traffic["hour"] / 24)
traffic["hour_cos"] = np.cos(2 * np.pi * traffic["hour"] / 24)
traffic["day_of_week"] = traffic["date"].dt.dayofweek

# Save the filtered historical data
output_path = OUTPUT_DIR / "regional_traffic.csv"
traffic.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("Final shape:", traffic.shape)
print("Date range:", traffic["date"].min(), "to", traffic["date"].max())
print("\nSample:")
print(traffic.head())
