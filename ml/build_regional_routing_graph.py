
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

REGION_PATH = "ml/candidate_region_edges.csv"
TRAFFIC_PATH = "ml/regional_data/regional_traffic.csv"
FORECAST_PATH = "ml/regional_model/next_hour_forecasts.csv"
OUTPUT_DIR = Path("ml/regional_model")

# Traffic penalty strength; an initial experimental setting
ALPHA = 0.25

# Load the regional graph, history and forecasts
edges = pd.read_csv(REGION_PATH, dtype={"segment_id": str})
traffic = pd.read_csv(
    TRAFFIC_PATH,
    parse_dates=["date"],
    dtype={"segment_id": str},
)
forecasts = pd.read_csv(
    FORECAST_PATH,
    dtype={"segment_id": str},
)

# Historical baseline uses training dates only
train = traffic[traffic["date"] < "2024-08-24"]

historical_mean = (
    train.groupby("segment_id")["probe_count"]
    .mean()
    .rename("historical_mean_probe_count")
)

# Merge all data by road-segment ID
edges = edges.merge(
    historical_mean,
    on="segment_id",
    how="left",
    validate="one_to_one",
)

edges = edges.merge(
    forecasts[["segment_id", "predicted_probe_count"]],
    on="segment_id",
    how="left",
    validate="one_to_one",
)

# Validate required data
print("Regional road segments:", len(edges))
print("Missing historical baselines:",
      edges["historical_mean_probe_count"].isna().sum())
print("Missing forecasts:",
      edges["predicted_probe_count"].isna().sum())
print("Invalid speed limits:",
      (edges["speed_limit"] <= 0).sum())

if edges[
    ["historical_mean_probe_count", "predicted_probe_count"]
].isna().any().any():
    raise ValueError("Missing historical values or forecasts.")

if (edges["speed_limit"] <= 0).any():
    raise ValueError("Some roads have non-positive speed limits.")

# Estimate free-flow travel time in minutes.
# Assumes distance is in metres and speed_limit is in km/h.
edges["free_flow_minutes"] = (
    edges["distance"] * 0.06 / edges["speed_limit"]
)

# Normalize each road's forecast against its own training-period mean.
# This is an activity proxy, not a measured congestion or delay value.
denominator = edges["historical_mean_probe_count"].clip(lower=1.0)

edges["traffic_ratio"] = (
    edges["predicted_probe_count"] / denominator
)

# Apply a penalty only when predicted activity exceeds its own baseline.
edges["traffic_penalty_factor"] = (
    1 + ALPHA * (edges["traffic_ratio"] - 1).clip(lower=0)
)

edges["traffic_aware_cost_minutes"] = (
    edges["free_flow_minutes"] * edges["traffic_penalty_factor"]
)

# Save enriched edge table
output_path = OUTPUT_DIR / "regional_routing_edges.csv"
edges.to_csv(output_path, index=False)

# Build a MultiGraph so parallel road segments retain their identities.
# Each edge is keyed by its segment ID.
G = nx.MultiGraph()

for row in edges.itertuples(index=False):
    G.add_edge(
        row.start_node,
        row.end_node,
        key=row.segment_id,
        segment_id=row.segment_id,
        distance=float(row.distance),
        free_flow_minutes=float(row.free_flow_minutes),
        predicted_probe_count=float(row.predicted_probe_count),
        traffic_ratio=float(row.traffic_ratio),
        traffic_aware_cost_minutes=float(
            row.traffic_aware_cost_minutes
        ),
    )

print("\n========== ROUTING GRAPH REPORT ==========")
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())
print("Connected components:", nx.number_connected_components(G))
print("Alpha:", ALPHA)
print("\nFree-flow minutes summary:")
print(edges["free_flow_minutes"].describe())
print("\nTraffic ratio summary:")
print(edges["traffic_ratio"].describe())
print("\nTraffic-aware cost summary:")
print(edges["traffic_aware_cost_minutes"].describe())

print("\nSaved enriched roads:", output_path)
