import pandas as pd
from pathlib import Path

BASE_DIR = Path("ml/regional_model")

FORECAST_PATH = BASE_DIR / "next_hour_forecasts.csv"
ROUTING_PATH = BASE_DIR / "regional_routing_edges.csv"

forecasts = pd.read_csv(
    FORECAST_PATH,
    dtype={"segment_id": str}
)

edges = pd.read_csv(
    ROUTING_PATH,
    dtype={"segment_id": str}
)

# Normalize IDs to avoid accidental whitespace mismatches.
forecast_ids = set(forecasts["segment_id"].str.strip())
routing_ids = set(edges["segment_id"].str.strip())

matched_ids = forecast_ids & routing_ids

print("========== REGIONAL LSTM COVERAGE ==========")
print(f"Forecast records: {len(forecasts)}")
print(f"Unique forecast segments: {len(forecast_ids)}")
print(f"Routing edge records: {len(edges)}")
print(f"Unique routing segments: {len(routing_ids)}")
print(f"Matching segment IDs: {len(matched_ids)}")

coverage = (
    100 * len(matched_ids) / len(routing_ids)
    if routing_ids else 0
)

print(f"Routing forecast coverage: {coverage:.2f}%")

missing = sorted(routing_ids - forecast_ids)

if missing:
    print(f"\nSegments without forecasts: {len(missing)}")
    print("First 10 missing IDs:", missing[:10])
else:
    print("\nEvery regional routing segment has a forecast.")

# Check the actual routing graph used in our previous experiments.
# These endpoints are graph nodes, not road-segment IDs.
route_nodes = {
    "Route 1": (11388, 14979),
    "Route 2": (11388, 14979),
    "Route 3": (11388, 14979),
}

print("\n========== ROUTE FORECAST AVAILABILITY ==========")

for name, (source, destination) in route_nodes.items():
    print(
        f"{name}: node {source} -> node {destination}; "
        "edge-level coverage is evaluated from the routing graph."
    )