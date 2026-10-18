import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("ml/regional_model")

LSTM_FILE = BASE / "rolling_horizon_route_segments_smoothed.csv"
HYBRID_FILE = BASE / "rolling_horizon_route_segments_hybrid.csv"
PRED_FILE = BASE / "regional_timestamped_test_predictions.csv"
EDGES_FILE = BASE / "regional_routing_edges.csv"

ALPHA = 0.25
SMOOTHING = 5.0

# Load route selections, observed test counts, and segment properties.
lstm = pd.read_csv(LSTM_FILE)
hybrid = pd.read_csv(HYBRID_FILE)
actual = pd.read_csv(PRED_FILE)
edges = pd.read_csv(EDGES_FILE)

print("Route types in LSTM file:", sorted(lstm["route_type"].unique()))
print("Route types in hybrid file:", sorted(hybrid["route_type"].unique()))

# Actual probe count is used only for retrospective evaluation.
actual = actual[
    ["segment_id", "forecast_timestamp", "actual_probe_count"]
].copy()

edges = edges[
    [
        "segment_id",
        "distance",
        "speed_limit",
        "historical_mean_probe_count",
    ]
].drop_duplicates("segment_id")

def evaluate_routes(routes, model_name):
    routes = routes.copy()
    routes["forecast_timestamp"] = pd.to_datetime(
        routes["forecast_timestamp"]
    )
    actual_copy = actual.copy()
    actual_copy["forecast_timestamp"] = pd.to_datetime(
        actual_copy["forecast_timestamp"]
    )

    # Join each chosen road segment to its observed count for that hour.
    routes = routes.merge(
        actual_copy,
        on=["segment_id", "forecast_timestamp"],
        how="left",
        validate="many_to_one",
    )
    routes = routes.merge(
        edges,
        on="segment_id",
        how="left",
        validate="many_to_one",
    )

    routes["free_flow_minutes"] = (
        routes["distance"] * 0.06 / routes["speed_limit"]
    )

    # Apply the same cost formula to both sets of selected routes.
    routes["observed_ratio"] = (
        routes["actual_probe_count"] + SMOOTHING
    ) / (
        routes["historical_mean_probe_count"] + SMOOTHING
    )

    routes["observed_penalty"] = (
        1 + ALPHA * (routes["observed_ratio"] - 1).clip(lower=0)
    )
    routes["observed_proxy_cost_minutes"] = (
        routes["free_flow_minutes"] * routes["observed_penalty"]
    )

    keys = [
        "forecast_timestamp",
        "source",
        "destination",
        "route_type",
    ]

    summary = routes.groupby(keys, as_index=False).agg(
        segment_count=("segment_id", "size"),
        missing_actual_counts=("actual_probe_count", lambda x: x.isna().sum()),
        missing_edge_properties=("free_flow_minutes", lambda x: x.isna().sum()),
        free_flow_minutes=("free_flow_minutes", "sum"),
        observed_proxy_cost_minutes=("observed_proxy_cost_minutes", "sum"),
    )

    summary["model"] = model_name
    summary["complete"] = (
        (summary["missing_actual_counts"] == 0)
        & (summary["missing_edge_properties"] == 0)
    )

    return routes, summary

lstm_rows, lstm_summary = evaluate_routes(lstm, "LSTM_smoothed")
hybrid_rows, hybrid_summary = evaluate_routes(hybrid, "Hybrid")

all_summary = pd.concat(
    [lstm_summary, hybrid_summary], ignore_index=True
)

summary_path = BASE / "routes_evaluated_on_actual_counts.csv"
all_summary.to_csv(summary_path, index=False)

# Compare matching route types, timestamps, and source/destination pairs.
keys = [
    "forecast_timestamp",
    "source",
    "destination",
    "route_type",
]

left = lstm_summary[
    keys + ["observed_proxy_cost_minutes", "complete"]
].rename(columns={
    "observed_proxy_cost_minutes": "lstm_actual_proxy_cost",
    "complete": "lstm_complete",
})

right = hybrid_summary[
    keys + ["observed_proxy_cost_minutes", "complete"]
].rename(columns={
    "observed_proxy_cost_minutes": "hybrid_actual_proxy_cost",
    "complete": "hybrid_complete",
})

comparison = left.merge(
    right, on=keys, how="inner", validate="one_to_one"
)

comparison = comparison[
    comparison["lstm_complete"] & comparison["hybrid_complete"]
].copy()

comparison["hybrid_minus_lstm_minutes"] = (
    comparison["hybrid_actual_proxy_cost"]
    - comparison["lstm_actual_proxy_cost"]
)

comparison["hybrid_lower_cost"] = (
    comparison["hybrid_minus_lstm_minutes"] < -1e-9
)
comparison["lstm_lower_cost"] = (
    comparison["hybrid_minus_lstm_minutes"] > 1e-9
)
comparison["equal_cost"] = (
    comparison["hybrid_minus_lstm_minutes"].abs() <= 1e-9
)

comparison_path = BASE / "routes_common_cost_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\n=== DATA COVERAGE ===")
print("LSTM route rows:", len(lstm))
print("Hybrid route rows:", len(hybrid))
print("LSTM missing actual counts:", lstm_rows["actual_probe_count"].isna().sum())
print("Hybrid missing actual counts:", hybrid_rows["actual_probe_count"].isna().sum())
print("LSTM missing edge properties:", lstm_rows["free_flow_minutes"].isna().sum())
print("Hybrid missing edge properties:", hybrid_rows["free_flow_minutes"].isna().sum())

print("\n=== ROUTE COSTS BY MODEL AND ROUTE TYPE ===")
print(
    all_summary[all_summary["complete"]]
    .groupby(["model", "route_type"])
    .agg(
        routes=("observed_proxy_cost_minutes", "size"),
        mean_cost_minutes=("observed_proxy_cost_minutes", "mean"),
        median_cost_minutes=("observed_proxy_cost_minutes", "median"),
    )
    .round(6)
    .to_string()
)

print("\n=== PAIRED COMPARISON ON COMMON OBSERVED COSTS ===")
print("Matched complete route evaluations:", len(comparison))

if len(comparison):
    print("Hybrid lower-cost routes:", int(comparison["hybrid_lower_cost"].sum()))
    print("LSTM lower-cost routes:", int(comparison["lstm_lower_cost"].sum()))
    print("Equal-cost routes:", int(comparison["equal_cost"].sum()))
    print(
        "Mean hybrid-minus-LSTM cost (minutes):",
        round(comparison["hybrid_minus_lstm_minutes"].mean(), 6),
    )
    print(
        "Median hybrid-minus-LSTM cost (minutes):",
        round(comparison["hybrid_minus_lstm_minutes"].median(), 6),
    )
    print(
        "Mean relative cost difference (%):",
        round(
            100 * (
                comparison["hybrid_actual_proxy_cost"]
                - comparison["lstm_actual_proxy_cost"]
            ).sum()
            / comparison["lstm_actual_proxy_cost"].sum(),
            4,
        ),
    )

print("\nSaved:")
print(summary_path)
print(comparison_path)
