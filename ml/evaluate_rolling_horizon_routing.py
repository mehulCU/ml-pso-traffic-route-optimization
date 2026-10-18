
import pandas as pd
import networkx as nx

from pathlib import Path

REGION_PATH = Path("ml/candidate_region_edges.csv")
TRAFFIC_PATH = Path("ml/regional_data/regional_traffic.csv")
PREDICTIONS_PATH = Path(
    "ml/regional_model/regional_timestamped_test_predictions.csv"
)
OUTPUT_DIR = Path("ml/regional_model")

ALPHA = 0.25

ROUTE_PAIRS = [
    (3319, 316),
    (11703, 6988),
    (2272, 2753),
    (2681, 15490),
    (3598, 15490),
    (4241, 2753),
    (6954, 2753),
    (4638, 6988),
    (458, 425),
    (3424, 425),
    (11388, 14979),
]


def build_graph(edges, cost_column):
    """Keep the cheapest parallel segment for each junction pair."""
    graph = nx.Graph()

    for row in edges.itertuples(index=False):
        u = row.start_node
        v = row.end_node

        if u == v:
            continue

        weight = float(getattr(row, cost_column))

        data = {
            "weight": weight,
            "segment_id": str(row.segment_id),
            "distance": float(row.distance),
        }

        if (
            not graph.has_edge(u, v)
            or weight < graph[u][v]["weight"]
        ):
            graph.add_edge(u, v, **data)

    return graph


def route_details(graph, path):
    """Get the ordered segment IDs, distance and graph cost."""
    segment_ids = []
    distance = 0.0
    cost = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge = graph[u][v]
        segment_ids.append(edge["segment_id"])
        distance += edge["distance"]
        cost += edge["weight"]

    return segment_ids, distance, cost


def main():
    print("Loading roads and historical baseline...", flush=True)

    edges = pd.read_csv(
        REGION_PATH,
        dtype={"segment_id": str},
    )

    traffic = pd.read_csv(
        TRAFFIC_PATH,
        parse_dates=["date"],
        dtype={"segment_id": str},
    )

    train = traffic[traffic["date"] < "2024-08-24"]

    historical_mean = (
        train.groupby("segment_id")["probe_count"]
        .mean()
        .rename("historical_mean_probe_count")
    )

    edges = edges.merge(
        historical_mean,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    if edges["historical_mean_probe_count"].isna().any():
        raise ValueError("Some road segments lack historical baselines.")

    if (edges["speed_limit"] <= 0).any():
        raise ValueError("Some roads have non-positive speed limits.")

    edges["free_flow_minutes"] = (
        edges["distance"] * 0.06 / edges["speed_limit"]
    )

    print("Loading timestamped forecasts...", flush=True)

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["forecast_timestamp"],
        dtype={"segment_id": str},
    )

    if predictions.duplicated(
        ["forecast_timestamp", "segment_id"]
    ).any():
        raise ValueError("Duplicate road/hour forecasts found.")

    # Store each forecast hour's results and ordered route segments.
    summary_rows = []
    route_rows = []

    hourly_groups = predictions.groupby(
        "forecast_timestamp", sort=True
    )

    print(
        f"Evaluating {len(hourly_groups)} hours...",
        flush=True,
    )

    for hour_number, (timestamp, hour_forecasts) in enumerate(
        hourly_groups, start=1
    ):
        hour_edges = edges.merge(
            hour_forecasts[
                ["segment_id", "predicted_probe_count"]
            ],
            on="segment_id",
            how="left",
            validate="one_to_one",
        )

        if hour_edges["predicted_probe_count"].isna().any():
            raise ValueError(
                f"Missing forecasts at {timestamp}"
            )

              # Smoothed probe-count ratio to reduce extreme ratios
        # on road segments with very low historical probe counts.
        SMOOTHING = 5.0

        baseline = (
            hour_edges["historical_mean_probe_count"]
        )

        hour_edges["traffic_ratio"] = (
            (hour_edges["predicted_probe_count"] + SMOOTHING)
            / (baseline + SMOOTHING)
        )

        hour_edges["forecast_cost_minutes"] = (
            hour_edges["free_flow_minutes"]
            * (
                1
                + ALPHA
                * (hour_edges["traffic_ratio"] - 1).clip(lower=0)
            )
        )

        # Independent graph construction makes segment selection explicit.
        free_graph = build_graph(
            hour_edges, "free_flow_minutes"
        )
        forecast_graph = build_graph(
            hour_edges, "forecast_cost_minutes"
        )

        # Exact cost for every original segment at this forecast hour.
        forecast_cost_by_segment = (
            hour_edges.set_index("segment_id")[
                "forecast_cost_minutes"
            ].to_dict()
        )

        for source, destination in ROUTE_PAIRS:
            row = {
                "forecast_timestamp": timestamp,
                "source": source,
                "destination": destination,
            }

            try:
                free_path = nx.shortest_path(
                    free_graph,
                    source,
                    destination,
                    weight="weight",
                )

                dynamic_path = nx.shortest_path(
                    forecast_graph,
                    source,
                    destination,
                    weight="weight",
                )

                free_ids, free_distance, free_cost = route_details(
                    free_graph, free_path
                )

                dynamic_ids, dynamic_distance, dynamic_cost = (
                    route_details(forecast_graph, dynamic_path)
                )

                # Score the exact static-route segments under this hour's
                # forecast costs. Do not substitute parallel segments.
                static_cost_under_forecast = sum(
                    forecast_cost_by_segment[segment_id]
                    for segment_id in free_ids
                )

                saving_minutes = (
                    static_cost_under_forecast - dynamic_cost
                )

                saving_percent = (
                    100.0 * saving_minutes
                    / static_cost_under_forecast
                    if static_cost_under_forecast > 0
                    else 0.0
                )

                path_changed = free_path != dynamic_path
                segments_changed = free_ids != dynamic_ids

                row.update({
                    "status": "ok",
                    "free_flow_route_cost_minutes": free_cost,
                    "static_route_cost_under_forecast_minutes":
                        static_cost_under_forecast,
                    "dynamic_route_cost_minutes": dynamic_cost,
                    "estimated_saving_minutes": saving_minutes,
                    "estimated_saving_percent": saving_percent,
                    "free_flow_route_distance_m": free_distance,
                    "dynamic_route_distance_m": dynamic_distance,
                    "free_flow_route_edges": len(free_ids),
                    "dynamic_route_edges": len(dynamic_ids),
                    "junction_path_changed": path_changed,
                    "road_segments_changed": segments_changed,
                })

                for route_type, segment_ids in [
                    ("static_free_flow", free_ids),
                    ("dynamic_forecast", dynamic_ids),
                ]:
                    for order, segment_id in enumerate(
                        segment_ids, start=1
                    ):
                        route_rows.append({
                            "forecast_timestamp": timestamp,
                            "source": source,
                            "destination": destination,
                            "route_type": route_type,
                            "edge_order": order,
                            "segment_id": segment_id,
                        })

            except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
                row.update({
                    "status": f"route_unavailable: {exc}",
                    "junction_path_changed": None,
                    "road_segments_changed": None,
                })

            summary_rows.append(row)

        if hour_number % 12 == 0:
            print(
                f"Processed {hour_number}/{len(hourly_groups)} hours",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    routes = pd.DataFrame(route_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = (
        OUTPUT_DIR / "rolling_horizon_routing_summary_smoothed.csv"
    )
    routes_path = (
        OUTPUT_DIR / "rolling_horizon_route_segments_smoothed.csv"
    )

    summary.to_csv(summary_path, index=False)
    routes.to_csv(routes_path, index=False)

    valid = summary[summary["status"] == "ok"]

    print("\n========== CORRECTED ROLLING RESULTS ==========")
    print("Forecast hours:", len(hourly_groups))
    print("Route evaluations:", len(summary))
    print("Successful evaluations:", len(valid))

    if not valid.empty:
        print(
            "Junction paths changed:",
            f"{int(valid['junction_path_changed'].sum())}/{len(valid)}",
        )
        print(
            "Road-segment selections changed:",
            f"{int(valid['road_segments_changed'].sum())}/{len(valid)}",
        )
        print(
            "Mean estimated saving (%):",
            f"{valid['estimated_saving_percent'].mean():.3f}",
        )
        print(
            "Median estimated saving (%):",
            f"{valid['estimated_saving_percent'].median():.3f}",
        )
        print(
            "Mean estimated saving (minutes):",
            f"{valid['estimated_saving_minutes'].mean():.5f}",
        )

    print("\nSaved summary:", summary_path)
    print("Saved route segments:", routes_path)
    print(
        "\nCosts are experimental probe-count-derived estimates, "
        "not measured travel times."
    )


if __name__ == "__main__":
    main()
