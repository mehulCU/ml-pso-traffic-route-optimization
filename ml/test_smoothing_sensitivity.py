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
SMOOTHING_VALUES = [0.0, 1.0, 5.0, 10.0]

ROUTE_PAIRS = [
    (3319, 316), (11703, 6988), (2272, 2753),
    (2681, 15490), (3598, 15490), (4241, 2753),
    (6954, 2753), (4638, 6988), (458, 425),
    (3424, 425), (11388, 14979),
]


def build_graph(edges, cost_column):
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        u, v = row.start_node, row.end_node
        if u == v:
            continue
        weight = float(getattr(row, cost_column))
        if not graph.has_edge(u, v) or weight < graph[u][v]["weight"]:
            graph.add_edge(
                u, v,
                weight=weight,
                segment_id=str(row.segment_id),
            )
    return graph


def main():
    print("Loading road network and historical data...", flush=True)

    edges = pd.read_csv(REGION_PATH, dtype={"segment_id": str})
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
        raise ValueError("Missing historical baselines.")
    if (edges["speed_limit"] <= 0).any():
        raise ValueError("Non-positive speed limits found.")

    edges["free_flow_minutes"] = (
        edges["distance"] * 0.06 / edges["speed_limit"]
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["forecast_timestamp"],
        dtype={"segment_id": str},
    )

    if predictions.duplicated(
        ["forecast_timestamp", "segment_id"]
    ).any():
        raise ValueError("Duplicate forecasts found.")

    output_rows = []
    total_hours = predictions["forecast_timestamp"].nunique()

    for smoothing in SMOOTHING_VALUES:
        print(
            f"\nTesting smoothing={smoothing:g} "
            f"over {total_hours} hours...",
            flush=True,
        )
        rows = []

        for timestamp, forecasts in predictions.groupby(
            "forecast_timestamp", sort=True
        ):
            hour_edges = edges.merge(
                forecasts[["segment_id", "predicted_probe_count"]],
                on="segment_id",
                how="left",
                validate="one_to_one",
            )

            if hour_edges["predicted_probe_count"].isna().any():
                raise ValueError(f"Missing forecast at {timestamp}")

            baseline = hour_edges["historical_mean_probe_count"]
            if smoothing == 0:
                # Preserve the original model's denominator floor.
                denominator = baseline.clip(lower=1.0)
                ratio = hour_edges["predicted_probe_count"] / denominator
            else:
                ratio = (
                    (hour_edges["predicted_probe_count"] + smoothing)
                    / (baseline + smoothing)
                )

            hour_edges["forecast_cost_minutes"] = (
                hour_edges["free_flow_minutes"]
                * (
                    1
                    + ALPHA * (ratio - 1).clip(lower=0)
                )
            )

            free_graph = build_graph(hour_edges, "free_flow_minutes")
            forecast_graph = build_graph(
                hour_edges, "forecast_cost_minutes"
            )
            forecast_cost_by_segment = hour_edges.set_index(
                "segment_id"
            )["forecast_cost_minutes"].to_dict()

            for source, destination in ROUTE_PAIRS:
                row = {
                    "smoothing": smoothing,
                    "forecast_timestamp": timestamp,
                    "source": source,
                    "destination": destination,
                }

                try:
                    free_path = nx.shortest_path(
                        free_graph, source, destination, weight="weight"
                    )
                    dynamic_path = nx.shortest_path(
                        forecast_graph, source, destination, weight="weight"
                    )

                    free_ids = [
                        free_graph[u][v]["segment_id"]
                        for u, v in zip(free_path[:-1], free_path[1:])
                    ]
                    dynamic_ids = [
                        forecast_graph[u][v]["segment_id"]
                        for u, v in zip(
                            dynamic_path[:-1], dynamic_path[1:]
                        )
                    ]

                    static_cost = sum(
                        forecast_cost_by_segment[sid] for sid in free_ids
                    )
                    dynamic_cost = sum(
                        forecast_graph[u][v]["weight"]
                        for u, v in zip(
                            dynamic_path[:-1], dynamic_path[1:]
                        )
                    )
                    saving = static_cost - dynamic_cost

                    row.update({
                        "status": "ok",
                        "static_route_cost_under_forecast_minutes": static_cost,
                        "dynamic_route_cost_minutes": dynamic_cost,
                        "estimated_saving_minutes": saving,
                        "estimated_saving_percent": (
                            100 * saving / static_cost
                            if static_cost > 0 else 0
                        ),
                        "junction_path_changed": free_path != dynamic_path,
                        "road_segments_changed": free_ids != dynamic_ids,
                        "dynamic_segment_ids": "|".join(dynamic_ids),
                    })

                except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
                    row["status"] = f"route_unavailable: {exc}"

                rows.append(row)

        result = pd.DataFrame(rows)
        output_path = OUTPUT_DIR / (
            f"smoothing_sensitivity_{smoothing:g}.csv"
        )
        result.to_csv(output_path, index=False)

        valid = result[result["status"] == "ok"]
        print(
            f"  Successful: {len(valid)}/{len(result)}; "
            f"mean saving: {valid['estimated_saving_percent'].mean():.3f}%; "
            f"median saving: {valid['estimated_saving_percent'].median():.3f}%; "
            f"junction changes: {int(valid['junction_path_changed'].sum())}",
            flush=True,
        )

        output_rows.append({
            "smoothing": smoothing,
            "successful_evaluations": len(valid),
            "total_evaluations": len(result),
            "mean_saving_percent": valid["estimated_saving_percent"].mean(),
            "median_saving_percent": valid["estimated_saving_percent"].median(),
            "mean_saving_minutes": valid["estimated_saving_minutes"].mean(),
            "junction_paths_changed": int(
                valid["junction_path_changed"].sum()
            ),
            "road_segment_selections_changed": int(
                valid["road_segments_changed"].sum()
            ),
        })

    summary = pd.DataFrame(output_rows)
    summary_path = OUTPUT_DIR / "smoothing_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n========== SENSITIVITY SUMMARY ==========")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
