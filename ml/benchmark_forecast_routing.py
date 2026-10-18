
import pandas as pd
import networkx as nx
from pathlib import Path

EDGE_PATH = Path("ml/regional_model/regional_routing_edges.csv")
OUTPUT_DIR = Path("ml/regional_model")

SUMMARY_PATH = OUTPUT_DIR / "forecast_routing_benchmark.csv"
ROUTES_PATH = OUTPUT_DIR / "forecast_routing_benchmark_routes.csv"

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
    graph = nx.Graph()

    for row in edges.itertuples(index=False):
        u = int(row.start_node)
        v = int(row.end_node)
        cost = float(getattr(row, cost_column))

        data = {
            "weight": cost,
            "segment_id": str(row.segment_id),
            "distance": float(row.distance),
        }

        if not graph.has_edge(u, v) or cost < graph[u][v]["weight"]:
            graph.add_edge(u, v, **data)

    return graph


def get_route(graph, source, destination):
    nodes = nx.shortest_path(
        graph,
        source=source,
        target=destination,
        weight="weight",
        method="dijkstra",
    )

    segments = []
    total_cost = 0.0
    total_distance = 0.0

    for u, v in zip(nodes[:-1], nodes[1:]):
        edge = graph[u][v]
        segments.append(edge["segment_id"])
        total_cost += edge["weight"]
        total_distance += edge["distance"]

    return {
        "nodes": nodes,
        "segments": segments,
        "cost": total_cost,
        "distance_m": total_distance,
        "edge_count": len(segments),
    }


def main():
    edges = pd.read_csv(
        EDGE_PATH,
        dtype={"segment_id": str},
    )

    required = [
        "segment_id",
        "start_node",
        "end_node",
        "distance",
        "free_flow_minutes",
        "traffic_aware_cost_minutes",
    ]

    if edges[required].isna().any().any():
        raise ValueError("Missing values found in routing data.")

    # Map each segment ID to its forecast-based cost.
    # Route costs are summed over ordered edge sequences, not sets.
    forecast_cost_by_segment = (
        edges.set_index("segment_id")["traffic_aware_cost_minutes"]
        .to_dict()
    )

    free_graph = build_graph(edges, "free_flow_minutes")
    forecast_graph = build_graph(edges, "traffic_aware_cost_minutes")

    summary_rows = []
    route_rows = []

    for source, destination in ROUTE_PAIRS:
        if (
            source not in free_graph
            or destination not in free_graph
            or not nx.has_path(free_graph, source, destination)
        ):
            print(f"Skipping unreachable pair: {source} -> {destination}")
            continue

        free = get_route(free_graph, source, destination)
        forecast = get_route(forecast_graph, source, destination)

        free_route_forecast_cost = sum(
            forecast_cost_by_segment[segment]
            for segment in free["segments"]
        )

        saving = free_route_forecast_cost - forecast["cost"]
        saving_pct = (
            100 * saving / free_route_forecast_cost
            if free_route_forecast_cost > 0 else 0.0
        )

        free_nodes = free["nodes"]
        forecast_nodes = forecast["nodes"]

        # Ordered edge sequence overlap, based on segment IDs.
        free_segment_set = set(free["segments"])
        forecast_segment_set = set(forecast["segments"])
        union_count = len(free_segment_set | forecast_segment_set)
        overlap_pct = (
            100 * len(free_segment_set & forecast_segment_set)
            / union_count
            if union_count else 100.0
        )

        summary_rows.append({
            "source": source,
            "destination": destination,
            "free_flow_cost_min": free["cost"],
            "forecast_route_cost_min": forecast["cost"],
            "free_route_cost_under_forecast_min": free_route_forecast_cost,
            "modeled_cost_saved_min": saving,
            "modeled_cost_saved_pct": saving_pct,
            "free_flow_distance_m": free["distance_m"],
            "forecast_distance_m": forecast["distance_m"],
            "free_flow_segments": free["edge_count"],
            "forecast_segments": forecast["edge_count"],
            "route_changed": free_nodes != forecast_nodes,
            "segment_overlap_pct": overlap_pct,
            "forecast_route_is_no_more_expensive": (
                forecast["cost"] <= free_route_forecast_cost + 1e-9
            ),
        })

        for route_type, route in [
            ("free_flow", free),
            ("forecast_based", forecast),
        ]:
            for index, (u, v, segment) in enumerate(
                zip(
                    route["nodes"][:-1],
                    route["nodes"][1:],
                    route["segments"],
                ),
                start=1,
            ):
                route_rows.append({
                    "source": source,
                    "destination": destination,
                    "route_type": route_type,
                    "step": index,
                    "from_node": u,
                    "to_node": v,
                    "segment_id": segment,
                    "distance_m": float(edges.loc[
                        edges["segment_id"] == segment, "distance"
                    ].iloc[0]),
                    "forecast_edge_cost_min": forecast_cost_by_segment[
                        segment
                    ],
                })

        print(
            f"{source} -> {destination}: "
            f"changed={free_nodes != forecast_nodes}, "
            f"saving={saving_pct:.2f}%, "
            f"forecast cost verified="
            f"{forecast['cost'] <= free_route_forecast_cost + 1e-9}"
        )

    summary = pd.DataFrame(summary_rows)
    routes = pd.DataFrame(route_rows)

    summary.to_csv(SUMMARY_PATH, index=False)
    routes.to_csv(ROUTES_PATH, index=False)

    print("\n========== BENCHMARK SUMMARY ==========")
    print("Pairs tested:", len(summary))

    if not summary.empty:
        print(
            "Routes changed:",
            f"{summary['route_changed'].sum()}/{len(summary)}",
        )
        print(
            "Mean modeled saving:",
            f"{summary['modeled_cost_saved_pct'].mean():.2f}%",
        )
        print(
            "Pairs where forecast route is no more expensive:",
            f"{summary['forecast_route_is_no_more_expensive'].sum()}"
            f"/{len(summary)}",
        )

    print("\nSaved summary:", SUMMARY_PATH)
    print("Saved route details:", ROUTES_PATH)


if __name__ == "__main__":
    main()
