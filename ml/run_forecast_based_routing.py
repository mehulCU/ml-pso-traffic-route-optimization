
import pandas as pd
import networkx as nx
from pathlib import Path

EDGE_PATH = Path("ml/regional_model/regional_routing_edges.csv")
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 11388
DESTINATION = 14979
COST_COLUMN = "traffic_aware_cost_minutes"


def main():
    edges = pd.read_csv(
        EDGE_PATH,
        dtype={"segment_id": str},
    )

    required = {
        "segment_id",
        "start_node",
        "end_node",
        "distance",
        "predicted_probe_count",
        "historical_mean_probe_count",
        "free_flow_minutes",
        "traffic_aware_cost_minutes",
    }

    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if edges["segment_id"].duplicated().any():
        raise ValueError("Duplicate segment IDs found.")

    if edges[list(required - {"segment_id"})].isna().any().any():
        raise ValueError("Missing values found in required numeric columns.")

    if (edges[COST_COLUMN] < 0).any():
        raise ValueError("Dijkstra requires non-negative edge costs.")

    # Use a simple graph, retaining the lowest-cost parallel edge.
    graph = nx.Graph()

    for row in edges.itertuples(index=False):
        u = int(row.start_node)
        v = int(row.end_node)
        cost = float(row.traffic_aware_cost_minutes)

        attributes = {
            "segment_id": str(row.segment_id),
            "distance": float(row.distance),
            "predicted_probe_count": float(row.predicted_probe_count),
            "historical_mean_probe_count": float(
                row.historical_mean_probe_count
            ),
            "free_flow_minutes": float(row.free_flow_minutes),
            "traffic_aware_cost_minutes": cost,
        }

        if not graph.has_edge(u, v) or cost < graph[u][v]["weight"]:
            graph.add_edge(u, v, weight=cost, **attributes)

    print("========== FORECAST-BASED ROUTING ==========")
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())
    print("Source:", SOURCE)
    print("Destination:", DESTINATION)

    if SOURCE not in graph or DESTINATION not in graph:
        raise ValueError("Source or destination is absent from the graph.")

    if not nx.has_path(graph, SOURCE, DESTINATION):
        raise ValueError("No route exists between the selected nodes.")

    route = nx.shortest_path(
        graph,
        source=SOURCE,
        target=DESTINATION,
        weight="weight",
        method="dijkstra",
    )

    rows = []
    for u, v in zip(route[:-1], route[1:]):
        data = graph[u][v]

        rows.append({
            "from_node": u,
            "to_node": v,
            "segment_id": data["segment_id"],
            "distance_m": data["distance"],
            "predicted_probe_count": data["predicted_probe_count"],
            "historical_mean_probe_count": (
                data["historical_mean_probe_count"]
            ),
            "free_flow_minutes": data["free_flow_minutes"],
            "traffic_aware_cost_minutes": (
                data["traffic_aware_cost_minutes"]
            ),
        })

    route_df = pd.DataFrame(rows)

    route_cost = route_df["traffic_aware_cost_minutes"].sum()
    route_distance = route_df["distance_m"].sum()

    output_path = OUTPUT_DIR / "forecast_based_route.csv"
    route_df.to_csv(output_path, index=False)

    print("\nRoute nodes:", len(route))
    print("Road segments:", len(route_df))
    print(f"Total distance: {route_distance:.2f} m")
    print(f"Estimated route cost: {route_cost:.6f} minutes")
    print("Cost basis:", COST_COLUMN)
    print("Saved route:", output_path)

    print("\nFirst 10 route segments:")
    print(route_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
