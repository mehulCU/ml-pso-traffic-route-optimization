
import pandas as pd
import networkx as nx
from pathlib import Path

EDGE_PATH = Path("ml/regional_model/regional_routing_edges.csv")
OUTPUT_PATH = Path("ml/regional_model/freeflow_vs_forecast_routes.csv")

SOURCE = 3319
DESTINATION = 316


def build_graph(df, cost_column):
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        u = int(row.start_node)
        v = int(row.end_node)
        cost = float(getattr(row, cost_column))

        data = {
            "segment_id": str(row.segment_id),
            "distance": float(row.distance),
            "free_flow_minutes": float(row.free_flow_minutes),
            "traffic_aware_cost_minutes": float(
                row.traffic_aware_cost_minutes
            ),
            "weight": cost,
        }

        if not graph.has_edge(u, v) or cost < graph[u][v]["weight"]:
            graph.add_edge(u, v, **data)

    return graph


def get_route(graph):
    nodes = nx.shortest_path(
        graph, SOURCE, DESTINATION, weight="weight"
    )
    rows = []

    for u, v in zip(nodes[:-1], nodes[1:]):
        data = graph[u][v]
        rows.append({
            "from_node": u,
            "to_node": v,
            "segment_id": data["segment_id"],
            "distance_m": data["distance"],
            "free_flow_minutes": data["free_flow_minutes"],
            "traffic_aware_cost_minutes": (
                data["traffic_aware_cost_minutes"]
            ),
            "optimized_cost_minutes": data["weight"],
        })

    return nodes, pd.DataFrame(rows)


def main():
    df = pd.read_csv(
        EDGE_PATH,
        dtype={"segment_id": str}
    )

    required = [
        "start_node",
        "end_node",
        "distance",
        "free_flow_minutes",
        "traffic_aware_cost_minutes",
    ]

    if df[required].isna().any().any():
        raise ValueError("Required routing values contain missing data.")

    free_graph = build_graph(df, "free_flow_minutes")
    forecast_graph = build_graph(df, "traffic_aware_cost_minutes")

    free_nodes, free_route = get_route(free_graph)
    forecast_nodes, forecast_route = get_route(forecast_graph)

    free_cost = free_route["optimized_cost_minutes"].sum()
    forecast_cost = forecast_route["optimized_cost_minutes"].sum()

    forecast_route["route_type"] = "forecast_based"
    free_route["route_type"] = "free_flow"

    combined = pd.concat(
        [free_route, forecast_route],
        ignore_index=True
    )
    combined.to_csv(OUTPUT_PATH, index=False)

    free_segments = set(free_route["segment_id"])
    forecast_segments = set(forecast_route["segment_id"])
    overlap = free_segments & forecast_segments

    print("========== FREE-FLOW VS FORECAST ROUTING ==========")
    print(f"Free-flow route cost: {free_cost:.6f} minutes")
    print(f"Forecast route cost:  {forecast_cost:.6f} minutes")
    print(f"Free-flow distance:   {free_route['distance_m'].sum():.2f} m")
    print(f"Forecast distance:    {forecast_route['distance_m'].sum():.2f} m")
    print(f"Free-flow segments:   {len(free_segments)}")
    print(f"Forecast segments:    {len(forecast_segments)}")
    print(f"Shared segments:      {len(overlap)}")
    print(f"Segment overlap:      {100 * len(overlap) / max(1, len(free_segments | forecast_segments)):.2f}%")
    print(f"Route changed:        {free_nodes != forecast_nodes}")
    print("Saved comparison:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
