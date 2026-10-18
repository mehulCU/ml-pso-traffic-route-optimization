from pathlib import Path

import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EDGES_PATH = PROJECT_ROOT / "ml" / "candidate_region_edges.csv"
TRAFFIC_PATH = PROJECT_ROOT / "ml" / "regional_data" / "regional_traffic.csv"
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "ml"
    / "regional_model"
    / "regional_hybrid_routing_predictions.csv"
)

ALPHA = 0.25
SMOOTHING = 5.0


def _load_data():
    """Load the road graph inputs and historical baselines."""

    edges = pd.read_csv(EDGES_PATH, dtype={"segment_id": str})
    traffic = pd.read_csv(
        TRAFFIC_PATH,
        parse_dates=["date"],
        dtype={"segment_id": str},
    )

    training = traffic[traffic["date"] < "2024-08-24"]

    historical_mean = (
        training.groupby("segment_id")["probe_count"]
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
        raise ValueError("Some road segments have no historical baseline.")

    if (edges["speed_limit"] <= 0).any():
        raise ValueError("Road speed limits must be positive.")

    edges["free_flow_minutes"] = (
        edges["distance"] * 0.06 / edges["speed_limit"]
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["forecast_timestamp"],
        dtype={"segment_id": str},
    )

    return edges, predictions


def _build_graph(edges: pd.DataFrame, cost_column: str) -> nx.Graph:
    """Build an undirected graph, retaining the cheapest parallel segment."""

    graph = nx.Graph()

    for row in edges.itertuples(index=False):
        source = int(row.start_node)
        destination = int(row.end_node)

        if source == destination:
            continue

        weight = float(getattr(row, cost_column))

        edge_data = {
            "weight": weight,
            "segment_id": str(row.segment_id),
            "distance": float(row.distance),
            "street_name": str(row.street_name),
        }

        if (
            not graph.has_edge(source, destination)
            or weight < graph[source][destination]["weight"]
        ):
            graph.add_edge(source, destination, **edge_data)

    return graph


def _route_result(graph: nx.Graph, source: int, destination: int) -> dict:
    """Return route nodes, segments, distance, and estimated cost."""

    path = nx.shortest_path(
        graph,
        source=source,
        target=destination,
        weight="weight",
    )

    segments = []
    total_distance = 0.0
    total_cost = 0.0

    for start, end in zip(path[:-1], path[1:]):
        edge = graph[start][end]
        total_distance += edge["distance"]
        total_cost += edge["weight"]

        segments.append(
            {
                "segment_id": edge["segment_id"],
                "from_node": start,
                "to_node": end,
                "street_name": edge["street_name"],
                "distance": round(edge["distance"], 2),
                "estimated_cost_minutes": round(edge["weight"], 5),
            }
        )

    return {
        "nodes": path,
        "segments": segments,
        "segment_count": len(segments),
        "distance": round(total_distance, 2),
        "estimated_cost_minutes": round(total_cost, 5),
    }


def get_route_comparison(
    source: int,
    destination: int,
    forecast_timestamp: str,
) -> dict:
    """Compare static and hybrid-forecast routes for one forecast hour."""

    if source == destination:
        raise ValueError("Source and destination must be different.")

    edges, predictions = _load_data()

    timestamp = pd.to_datetime(forecast_timestamp, errors="raise")
    if timestamp.tzinfo is not None:
        raise ValueError(
            "Use a timezone-naive forecast timestamp, e.g. 2024-08-26T10:00:00."
        )
    timestamp = timestamp.floor("h")

    hour_predictions = predictions[
        predictions["forecast_timestamp"] == timestamp
    ]

    if hour_predictions.empty:
        available = predictions["forecast_timestamp"]
        raise ValueError(
            "No hybrid predictions exist for this timestamp. "
            f"Available range: {available.min()} to {available.max()}."
        )

    if hour_predictions["segment_id"].duplicated().any():
        raise ValueError("Duplicate segment predictions exist for this hour.")

    hour_edges = edges.merge(
        hour_predictions[["segment_id", "hybrid_prediction"]],
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    if hour_edges["hybrid_prediction"].isna().any():
        raise ValueError("Some road segments lack predictions for this hour.")

    baseline = hour_edges["historical_mean_probe_count"]
    ratio = (
        (hour_edges["hybrid_prediction"] + SMOOTHING)
        / (baseline + SMOOTHING)
    )

    hour_edges["forecast_cost_minutes"] = (
        hour_edges["free_flow_minutes"]
        * (1 + ALPHA * (ratio - 1).clip(lower=0))
    )

    static_graph = _build_graph(hour_edges, "free_flow_minutes")
    forecast_graph = _build_graph(hour_edges, "forecast_cost_minutes")

    for node in (source, destination):
        if node not in forecast_graph:
            raise ValueError(f"Junction {node} is not present in the road graph.")

    try:
        static_route = _route_result(static_graph, source, destination)
        forecast_route = _route_result(forecast_graph, source, destination)
    except nx.NetworkXNoPath as exc:
        raise ValueError(
            "No connected route exists between these junctions."
        ) from exc

    return {
        "source": source,
        "destination": destination,
        "forecast_timestamp": timestamp.isoformat(),
        "routing_method": "NetworkX weighted shortest path",
        "forecast_method": "Saved hybrid predictions",
        "cost_model": {
            "alpha": ALPHA,
            "smoothing": SMOOTHING,
            "note": (
                "Estimated costs based on distance, speed limit, and "
                "probe-count-derived penalties; not measured travel times."
            ),
        },
        "static_route": static_route,
        "forecast_route": forecast_route,
    }
