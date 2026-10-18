import time

import networkx as nx
import numpy as np
import pandas as pd

from api.services.routing_service import (
    ALPHA,
    SMOOTHING,
    _build_graph,
    _load_data,
    _route_result,
)


def get_pso_route_comparison(
    source: int,
    destination: int,
    forecast_timestamp: str,
    n_particles: int = 20,
    n_iterations: int = 40,
    seed: int = 42,
) -> dict:
    if source == destination:
        raise ValueError("Source and destination must be different.")

    if n_particles < 2 or n_iterations < 1:
        raise ValueError("PSO requires at least 2 particles and 1 iteration.")

    edges, predictions = _load_data()

    timestamp = pd.to_datetime(forecast_timestamp, errors="raise")
    if timestamp.tzinfo is not None:
        raise ValueError("Use a timezone-naive forecast timestamp.")
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

    graph = _build_graph(hour_edges, "forecast_cost_minutes")

    if source not in graph or destination not in graph:
        raise ValueError("Source or destination is not in the road graph.")

    if not nx.has_path(graph, source, destination):
        raise ValueError("No connected route exists between these junctions.")

    # Dijkstra provides the exact shortest-path baseline for these costs.
    dijkstra_route = _route_result(graph, source, destination)

    # Stable edge indices for random-key PSO.
    edge_list = sorted(
        (min(u, v), max(u, v))
        for u, v in graph.edges()
    )
    edge_index = {edge: i for i, edge in enumerate(edge_list)}
    n_edges = len(edge_list)

    rng = np.random.default_rng(seed)
    inertia = 0.7
    cognitive = 1.5
    social = 1.5

    def decode(position):
        def modified_weight(u, v, attributes):
            key = (min(u, v), max(u, v))
            idx = edge_index[key]
            return attributes["weight"] * np.exp(position[idx])

        return nx.shortest_path(
            graph,
            source=source,
            target=destination,
            weight=modified_weight,
        )

    def score(path):
        return sum(
            graph[u][v]["weight"]
            for u, v in zip(path[:-1], path[1:])
        )

    positions = rng.uniform(-0.35, 0.35, (n_particles, n_edges))
    velocities = rng.uniform(-0.05, 0.05, (n_particles, n_edges))

    personal_positions = positions.copy()
    personal_costs = np.full(n_particles, np.inf)

    global_position = None
    global_path = None
    global_cost = np.inf

    evaluations = 0
    started = time.perf_counter()

    for i in range(n_particles):
        path = decode(positions[i])
        cost = score(path)
        evaluations += 1
        personal_costs[i] = cost

        if cost < global_cost:
            global_cost = cost
            global_position = positions[i].copy()
            global_path = path

    initial_best_cost = global_cost

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1 = rng.random(n_edges)
            r2 = rng.random(n_edges)

            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1 * (personal_positions[i] - positions[i])
                + social * r2 * (global_position - positions[i])
            )
            velocities[i] = np.clip(velocities[i], -0.5, 0.5)
            positions[i] = np.clip(
                positions[i] + velocities[i], -2.0, 2.0
            )

            path = decode(positions[i])
            cost = score(path)
            evaluations += 1

            if cost < personal_costs[i]:
                personal_costs[i] = cost
                personal_positions[i] = positions[i].copy()

            if cost < global_cost:
                global_cost = cost
                global_position = positions[i].copy()
                global_path = path

    runtime_ms = (time.perf_counter() - started) * 1000
    pso_route = _route_result(graph, source, destination)

    # Build route details for the PSO's actual best path.
    def details_for_path(path):
        segments = []
        distance = 0.0
        cost = 0.0

        for u, v in zip(path[:-1], path[1:]):
            edge = graph[u][v]
            distance += edge["distance"]
            cost += edge["weight"]
            segments.append({
                "segment_id": edge["segment_id"],
                "from_node": u,
                "to_node": v,
                "street_name": edge["street_name"],
                "distance": round(edge["distance"], 2),
                "estimated_cost_minutes": round(edge["weight"], 5),
            })

        return {
            "nodes": path,
            "segments": segments,
            "segment_count": len(segments),
            "distance": round(distance, 2),
            "estimated_cost_minutes": round(cost, 5),
        }

    pso_route = details_for_path(global_path)
    dijkstra_cost = dijkstra_route["estimated_cost_minutes"]
    pso_cost = pso_route["estimated_cost_minutes"]

    return {
        "source": source,
        "destination": destination,
        "forecast_timestamp": timestamp.isoformat(),
        "routing_method": "Random-key Particle Swarm Optimization",
        "forecast_method": "Saved hybrid predictions",
        "pso_parameters": {
            "particles": n_particles,
            "iterations": n_iterations,
            "seed": seed,
            "evaluations": evaluations,
            "runtime_ms": round(runtime_ms, 2),
            "initial_best_cost_minutes": round(initial_best_cost, 6),
        },
        "dijkstra_baseline": dijkstra_route,
        "pso_route": pso_route,
        "comparison": {
            "pso_minus_dijkstra_minutes": round(pso_cost - dijkstra_cost, 6),
            "pso_matches_dijkstra_cost": bool(
                np.isclose(pso_cost, dijkstra_cost, rtol=1e-9, atol=1e-9)
            ),
            "note": (
                "Dijkstra is optimal for this graph and cost function. "
                "PSO is a stochastic search and may match it, but is not "
                "guaranteed to find a better route."
            ),
        },
    }
