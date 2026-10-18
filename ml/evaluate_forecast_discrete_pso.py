import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "ml/candidate_region_edges.csv"
TRAFFIC = ROOT / "ml/regional_data/regional_traffic.csv"
PREDICTIONS = ROOT / "ml/regional_model/regional_hybrid_routing_predictions.csv"
OUT = ROOT / "ml/regional_model"

ALPHA, SMOOTHING = 0.25, 5.0
PARTICLES, ITERATIONS = 30, 80
W, C1, C2, MUTATION = 0.70, 1.50, 1.50, 0.15
PATIENCE, TOLERANCE, SEED = 15, 1e-9, 42


def load_graph(timestamp):
    edges = pd.read_csv(EDGES, dtype={"segment_id": str})
    traffic = pd.read_csv(
        TRAFFIC, parse_dates=["date"], dtype={"segment_id": str}
    )
    training = traffic[traffic["date"] < "2024-08-24"]
    baseline = training.groupby("segment_id")["probe_count"].mean()

    edges["historical_mean_probe_count"] = edges["segment_id"].map(baseline)
    if edges["historical_mean_probe_count"].isna().any():
        raise ValueError("Road segments are missing historical baselines.")
    if (edges["speed_limit"] <= 0).any():
        raise ValueError("Speed limits must be positive.")

    edges["free_flow_minutes"] = edges["distance"] * 0.06 / edges["speed_limit"]

    predictions = pd.read_csv(
        PREDICTIONS,
        parse_dates=["forecast_timestamp"],
        dtype={"segment_id": str},
    )
    ts = pd.to_datetime(timestamp, errors="raise")
    if ts.tzinfo is not None:
        raise ValueError("Timestamp must not contain a timezone.")
    ts = ts.floor("h")
    hour = predictions[predictions["forecast_timestamp"] == ts]

    if hour.empty:
        raise ValueError(f"No saved hybrid predictions exist for {ts}.")
    if hour["segment_id"].duplicated().any():
        raise ValueError("Duplicate predictions exist for this hour.")

    pred = hour.set_index("segment_id")["hybrid_prediction"]
    edges["hybrid_prediction"] = edges["segment_id"].map(pred)
    if edges["hybrid_prediction"].isna().any():
        raise ValueError("Some road segments have no prediction for this hour.")

    ratio = (
        (edges["hybrid_prediction"] + SMOOTHING)
        / (edges["historical_mean_probe_count"] + SMOOTHING)
    )
    edges["cost"] = edges["free_flow_minutes"] * (
        1 + ALPHA * (ratio - 1).clip(lower=0)
    )

    graph = nx.Graph()
    for r in edges.itertuples(index=False):
        u, v = int(r.start_node), int(r.end_node)
        if u == v:
            continue
        data = {
            "cost": float(r.cost),
            "distance": float(r.distance),
            "segment_id": str(r.segment_id),
            "street_name": str(r.street_name),
        }
        if not graph.has_edge(u, v) or data["cost"] < graph[u][v]["cost"]:
            graph.add_edge(u, v, **data)

    return graph, ts


def cost(graph, route):
    return sum(graph[u][v]["cost"] for u, v in zip(route, route[1:]))


def distance(graph, route):
    return sum(graph[u][v]["distance"] for u, v in zip(route, route[1:]))


def valid(graph, route, source, target):
    return (
        bool(route)
        and route[0] == source
        and route[-1] == target
        and len(route) == len(set(route))
        and all(graph.has_edge(u, v) for u, v in zip(route, route[1:]))
    )


def randomized_route(graph, source, target, rng):
    noisy = nx.Graph()
    for u, v, data in graph.edges(data=True):
        noisy.add_edge(
            u, v,
            weight=data["cost"] * float(np.exp(rng.normal(0, 0.65))),
        )
    return nx.shortest_path(noisy, source, target, weight="weight")


def make_edit(current, target, rng):
    target_pos = {node: i for i, node in enumerate(target)}
    anchors = [
        (i, target_pos[node])
        for i, node in enumerate(current)
        if node in target_pos
    ]
    pairs = [
        (a, b) for a in anchors for b in anchors
        if a[0] < b[0] and a[1] < b[1]
    ]
    if not pairs:
        return None
    (i, ti), (j, tj) = pairs[int(rng.integers(len(pairs)))]
    return current[i], current[j], target[ti:tj + 1]


def apply_edit(graph, route, edit):
    if edit is None:
        return route.copy()
    start, end, replacement = edit
    try:
        i = route.index(start)
        j = route.index(end, i + 1)
    except ValueError:
        return route.copy()
    candidate = route[:i] + replacement + route[j + 1:]
    if (
        len(candidate) != len(set(candidate))
        or any(not graph.has_edge(u, v) for u, v in zip(candidate, candidate[1:]))
    ):
        return route.copy()
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, default=3319)
    parser.add_argument("--destination", type=int, default=316)
    parser.add_argument("--timestamp", default="2024-08-26T10:00:00")
    parser.add_argument("--particles", type=int, default=PARTICLES)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.source == args.destination:
        raise SystemExit("Source and destination must differ.")
    if args.particles < 2 or args.iterations < 1:
        raise SystemExit("Use at least 2 particles and 1 iteration.")

    graph, ts = load_graph(args.timestamp)
    source, target = args.source, args.destination
    for node in (source, target):
        if node not in graph:
            raise SystemExit(f"Junction {node} is absent from the graph.")

    try:
        t0 = time.perf_counter()
        dijkstra = nx.shortest_path(graph, source, target, weight="cost")
        dijkstra_ms = (time.perf_counter() - t0) * 1000
    except nx.NetworkXNoPath:
        raise SystemExit("No connected route exists between these junctions.")

    dijkstra_cost = cost(graph, dijkstra)
    dijkstra_distance = distance(graph, dijkstra)
    rng = np.random.default_rng(args.seed)

    particles = []
    initial_routes = []
    for _ in range(args.particles):
        route = randomized_route(graph, source, target, rng)
        initial_routes.append(tuple(route))
        particles.append({
            "position": route,
            "velocity": [],
            "best": route.copy(),
            "best_cost": cost(graph, route),
        })

    best_particle = min(particles, key=lambda p: p["best_cost"])
    global_best = best_particle["best"].copy()
    global_cost = cost(graph, global_best)
    initial_best_cost = global_cost
    initial_gap = max(0, (global_cost - dijkstra_cost) / dijkstra_cost * 100)
    first_optimal = 0 if global_cost <= dijkstra_cost + TOLERANCE else None
    history = []
    stagnant = 0
    t0 = time.perf_counter()

    for iteration in range(args.iterations):
        improved = False
        for particle in particles:
            current = particle["position"]
            velocity = [
                e for e in particle["velocity"] if rng.random() < W
            ]

            if rng.random() < min(1, C1 / 2):
                e = make_edit(current, particle["best"], rng)
                if e is not None:
                    velocity.append(e)
            if rng.random() < min(1, C2 / 2):
                e = make_edit(current, global_best, rng)
                if e is not None:
                    velocity.append(e)

            candidate = current.copy()
            for edit in velocity:
                candidate = apply_edit(graph, candidate, edit)
                if not valid(graph, candidate, source, target):
                    candidate = current.copy()
                    velocity = []
                    break

            if rng.random() < MUTATION:
                candidate = randomized_route(graph, source, target, rng)
                velocity = []

            particle["position"] = candidate
            particle["velocity"] = velocity
            candidate_cost = cost(graph, candidate)

            if candidate_cost < particle["best_cost"] - TOLERANCE:
                particle["best"] = candidate.copy()
                particle["best_cost"] = candidate_cost

            if candidate_cost < global_cost - TOLERANCE:
                global_best = candidate.copy()
                global_cost = candidate_cost
                improved = True
                if (
                    first_optimal is None
                    and global_cost <= dijkstra_cost + TOLERANCE
                ):
                    first_optimal = iteration + 1

        gap = max(0, (global_cost - dijkstra_cost) / dijkstra_cost * 100)
        history.append({
            "iteration": iteration + 1,
            "best_cost_minutes": global_cost,
            "optimality_gap_percent": gap,
        })
        stagnant = 0 if improved else stagnant + 1
        if stagnant >= PATIENCE:
            print(f"Early stopping at iteration {iteration + 1}.")
            break

    pso_ms = (time.perf_counter() - t0) * 1000
    pso_cost = cost(graph, global_best)
    pso_distance = distance(graph, global_best)
    gap = max(0, (pso_cost - dijkstra_cost) / dijkstra_cost * 100)
    is_valid = valid(graph, global_best, source, target)

    OUT.mkdir(parents=True, exist_ok=True)
    route_rows = []
    for u, v in zip(global_best, global_best[1:]):
        e = graph[u][v]
        route_rows.append({
            "forecast_timestamp": ts.isoformat(),
            "from_node": u,
            "to_node": v,
            "segment_id": e["segment_id"],
            "street_name": e["street_name"],
            "distance_m": e["distance"],
            "cost_minutes": e["cost"],
        })

    pd.DataFrame(route_rows).to_csv(
        OUT / "forecast_discrete_pso_best_route.csv", index=False
    )
    pd.DataFrame(history).to_csv(
        OUT / "forecast_discrete_pso_convergence.csv", index=False
    )
    pd.DataFrame([{
        "forecast_timestamp": ts.isoformat(),
        "source": source,
        "destination": target,
        "particles": args.particles,
        "iterations_completed": len(history),
        "first_optimal_iteration": first_optimal,
        "distinct_initial_routes": len(set(initial_routes)),
        "valid_pso_route": is_valid,
        "dijkstra_cost_minutes": dijkstra_cost,
        "pso_cost_minutes": pso_cost,
        "optimality_gap_percent": gap,
        "dijkstra_distance_m": dijkstra_distance,
        "pso_distance_m": pso_distance,
        "dijkstra_runtime_ms": dijkstra_ms,
        "pso_runtime_ms": pso_ms,
        "initial_best_cost_minutes": initial_best_cost,
        "initial_optimality_gap_percent": initial_gap,
        "alpha": ALPHA,
        "smoothing": SMOOTHING,
    }]).to_csv(OUT / "forecast_discrete_pso_summary.csv", index=False)

    print("\n=== FORECAST-AWARE DISCRETE PSO VS DIJKSTRA ===")
    print(f"Timestamp: {ts.isoformat()}")
    print(f"Route: {source} -> {target}")
    print(f"Graph: {graph.number_of_nodes()} junctions, {graph.number_of_edges()} edges")
    print(f"Distinct initial routes: {len(set(initial_routes))}")
    print(f"Iterations completed: {len(history)}")
    print(f"First optimal iteration (0 = initially optimal): {first_optimal}")
    print(f"Dijkstra: {dijkstra_cost:.8f} min, {dijkstra_distance:.2f} m, {dijkstra_ms:.3f} ms")
    print(f"PSO:      {pso_cost:.8f} min, {pso_distance:.2f} m, {pso_ms:.3f} ms")
    print(f"Valid PSO route: {is_valid}; optimality gap: {gap:.8f}%")
    print("Costs are probe-count-derived estimates, not measured travel times.")
    print("Saved: ml/regional_model/forecast_discrete_pso_best_route.csv")
    print("Saved: ml/regional_model/forecast_discrete_pso_convergence.csv")
    print("Saved: ml/regional_model/forecast_discrete_pso_summary.csv")


if __name__ == "__main__":
    main()
