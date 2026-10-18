import argparse
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from evaluate_forecast_discrete_pso import (
    ROOT, OUT, load_graph, cost, distance, valid,
    make_edit, apply_edit,
)

PARTICLES = 30
ITERATIONS = 80
W, C1, C2, MUTATION = 0.70, 1.50, 1.50, 0.15
PATIENCE, TOLERANCE = 15, 1e-9


def random_simple_route(graph, source, target, rng, max_expansions=5000):
    """Generate a randomized simple path with bounded DFS and a safe fallback."""
    shortest_hops = nx.shortest_path_length(graph, source, target)
    max_depth = min(100, max(12, 4 * shortest_hops + 8))

    # Retry with fresh randomized neighbor orders.
    for _ in range(12):
        path = [source]
        visited = {source}
        stack = [(source, None)]
        expansions = 0

        while stack and expansions < max_expansions:
            node, neighbors = stack[-1]

            if node == target:
                route = path.copy()
                if valid(graph, route, source, target):
                    return route

            if neighbors is None:
                if len(path) >= max_depth:
                    stack.pop()
                    visited.remove(path.pop())
                    continue

                options = list(graph.neighbors(node))
                rng.shuffle(options)
                stack[-1] = (node, iter(options))

            try:
                nxt = next(stack[-1][1])
            except StopIteration:
                stack.pop()
                visited.remove(path.pop())
                continue

            expansions += 1
            if nxt in visited:
                continue

            path.append(nxt)
            visited.add(nxt)
            stack.append((nxt, None))

    # Fallback: use independently randomized positive edge weights.
    noisy = nx.Graph()
    for u, v, data in graph.edges(data=True):
        noisy.add_edge(
            u, v,
            weight=float(data["cost"]) * float(np.exp(rng.gauss(0, 1.2)))
        )

    route = nx.shortest_path(noisy, source, target, weight="weight")
    if not valid(graph, route, source, target):
        raise RuntimeError("Randomized fallback produced an invalid route.")
    return route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--destination", type=int, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--particles", type=int, default=PARTICLES)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    graph, ts = load_graph(args.timestamp)

    source, target = args.source, args.destination
    if source not in graph or target not in graph:
        raise ValueError("Source or destination is not in the graph.")
    if not nx.has_path(graph, source, target):
        raise ValueError("No route exists between source and destination.")

    start = time.perf_counter()
    dijkstra_route = nx.shortest_path(graph, source, target, weight="cost")
    dijkstra_ms = (time.perf_counter() - start) * 1000
    dijkstra_cost = cost(graph, dijkstra_route)

    particles = []
    initial_routes = []

    for _ in range(args.particles):
        route = random_simple_route(graph, source, target, rng)
        initial_routes.append(tuple(route))
        particles.append({
            "position": route,
            "velocity": [],
            "best": route.copy(),
            "best_cost": cost(graph, route),
        })

    global_particle = min(particles, key=lambda p: p["best_cost"])
    global_best = global_particle["best"].copy()
    global_cost = cost(graph, global_best)
    initial_best_cost = global_cost
    initial_gap = max(0.0, (global_cost - dijkstra_cost) / dijkstra_cost * 100)
    first_optimal = 0 if abs(global_cost - dijkstra_cost) <= TOLERANCE else None
    convergence = [{
        "iteration": 0,
        "best_cost_minutes": global_cost,
        "optimality_gap_percent": initial_gap,
    }]

    start = time.perf_counter()
    stale = 0

    for iteration in range(1, args.iterations + 1):
        improved_this_iteration = False

        for particle in particles:
            current = particle["position"]
            velocity = [
                edit for edit in particle["velocity"]
                if rng.random() < W
            ]

            if rng.random() < min(1.0, C1 / 2):
                edit = make_edit(current, particle["best"], np_rng)
                if edit is not None:
                    velocity.append(edit)

            if rng.random() < min(1.0, C2 / 2):
                edit = make_edit(current, global_best, np_rng)
                if edit is not None:
                    velocity.append(edit)

            candidate = current.copy()
            for edit in velocity:
                candidate = apply_edit(graph, candidate, edit)

            if rng.random() < MUTATION:
                candidate = random_simple_route(graph, source, target, rng)

            if not valid(graph, candidate, source, target):
                candidate = current.copy()

            particle["position"] = candidate
            particle["velocity"] = velocity
            candidate_cost = cost(graph, candidate)

            if candidate_cost < particle["best_cost"] - TOLERANCE:
                particle["best"] = candidate.copy()
                particle["best_cost"] = candidate_cost

            if candidate_cost < global_cost - TOLERANCE:
                global_best = candidate.copy()
                global_cost = candidate_cost
                improved_this_iteration = True

        if first_optimal is None and abs(global_cost - dijkstra_cost) <= TOLERANCE:
            first_optimal = iteration

        convergence.append({
            "iteration": iteration,
            "best_cost_minutes": global_cost,
            "optimality_gap_percent": max(
                0.0, (global_cost - dijkstra_cost) / dijkstra_cost * 100
            ),
        })

        stale = 0 if improved_this_iteration else stale + 1
        if stale >= PATIENCE:
            break

    pso_ms = (time.perf_counter() - start) * 1000
    is_valid = valid(graph, global_best, source, target)
    final_gap = max(0.0, (global_cost - dijkstra_cost) / dijkstra_cost * 100)

    summary = {
        "source": source,
        "destination": target,
        "timestamp": str(ts),
        "seed": args.seed,
        "particles": args.particles,
        "iterations_completed": len(convergence) - 1,
        "distinct_initial_routes": len(set(initial_routes)),
        "initial_best_cost_minutes": initial_best_cost,
        "initial_optimality_gap_percent": initial_gap,
        "dijkstra_cost_minutes": dijkstra_cost,
        "pso_cost_minutes": cost(graph, global_best),
        "optimality_gap_percent": final_gap,
        "first_optimal_iteration": first_optimal,
        "valid_pso_route": is_valid,
        "dijkstra_runtime_ms": dijkstra_ms,
        "pso_runtime_ms": pso_ms,
        "route_distance_m": distance(graph, global_best),
    }

    pd.DataFrame([summary]).to_csv(
        OUT / "forecast_pso_random_init_summary.csv", index=False
    )
    pd.DataFrame(convergence).to_csv(
        OUT / "forecast_pso_random_init_convergence.csv", index=False
    )
    pd.DataFrame({
        "step": range(len(global_best)),
        "node": global_best,
    }).to_csv(OUT / "forecast_pso_random_init_best_route.csv", index=False)

    print("\n========== RANDOM-INITIALIZATION PSO ==========")
    print(f"Route: {source} -> {target} at {ts}")
    print(f"Distinct initial routes: {len(set(initial_routes))}/{args.particles}")
    print(f"Initial optimality gap: {initial_gap:.6f}%")
    print(f"Final optimality gap: {final_gap:.6f}%")
    print(f"First optimal iteration: {first_optimal}")
    print(f"Valid route: {is_valid}")
    print(f"Dijkstra runtime: {dijkstra_ms:.3f} ms")
    print(f"PSO runtime: {pso_ms:.3f} ms")
    print(f"Summary saved: {OUT / 'forecast_pso_random_init_summary.csv'}")


if __name__ == "__main__":
    main()
