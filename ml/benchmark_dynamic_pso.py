import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 11388
DESTINATION = 14979

SCENARIO_SEED = 42
AFFECTED_FRACTION = 0.15
COST_MULTIPLIER = 2.0

N_PARTICLES = 30
N_ITERATIONS = 80
N_TRIALS = 20

W = 0.70
C1 = 1.50
C2 = 1.50
MUTATION_RATE = 0.15


def build_graph(df):
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        u, v = int(row.start_node), int(row.end_node)
        if u == v:
            continue

        cost = float(row.traffic_aware_cost_minutes)
        attrs = {
            "cost": cost,
            "segment_id": str(row.segment_id),
        }

        if not graph.has_edge(u, v) or cost < graph[u][v]["cost"]:
            graph.add_edge(u, v, **attrs)

    return graph


df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
G = build_graph(df)

# Apply the same deterministic disruption to the simplified graph.
edge_list = sorted((min(u, v), max(u, v)) for u, v in G.edges())
scenario_rng = random.Random(SCENARIO_SEED)
affected_count = round(len(edge_list) * AFFECTED_FRACTION)
affected_edges = set(scenario_rng.sample(edge_list, affected_count))

for u, v in affected_edges:
    G[u][v]["cost"] *= COST_MULTIPLIER


def route_cost(path):
    return sum(G[u][v]["cost"] for u, v in zip(path[:-1], path[1:]))


def route_valid(path):
    return (
        bool(path)
        and path[0] == SOURCE
        and path[-1] == DESTINATION
        and len(path) == len(set(path))
        and all(G.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))
    )


def randomized_route(rng):
    weights = {}

    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        weights[key] = data["cost"] * np.exp(rng.normal(0, 0.65))

    def weight(u, v, data):
        return weights[(min(u, v), max(u, v))]

    return nx.shortest_path(
        G, SOURCE, DESTINATION, weight=weight
    )


def make_edit(current, target, rng):
    cp = {node: i for i, node in enumerate(current)}
    tp = {node: i for i, node in enumerate(target)}
    shared = [node for node in target if node in cp]
    edits = []

    for i, start in enumerate(shared):
        for end in shared[i + 1:]:
            if cp[start] < cp[end] and tp[start] < tp[end]:
                edits.append(
                    (start, end, target[tp[start]:tp[end] + 1])
                )

    return edits[rng.integers(len(edits))] if edits else None


def apply_edit(path, edit):
    if edit is None:
        return path

    start, end, replacement = edit

    if start not in path or end not in path:
        return path

    i, j = path.index(start), path.index(end)
    if i >= j:
        return path

    candidate = path[:i] + list(replacement) + path[j + 1:]

    if (
        len(candidate) == len(set(candidate))
        and all(G.has_edge(u, v) for u, v in zip(candidate[:-1], candidate[1:]))
        and candidate[0] == SOURCE
        and candidate[-1] == DESTINATION
    ):
        return candidate

    return path


def run_pso(seed):
    rng = np.random.default_rng(seed)

    # Initialize without supplying the Dijkstra route.
    particles = []

    for _ in range(N_PARTICLES):
        path = randomized_route(rng)
        particles.append({
            "position": path,
            "velocity": [],
            "pbest": path.copy(),
            "pbest_cost": route_cost(path),
        })

    initial_routes = {tuple(p["position"]) for p in particles}
    initial_best_cost = min(p["pbest_cost"] for p in particles)

    best_particle = min(particles, key=lambda p: p["pbest_cost"])
    global_best = best_particle["pbest"].copy()
    global_best_cost = route_cost(global_best)

    evaluations = N_PARTICLES
    start_time = time.perf_counter()

    for _ in range(N_ITERATIONS):
        for particle in particles:
            current = particle["position"]

            velocity = [
                edit for edit in particle["velocity"]
                if rng.random() < W
            ]

            if rng.random() < min(1.0, C1 / 2):
                edit = make_edit(current, particle["pbest"], rng)
                if edit is not None:
                    velocity.append(edit)

            if rng.random() < min(1.0, C2 / 2):
                edit = make_edit(current, global_best, rng)
                if edit is not None:
                    velocity.append(edit)

            candidate = current.copy()
            for edit in velocity:
                candidate = apply_edit(candidate, edit)

            if rng.random() < MUTATION_RATE:
                candidate = randomized_route(rng)

            if not route_valid(candidate):
                candidate = current.copy()

            particle["position"] = candidate
            particle["velocity"] = velocity

            cost = route_cost(candidate)
            evaluations += 1

            if cost < particle["pbest_cost"]:
                particle["pbest"] = candidate.copy()
                particle["pbest_cost"] = cost

            if cost < global_best_cost:
                global_best = candidate.copy()
                global_best_cost = cost

    runtime_ms = (time.perf_counter() - start_time) * 1000

    return {
        "initial_best_cost": initial_best_cost,
        "final_best_cost": route_cost(global_best),
        "initial_unique_routes": len(initial_routes),
        "final_unique_routes": len({
            tuple(p["position"]) for p in particles
        }),
        "evaluations": evaluations,
        "runtime_ms": runtime_ms,
        "valid_route": route_valid(global_best),
        "final_route": " -> ".join(map(str, global_best)),
    }


# Dijkstra is the reference optimum under changed costs.
reference_route = nx.shortest_path(G, SOURCE, DESTINATION, weight="cost")
optimum = route_cost(reference_route)

print("\n========== DYNAMIC PSO BENCHMARK ==========")
print(f"Route: {SOURCE} -> {DESTINATION}")
print(f"Graph edges: {G.number_of_edges()}")
print(f"Disrupted edges: {len(affected_edges)}")
print(f"Dijkstra changed-scenario cost: {optimum:.8f} minutes")

records = []

for trial in range(1, N_TRIALS + 1):
    seed = 700000 + trial
    result = run_pso(seed)

    initial_gap = (result["initial_best_cost"] / optimum - 1) * 100
    final_gap = (result["final_best_cost"] / optimum - 1) * 100

    record = {
        "trial": trial,
        "seed": seed,
        "dijkstra_cost": optimum,
        "initial_gap_percent": initial_gap,
        "final_gap_percent": final_gap,
        **result,
    }
    records.append(record)

    print(
        f"Trial {trial:02d}: initial gap={initial_gap:.4f}%, "
        f"final gap={final_gap:.4f}%, "
        f"runtime={result['runtime_ms']:.1f} ms"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(records)
results_path = OUTPUT_DIR / "dynamic_pso_trials.csv"
results.to_csv(results_path, index=False)

summary = pd.DataFrame([{
    "trials": len(results),
    "initial_optimal_rate_percent": (
        results["initial_gap_percent"].abs().lt(1e-7).mean() * 100
    ),
    "final_optimal_rate_percent": (
        results["final_gap_percent"].abs().lt(1e-7).mean() * 100
    ),
    "mean_initial_gap_percent": results["initial_gap_percent"].mean(),
    "mean_final_gap_percent": results["final_gap_percent"].mean(),
    "mean_runtime_ms": results["runtime_ms"].mean(),
}])

summary_path = OUTPUT_DIR / "dynamic_pso_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n========== SUMMARY ==========")
print(summary.round(4).to_string(index=False))
print(f"\nSaved trials: {results_path}")
print(f"Saved summary: {summary_path}")
