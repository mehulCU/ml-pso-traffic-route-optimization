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

edge_list = sorted((min(u, v), max(u, v)) for u, v in G.edges())
scenario_rng = random.Random(SCENARIO_SEED)
affected_count = round(len(edge_list) * AFFECTED_FRACTION)
affected_edges = set(scenario_rng.sample(edge_list, affected_count))

for u, v in affected_edges:
    G[u][v]["cost"] *= COST_MULTIPLIER


def route_cost(path):
    return sum(G[u][v]["cost"] for u, v in zip(path[:-1], path[1:]))


def randomized_route(rng):
    weights = {}

    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        weights[key] = data["cost"] * np.exp(rng.normal(0, 0.65))

    def weight(u, v, data):
        return weights[(min(u, v), max(u, v))]

    return nx.shortest_path(G, SOURCE, DESTINATION, weight=weight)


def run_random_search(seed):
    rng = np.random.default_rng(seed)
    best_cost = float("inf")
    best_route = None
    unique_routes = set()
    evaluations = 0

    start_time = time.perf_counter()

    # Match the nominal 30 × 80 candidate-generation budget.
    for _ in range(N_ITERATIONS):
        for _ in range(N_PARTICLES):
            path = randomized_route(rng)
            cost = route_cost(path)
            evaluations += 1
            unique_routes.add(tuple(path))

            if cost < best_cost:
                best_cost = cost
                best_route = path.copy()

    runtime_ms = (time.perf_counter() - start_time) * 1000

    return {
        "best_cost": best_cost,
        "evaluations": evaluations,
        "unique_routes": len(unique_routes),
        "runtime_ms": runtime_ms,
        "best_route": " -> ".join(map(str, best_route)),
    }


reference_route = nx.shortest_path(G, SOURCE, DESTINATION, weight="cost")
optimum = route_cost(reference_route)

print("\n========== DYNAMIC RANDOMIZED SEARCH ==========")
print(f"Route: {SOURCE} -> {DESTINATION}")
print(f"Graph edges: {G.number_of_edges()}")
print(f"Disrupted edges: {len(affected_edges)}")
print(f"Dijkstra reference cost: {optimum:.8f}")
print(f"Candidate-generation budget: {N_PARTICLES * N_ITERATIONS}")

records = []

for trial in range(1, N_TRIALS + 1):
    seed = 800000 + trial
    result = run_random_search(seed)
    gap = (result["best_cost"] / optimum - 1) * 100

    records.append({
        "trial": trial,
        "seed": seed,
        "dijkstra_cost": optimum,
        "gap_percent": gap,
        **result,
    })

    print(
        f"Trial {trial:02d}: gap={gap:.4f}%, "
        f"unique routes={result['unique_routes']}, "
        f"runtime={result['runtime_ms']:.1f} ms"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(records)

results_path = OUTPUT_DIR / "dynamic_random_search_trials.csv"
results.to_csv(results_path, index=False)

summary = pd.DataFrame([{
    "trials": len(results),
    "optimal_rate_percent": (
        results["gap_percent"].abs().lt(1e-7).mean() * 100
    ),
    "mean_gap_percent": results["gap_percent"].mean(),
    "mean_runtime_ms": results["runtime_ms"].mean(),
    "mean_unique_routes": results["unique_routes"].mean(),
}])

summary_path = OUTPUT_DIR / "dynamic_random_search_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n========== SUMMARY ==========")
print(summary.round(4).to_string(index=False))
print(f"\nSaved trials: {results_path}")
print(f"Saved summary: {summary_path}")
