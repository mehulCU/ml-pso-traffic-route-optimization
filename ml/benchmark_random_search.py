import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
SOURCE = 11388
DESTINATION = 14979

N_PARTICLES = 30
N_ITERATIONS = 80
N_TRIALS = 20

OUTPUT_DIR = Path("ml/regional_model")


# Build the same graph used by the PSO benchmark.
df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
G = nx.Graph()

for row in df.itertuples(index=False):
    u, v = int(row.start_node), int(row.end_node)

    if u == v:
        continue

    cost = float(row.traffic_aware_cost_minutes)
    attrs = {
        "cost": cost,
        "distance": float(row.distance),
        "segment_id": str(row.segment_id),
    }

    if not G.has_edge(u, v) or cost < G[u][v]["cost"]:
        G.add_edge(u, v, **attrs)


def route_cost(path):
    return sum(
        G[u][v]["cost"]
        for u, v in zip(path[:-1], path[1:])
    )


def randomized_route(rng):
    """Generate a route using randomly perturbed edge costs."""
    weights = {}

    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        weights[key] = data["cost"] * np.exp(
            rng.normal(0, 0.65)
        )

    def weight(u, v, data):
        return weights[(min(u, v), max(u, v))]

    return nx.shortest_path(
        G,
        SOURCE,
        DESTINATION,
        weight=weight,
    )


def run_random_search(seed):
    rng = np.random.default_rng(seed)
    best_route = None
    best_cost = float("inf")
    unique_routes = set()
    evaluations = 0

    start = time.perf_counter()

    # Match the PSO experiment's 30 × 80 candidate budget.
    for _ in range(N_ITERATIONS):
        for _ in range(N_PARTICLES):
            path = randomized_route(rng)
            cost = route_cost(path)
            evaluations += 1
            unique_routes.add(tuple(path))

            if cost < best_cost:
                best_cost = cost
                best_route = path.copy()

    runtime_ms = (time.perf_counter() - start) * 1000

    return {
        "best_cost": best_cost,
        "evaluations": evaluations,
        "unique_routes": len(unique_routes),
        "runtime_ms": runtime_ms,
        "best_route": " -> ".join(map(str, best_route)),
    }


# Dijkstra is used only as the reference optimum.
reference_route = nx.shortest_path(
    G, SOURCE, DESTINATION, weight="cost"
)
optimum = route_cost(reference_route)

print(f"Route: {SOURCE} -> {DESTINATION}")
print(f"Dijkstra reference cost: {optimum:.8f}")
print(f"Randomized-search budget: {N_PARTICLES * N_ITERATIONS} candidates")

records = []

for trial in range(1, N_TRIALS + 1):
    seed = 900000 + trial
    result = run_random_search(seed)

    gap = (result["best_cost"] / optimum - 1) * 100

    records.append({
        "trial": trial,
        "seed": seed,
        "dijkstra_cost": optimum,
        "best_cost": result["best_cost"],
        "gap_percent": gap,
        **{k: v for k, v in result.items() if k != "best_route"},
        "best_route": result["best_route"],
    })

    print(
        f"Trial {trial:02d}: "
        f"gap={gap:.4f}%, "
        f"unique routes={result['unique_routes']}, "
        f"runtime={result['runtime_ms']:.1f} ms"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(records)

results_path = OUTPUT_DIR / "random_search_trials.csv"
results.to_csv(results_path, index=False)

summary = results.agg({
    "gap_percent": ["mean", "min", "max"],
    "runtime_ms": ["mean", "min", "max"],
    "unique_routes": ["mean", "min", "max"],
}).round(4)

print("\n========== RANDOMIZED SEARCH SUMMARY ==========")
print(summary.to_string())
print(
    "\nOptimal-cost rate:",
    f"{(results['gap_percent'].abs() < 1e-7).mean() * 100:.1f}%"
)
print(f"Saved results: {results_path}")
