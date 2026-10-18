
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
PAIRS = [(11388, 14979)]

N_PARTICLES = 30
N_ITERATIONS = 80
N_TRIALS = 20

W = 0.70
C1 = 1.50
C2 = 1.50
MUTATION_RATE = 0.15

OUTPUT_DIR = Path("ml/regional_model")

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
    return sum(G[u][v]["cost"] for u, v in zip(path[:-1], path[1:]))


def route_valid(path, source, destination):
    return (
        bool(path)
        and path[0] == source
        and path[-1] == destination
        and len(path) == len(set(path))
        and all(G.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))
    )


def run_pso(source, destination, seed):
    rng = np.random.default_rng(seed)

    def randomized_route():
        weights = {}
        for u, v, data in G.edges(data=True):
            key = (min(u, v), max(u, v))
            weights[key] = data["cost"] * np.exp(rng.normal(0, 0.65))

        def weight(u, v, data):
            return weights[(min(u, v), max(u, v))]

        return nx.shortest_path(
            G, source, destination, weight=weight
        )

    def make_edit(current, target):
        cp = {node: i for i, node in enumerate(current)}
        tp = {node: i for i, node in enumerate(target)}
        shared = [node for node in target if node in cp]
        edits = []

        for i, start in enumerate(shared):
            for end in shared[i + 1:]:
                if cp[start] < cp[end] and tp[start] < tp[end]:
                    edits.append((
                        start, end, target[tp[start]:tp[end] + 1]
                    ))

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
        return candidate if route_valid(candidate, source, destination) else path

    # Initialize without giving PSO the Dijkstra route.
    particles = []

    for _ in range(N_PARTICLES):
        path = randomized_route()
        particles.append({
            "position": path,
            "velocity": [],
            "pbest": path.copy(),
            "pbest_cost": route_cost(path),
        })

    initial_routes = {tuple(p["position"]) for p in particles}
    best_particle = min(particles, key=lambda p: p["pbest_cost"])
    global_best = best_particle["pbest"].copy()
    global_best_cost = route_cost(global_best)
    initial_best_cost = global_best_cost

    start = time.perf_counter()

    for _ in range(N_ITERATIONS):
        for particle in particles:
            current = particle["position"]
            velocity = [
                edit for edit in particle["velocity"]
                if rng.random() < W
            ]

            if rng.random() < min(1.0, C1 / 2):
                edit = make_edit(current, particle["pbest"])
                if edit is not None:
                    velocity.append(edit)

            if rng.random() < min(1.0, C2 / 2):
                edit = make_edit(current, global_best)
                if edit is not None:
                    velocity.append(edit)

            candidate = current.copy()
            for edit in velocity:
                candidate = apply_edit(candidate, edit)

            if rng.random() < MUTATION_RATE:
                candidate = randomized_route()

            if not route_valid(candidate, source, destination):
                candidate = current.copy()

            particle["position"] = candidate
            particle["velocity"] = velocity
            cost = route_cost(candidate)

            if cost < particle["pbest_cost"]:
                particle["pbest"] = candidate.copy()
                particle["pbest_cost"] = cost

            if cost < global_best_cost:
                global_best = candidate.copy()
                global_best_cost = cost

    runtime_ms = (time.perf_counter() - start) * 1000

    return {
        "initial_best_cost": initial_best_cost,
        "final_best_cost": route_cost(global_best),
        "initial_unique_routes": len(initial_routes),
        "final_unique_routes": len({
            tuple(p["position"]) for p in particles
        }),
        "valid_route": route_valid(global_best, source, destination),
        "runtime_ms": runtime_ms,
        "final_route": " -> ".join(map(str, global_best)),
    }


# ---------------- RUN EXPERIMENT ----------------

records = []

for pair_index, (source, destination) in enumerate(PAIRS, start=1):
    reference = nx.shortest_path(G, source, destination, weight="cost")
    optimum = route_cost(reference)

    print(f"\nPair {pair_index}: {source} -> {destination}")
    print(f"Dijkstra reference cost: {optimum:.8f}")

    for trial in range(1, N_TRIALS + 1):
        seed = pair_index * 100000 + trial
        result = run_pso(source, destination, seed)

        initial_gap = (result["initial_best_cost"] / optimum - 1) * 100
        final_gap = (result["final_best_cost"] / optimum - 1) * 100

        record = {
            "pair_index": pair_index,
            "source": source,
            "destination": destination,
            "trial": trial,
            "seed": seed,
            "dijkstra_cost": optimum,
            "initial_gap_percent": initial_gap,
            "final_gap_percent": final_gap,
            "initial_improvement_percent": (
                (result["initial_best_cost"] - result["final_best_cost"])
                / result["initial_best_cost"] * 100
                if result["initial_best_cost"] else 0
            ),
            **result,
        }
        records.append(record)

        print(
            f"  Trial {trial:02d}: initial gap={initial_gap:.3f}%, "
            f"final gap={final_gap:.3f}%, "
            f"unique routes={result['initial_unique_routes']}, "
            f"runtime={result['runtime_ms']:.1f} ms"
        )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(records)
results_path = OUTPUT_DIR / "pso_diagnostic_trials.csv"
results.to_csv(results_path, index=False)

summary = results.groupby(
    ["pair_index", "source", "destination"]
).agg(
    trials=("trial", "count"),
    initial_optimal_rate=("initial_gap_percent", lambda x: (x.abs() < 1e-7).mean() * 100),
    final_optimal_rate=("final_gap_percent", lambda x: (x.abs() < 1e-7).mean() * 100),
    mean_initial_gap=("initial_gap_percent", "mean"),
    mean_final_gap=("final_gap_percent", "mean"),
    mean_initial_unique_routes=("initial_unique_routes", "mean"),
    mean_final_unique_routes=("final_unique_routes", "mean"),
    mean_runtime_ms=("runtime_ms", "mean"),
).reset_index()

summary_path = OUTPUT_DIR / "pso_diagnostic_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n========== DIAGNOSTIC SUMMARY ==========")
print(summary.to_string(index=False))
print(f"\nSaved trials: {results_path}")
print(f"Saved summary: {summary_path}")
