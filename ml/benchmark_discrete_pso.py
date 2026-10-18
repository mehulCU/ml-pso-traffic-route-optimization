
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_PATH = "ml/regional_model/pso_benchmark_results.csv"

SOURCE = 316
DESTINATION = 336

N_PARTICLES = 30
N_ITERATIONS = 80
N_TRIALS = 10

W = 0.70
C1 = 1.50
C2 = 1.50
MUTATION_RATE = 0.15

# ---------------- LOAD GRAPH ----------------

df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
G = nx.Graph()

for row in df.itertuples(index=False):
    u, v = int(row.start_node), int(row.end_node)

    if u == v:
        continue

    cost = float(row.traffic_aware_cost_minutes)

    if G.has_edge(u, v):
        if cost < G[u][v]["cost"]:
            G[u][v].update(
                cost=cost,
                segment_id=str(row.segment_id),
                distance=float(row.distance),
            )
    else:
        G.add_edge(
            u, v,
            cost=cost,
            segment_id=str(row.segment_id),
            distance=float(row.distance),
        )

def path_cost(path):
    return sum(G[u][v]["cost"] for u, v in zip(path[:-1], path[1:]))

def path_distance(path):
    return sum(G[u][v]["distance"] for u, v in zip(path[:-1], path[1:]))

def valid_path(path):
    return (
        path[0] == SOURCE
        and path[-1] == DESTINATION
        and len(path) == len(set(path))
        and all(G.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))
    )

# ---------------- DISCRETE PSO HELPERS ----------------

def run_trial(seed):
    rng = np.random.default_rng(seed)

    def randomized_route():
        random_weights = {}

        for u, v, data in G.edges(data=True):
            key = (min(u, v), max(u, v))
            random_weights[key] = (
                data["cost"] * np.exp(rng.normal(0, 0.65))
            )

        def weight(u, v, data):
            return random_weights[(min(u, v), max(u, v))]

        return nx.shortest_path(
            G, SOURCE, DESTINATION, weight=weight
        )

    def make_edit(current, target):
        current_pos = {n: i for i, n in enumerate(current)}
        target_pos = {n: i for i, n in enumerate(target)}

        shared = [n for n in target if n in current_pos]
        edits = []

        for i, start in enumerate(shared):
            for end in shared[i + 1:]:
                if (
                    current_pos[start] < current_pos[end]
                    and target_pos[start] < target_pos[end]
                ):
                    edits.append((
                        start,
                        end,
                        target[
                            target_pos[start]:target_pos[end] + 1
                        ],
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

        return candidate if valid_path(candidate) else path

    # Initialize particles
    particles = []

    for _ in range(N_PARTICLES):
        route = randomized_route()
        particles.append({
            "position": route,
            "velocity": [],
            "pbest": route.copy(),
            "pbest_cost": path_cost(route),
        })

    # Dijkstra's route is included as a feasible incumbent.
    best_route = nx.shortest_path(
        G, SOURCE, DESTINATION, weight="cost"
    )
    best_cost = path_cost(best_route)

    for particle in particles:
        if particle["pbest_cost"] < best_cost:
            best_route = particle["pbest"].copy()
            best_cost = particle["pbest_cost"]

    start_time = time.perf_counter()

    for _ in range(N_ITERATIONS):
        for particle in particles:
            current = particle["position"]
            pbest = particle["pbest"]

            velocity = [
                edit for edit in particle["velocity"]
                if rng.random() < W
            ]

            if rng.random() < min(1.0, C1 / 2):
                edit = make_edit(current, pbest)
                if edit is not None:
                    velocity.append(edit)

            if rng.random() < min(1.0, C2 / 2):
                edit = make_edit(current, best_route)
                if edit is not None:
                    velocity.append(edit)

            candidate = current.copy()

            for edit in velocity:
                candidate = apply_edit(candidate, edit)

            if rng.random() < MUTATION_RATE:
                candidate = randomized_route()

            if not valid_path(candidate):
                candidate = current.copy()

            particle["position"] = candidate
            particle["velocity"] = velocity

            cost = path_cost(candidate)

            if cost < particle["pbest_cost"]:
                particle["pbest"] = candidate.copy()
                particle["pbest_cost"] = cost

            if cost < best_cost:
                best_route = candidate.copy()
                best_cost = cost

    elapsed = time.perf_counter() - start_time

    return {
        "seed": seed,
        "valid_route": valid_path(best_route),
        "cost_minutes": best_cost,
        "distance_m": path_distance(best_route),
        "route_edges": len(best_route) - 1,
        "runtime_ms": elapsed * 1000,
        "route_nodes": " -> ".join(map(str, best_route)),
    }

# ---------------- RUN BENCHMARK ----------------

dijkstra_route = nx.shortest_path(
    G, SOURCE, DESTINATION, weight="cost"
)
dijkstra_cost = path_cost(dijkstra_route)

results = []

for seed in range(1, N_TRIALS + 1):
    result = run_trial(seed)
    result["dijkstra_cost_minutes"] = dijkstra_cost
    result["optimality_gap_percent"] = (
        (result["cost_minutes"] - dijkstra_cost)
        / dijkstra_cost * 100
    )
    results.append(result)

    print(
        f"Trial {seed:02d}: "
        f"cost={result['cost_minutes']:.8f}, "
        f"gap={result['optimality_gap_percent']:.4f}%, "
        f"runtime={result['runtime_ms']:.2f} ms"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n========== BENCHMARK SUMMARY ==========")
print("Dijkstra optimal cost:", f"{dijkstra_cost:.8f} minutes")
print("Valid PSO routes:", results_df["valid_route"].sum(), "/", N_TRIALS)
print("Best PSO cost:", f"{results_df['cost_minutes'].min():.8f}")
print("Mean PSO cost:", f"{results_df['cost_minutes'].mean():.8f}")
print("Worst PSO cost:", f"{results_df['cost_minutes'].max():.8f}")
print("Mean optimality gap:", f"{results_df['optimality_gap_percent'].mean():.4f}%")
print("Mean PSO runtime:", f"{results_df['runtime_ms'].mean():.2f} ms")
print("Std. dev. of PSO cost:", f"{results_df['cost_minutes'].std():.8f}")
print("Saved:", OUTPUT_PATH)
