
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

N_PAIRS = 5
N_PARTICLES = 30
N_ITERATIONS = 80
N_TRIALS = 5

W = 0.70
C1 = 1.50
C2 = 1.50
MUTATION_RATE = 0.15

df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
G = nx.Graph()

# Same parallel-edge reduction as the Dijkstra baseline.
for row in df.itertuples(index=False):
    u, v = int(row.start_node), int(row.end_node)
    if u == v:
        continue

    cost = float(row.traffic_aware_cost_minutes)
    attrs = {
        "cost": cost,
        "segment_id": str(row.segment_id),
        "distance": float(row.distance),
    }

    if not G.has_edge(u, v) or cost < G[u][v]["cost"]:
        G.add_edge(u, v, **attrs)

def route_cost(path):
    return sum(G[u][v]["cost"] for u, v in zip(path[:-1], path[1:]))

def route_distance(path):
    return sum(G[u][v]["distance"] for u, v in zip(path[:-1], path[1:]))

def is_valid(path, source, destination):
    return (
        path[0] == source
        and path[-1] == destination
        and len(path) == len(set(path))
        and all(G.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))
    )

# Find pairs with multiple distinct routes.
def find_pairs():
    junctions = sorted(
        node for node, degree in G.degree() if degree >= 3
    )
    pairs = []

    for source in junctions:
        lengths = nx.single_source_dijkstra_path_length(
            G, source, weight="cost"
        )

        destinations = [
            node for node in junctions
            if node != source
            and node in lengths
            and 0.10 <= lengths[node] <= 0.80
        ]

        for destination in destinations:
            try:
                paths = []
                generator = nx.shortest_simple_paths(
                    G, source, destination, weight="cost"
                )

                for path in generator:
                    paths.append(path)
                    if len(paths) == 3:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            if len(paths) < 2:
                continue

            costs = [route_cost(path) for path in paths]

            # Require at least one meaningfully different alternative.
            if max(costs) < min(costs) * 1.02:
                continue

            pairs.append((source, destination))
            if len(pairs) >= N_PAIRS:
                return pairs

    return pairs

def run_pso(source, destination, seed):
    rng = np.random.default_rng(seed)

    def randomized_route():
        perturbed = {}

        for u, v, data in G.edges(data=True):
            key = (min(u, v), max(u, v))
            perturbed[key] = data["cost"] * np.exp(rng.normal(0, 0.65))

        def weight(u, v, data):
            return perturbed[(min(u, v), max(u, v))]

        return nx.shortest_path(
            G, source, destination, weight=weight
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
                        target[target_pos[start]:target_pos[end] + 1],
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
        return candidate if is_valid(candidate, source, destination) else path

    # Initialize particles only from randomized feasible paths.
    # Dijkstra's route is NOT used to initialize or seed PSO.
    particles = []

    for _ in range(N_PARTICLES):
        path = randomized_route()
        particles.append({
            "position": path,
            "velocity": [],
            "pbest": path.copy(),
            "pbest_cost": route_cost(path),
        })

    best_particle = min(particles, key=lambda p: p["pbest_cost"])
    global_best = best_particle["pbest"].copy()
    global_best_cost = best_particle["pbest_cost"]

    start_time = time.perf_counter()

    for _ in range(N_ITERATIONS):
        for particle in particles:
            current = particle["position"]
            pbest = particle["pbest"]

            # Inertia: retain some previous discrete edit operations.
            velocity = [
                edit for edit in particle["velocity"]
                if rng.random() < W
            ]

            # Cognitive update toward this particle's best route.
            if rng.random() < min(1.0, C1 / 2):
                edit = make_edit(current, pbest)
                if edit is not None:
                    velocity.append(edit)

            # Social update toward the swarm's best route.
            if rng.random() < min(1.0, C2 / 2):
                edit = make_edit(current, global_best)
                if edit is not None:
                    velocity.append(edit)

            candidate = current.copy()
            for edit in velocity:
                candidate = apply_edit(candidate, edit)

            # Randomized feasible-path mutation for exploration.
            if rng.random() < MUTATION_RATE:
                candidate = randomized_route()

            if not is_valid(candidate, source, destination):
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

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "valid_route": is_valid(global_best, source, destination),
        "pso_cost": route_cost(global_best),
        "pso_distance_m": route_distance(global_best),
        "pso_runtime_ms": elapsed_ms,
        "route_nodes": " -> ".join(map(str, global_best)),
    }

# ---------------- BENCHMARK ----------------

pairs = find_pairs()

if not pairs:
    raise RuntimeError("No suitable source-destination pairs found.")

print("Selected pairs:", pairs)
results = []

for pair_index, (source, destination) in enumerate(pairs, start=1):
    # Dijkstra is used only as the reference optimum.
    start = time.perf_counter()
    reference_path = nx.shortest_path(
        G, source, destination, weight="cost"
    )
    dijkstra_runtime_ms = (time.perf_counter() - start) * 1000
    dijkstra_cost = route_cost(reference_path)

    print(
        f"\nPair {pair_index}: {source} -> {destination}; "
        f"Dijkstra cost={dijkstra_cost:.8f}"
    )

    for trial in range(1, N_TRIALS + 1):
        seed = pair_index * 1000 + trial

        outcome = run_pso(source, destination, seed)
        gap = (
            (outcome["pso_cost"] - dijkstra_cost)
            / dijkstra_cost * 100
        )

        record = {
            "pair_index": pair_index,
            "source": source,
            "destination": destination,
            "trial": trial,
            "seed": seed,
            "dijkstra_cost": dijkstra_cost,
            "dijkstra_runtime_ms": dijkstra_runtime_ms,
            "optimality_gap_percent": gap,
            **outcome,
        }
        results.append(record)

        print(
            f"  Trial {trial}: valid={outcome['valid_route']}, "
            f"PSO={outcome['pso_cost']:.8f}, "
            f"gap={gap:.3f}%, "
            f"runtime={outcome['pso_runtime_ms']:.1f} ms"
        )

results_df = pd.DataFrame(results)
results_path = OUTPUT_DIR / "pso_multi_pair_results.csv"
results_df.to_csv(results_path, index=False)

summary = results_df.groupby(
    ["pair_index", "source", "destination"]
).agg(
    valid_routes=("valid_route", "sum"),
    mean_pso_cost=("pso_cost", "mean"),
    best_pso_cost=("pso_cost", "min"),
    mean_gap_percent=("optimality_gap_percent", "mean"),
    mean_pso_runtime_ms=("pso_runtime_ms", "mean"),
).reset_index()

summary_path = OUTPUT_DIR / "pso_multi_pair_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n========== SUMMARY BY PAIR ==========")
print(summary.to_string(index=False))
print("\nTotal valid routes:", results_df["valid_route"].sum(), "/", len(results_df))
print("Overall mean optimality gap:",
      f"{results_df['optimality_gap_percent'].mean():.3f}%")
print("Saved trial results:", results_path)
print("Saved summary:", summary_path)
