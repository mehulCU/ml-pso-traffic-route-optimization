
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ---------------- CONFIGURATION ----------------

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 316
DESTINATION = 336

N_PARTICLES = 30
N_ITERATIONS = 80

W = 0.70
C1 = 1.50
C2 = 1.50
RANDOM_MUTATION_RATE = 0.15
SEED = 42

rng = np.random.default_rng(SEED)

# ---------------- LOAD GRAPH ----------------

df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
G = nx.Graph()

for row in df.itertuples(index=False):
    u, v = int(row.start_node), int(row.end_node)
    if u == v:
        continue

    cost = float(row.traffic_aware_cost_minutes)
    distance = float(row.distance)
    segment_id = str(row.segment_id)

    if G.has_edge(u, v):
        if cost < G[u][v]["cost"]:
            G[u][v].update(
                cost=cost, distance=distance, segment_id=segment_id
            )
    else:
        G.add_edge(
            u, v, cost=cost, distance=distance,
            segment_id=segment_id
        )

if not nx.has_path(G, SOURCE, DESTINATION):
    raise ValueError("No route exists between source and destination.")

def route_cost(path):
    return sum(G[u][v]["cost"] for u, v in zip(path[:-1], path[1:]))

def route_is_valid(path):
    return (
        bool(path)
        and path[0] == SOURCE
        and path[-1] == DESTINATION
        and len(path) == len(set(path))
        and all(G.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))
    )

def randomized_route():
    random_weights = {}
    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        random_weights[key] = data["cost"] * np.exp(rng.normal(0, 0.65))

    def weight(u, v, data):
        return random_weights[(min(u, v), max(u, v))]

    return nx.shortest_path(G, SOURCE, DESTINATION, weight=weight)

def make_route_edit(current, target):
    cp = {node: i for i, node in enumerate(current)}
    tp = {node: i for i, node in enumerate(target)}
    shared = [node for node in target if node in cp]
    edits = []

    for i, start_node in enumerate(shared):
        for end_node in shared[i + 1:]:
            ci, cj = cp[start_node], cp[end_node]
            ti, tj = tp[start_node], tp[end_node]

            if ci < cj and ti < tj:
                replacement = target[ti:tj + 1]
                if len(replacement) >= 2:
                    edits.append((start_node, end_node, replacement))

    return edits[rng.integers(len(edits))] if edits else None

def apply_edit(path, edit):
    if edit is None:
        return path

    start_node, end_node, replacement = edit
    if start_node not in path or end_node not in path:
        return path

    i, j = path.index(start_node), path.index(end_node)
    if i >= j:
        return path

    candidate = path[:i] + list(replacement) + path[j + 1:]
    return candidate if route_is_valid(candidate) else path

def apply_velocity(path, velocity):
    result = path
    for edit in velocity:
        result = apply_edit(result, edit)
    return result

# ---------------- REFERENCE ONLY ----------------
# Dijkstra is measured for comparison but is NOT given to PSO.

dijkstra_path = nx.shortest_path(G, SOURCE, DESTINATION, weight="cost")
dijkstra_cost = route_cost(dijkstra_path)

# ---------------- INITIALIZE PSO ----------------

particles = []
for _ in range(N_PARTICLES):
    path = randomized_route()
    cost = route_cost(path)
    particles.append({
        "position": path,
        "velocity": [],
        "personal_best": path.copy(),
        "personal_best_cost": cost,
    })

# Crucially, initialize from the particles only.
global_best_particle = min(
    particles, key=lambda p: p["personal_best_cost"]
)
global_best = global_best_particle["personal_best"].copy()
global_best_cost = route_cost(global_best)

initial_best_cost = global_best_cost
initial_unique_routes = len({
    tuple(p["position"]) for p in particles
})

history = [{
    "iteration": 0,
    "best_cost_minutes": global_best_cost,
    "unique_current_routes": initial_unique_routes,
}]

# ---------------- DISCRETE PSO ----------------

start_time = time.perf_counter()

for iteration in range(N_ITERATIONS):
    for particle in particles:
        current = particle["position"]
        pbest = particle["personal_best"]

        velocity = [
            edit for edit in particle["velocity"]
            if rng.random() < W
        ]

        if rng.random() < min(1.0, C1 / 2):
            edit = make_route_edit(current, pbest)
            if edit is not None:
                velocity.append(edit)

        if rng.random() < min(1.0, C2 / 2):
            edit = make_route_edit(current, global_best)
            if edit is not None:
                velocity.append(edit)

        candidate = apply_velocity(current, velocity)

        if rng.random() < RANDOM_MUTATION_RATE:
            candidate = randomized_route()

        if not route_is_valid(candidate):
            candidate = current.copy()

        particle["position"] = candidate
        particle["velocity"] = velocity

        cost = route_cost(candidate)

        if cost < particle["personal_best_cost"]:
            particle["personal_best"] = candidate.copy()
            particle["personal_best_cost"] = cost

        if cost < global_best_cost:
            global_best = candidate.copy()
            global_best_cost = cost

    history.append({
        "iteration": iteration + 1,
        "best_cost_minutes": global_best_cost,
        "unique_current_routes": len({
            tuple(p["position"]) for p in particles
        }),
    })

pso_runtime_ms = (time.perf_counter() - start_time) * 1000
final_cost = route_cost(global_best)

# ---------------- REPORT ----------------

improvement = (
    (initial_best_cost - final_cost) / initial_best_cost * 100
    if initial_best_cost else 0.0
)
gap = (
    (final_cost - dijkstra_cost) / dijkstra_cost * 100
    if dijkstra_cost else 0.0
)

print("\n========== UNBIASED PSO DIAGNOSTIC ==========")
print(f"Source -> destination: {SOURCE} -> {DESTINATION}")
print(f"Particles: {N_PARTICLES}; iterations: {N_ITERATIONS}")
print(f"Initial unique routes: {initial_unique_routes}")
print(f"Initial best cost: {initial_best_cost:.8f} min")
print(f"Final best cost:   {final_cost:.8f} min")
print(f"Improvement:       {improvement:.4f}%")
print(f"Dijkstra cost:     {dijkstra_cost:.8f} min")
print(f"Final optimality gap: {gap:.4f}%")
print(f"PSO runtime:       {pso_runtime_ms:.3f} ms")
print(f"Valid final route: {route_is_valid(global_best)}")
print("Final route:", global_best)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "unbiased_pso_diagnostic_convergence.csv",
    index=False,
)

print("\nSaved convergence diagnostics to:")
print(OUTPUT_DIR / "unbiased_pso_diagnostic_convergence.csv")
