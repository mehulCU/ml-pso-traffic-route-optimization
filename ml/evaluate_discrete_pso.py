
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ---------------- CONFIGURATION ----------------

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 11388
DESTINATION = 14979

N_PARTICLES = 30
N_ITERATIONS = 80

W = 0.70
C1 = 1.50
C2 = 1.50

RANDOM_MUTATION_RATE = 0.15
PATIENCE = 15
MIN_IMPROVEMENT = 1e-9
SEED = 42

rng = np.random.default_rng(SEED)

# ---------------- LOAD GRAPH ----------------

df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
G = nx.Graph()

# Keep the cheapest parallel segment for each node pair.
for row in df.itertuples(index=False):
    u = int(row.start_node)
    v = int(row.end_node)

    if u == v:
        continue

    cost = float(row.traffic_aware_cost_minutes)
    distance = float(row.distance)
    segment_id = str(row.segment_id)

    if not G.has_edge(u, v) or cost < G[u][v]["cost"]:
        G.add_edge(
            u,
            v,
            cost=cost,
            distance=distance,
            segment_id=segment_id,
        )

if SOURCE not in G or DESTINATION not in G:
    raise ValueError("Source or destination is not in the graph.")

if not nx.has_path(G, SOURCE, DESTINATION):
    raise ValueError("No route exists between source and destination.")

# ---------------- ROUTE UTILITIES ----------------

def route_cost(path):
    return sum(
        G[u][v]["cost"]
        for u, v in zip(path[:-1], path[1:])
    )


def route_distance(path):
    return sum(
        G[u][v]["distance"]
        for u, v in zip(path[:-1], path[1:])
    )


def route_is_valid(path):
    if not path:
        return False

    if path[0] != SOURCE or path[-1] != DESTINATION:
        return False

    if len(path) != len(set(path)):
        return False

    return all(
        G.has_edge(u, v)
        for u, v in zip(path[:-1], path[1:])
    )


def randomized_route():
    """Generate a diverse route using perturbed costs."""
    random_weights = {}

    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        noise = rng.normal(0, 0.65)
        random_weights[key] = data["cost"] * np.exp(noise)

    def weight(u, v, data):
        return random_weights[(min(u, v), max(u, v))]

    return nx.shortest_path(
        G, SOURCE, DESTINATION, weight=weight
    )


# ---------------- DISCRETE VELOCITY ----------------

def make_route_edit(current, target):
    """Copy a target subpath between two shared ordered anchors."""
    current_positions = {
        node: i for i, node in enumerate(current)
    }
    target_positions = {
        node: i for i, node in enumerate(target)
    }

    shared = [
        node for node in target
        if node in current_positions
    ]

    possible_edits = []

    for i, start_node in enumerate(shared):
        for end_node in shared[i + 1:]:
            ci = current_positions[start_node]
            cj = current_positions[end_node]
            ti = target_positions[start_node]
            tj = target_positions[end_node]

            if ci < cj and ti < tj:
                replacement = target[ti:tj + 1]

                if len(replacement) >= 2:
                    possible_edits.append(
                        (start_node, end_node, replacement)
                    )

    if not possible_edits:
        return None

    return possible_edits[
        int(rng.integers(len(possible_edits)))
    ]


def apply_edit(path, edit):
    if edit is None:
        return path

    start_node, end_node, replacement = edit

    if start_node not in path or end_node not in path:
        return path

    i = path.index(start_node)
    j = path.index(end_node)

    if i >= j:
        return path

    candidate = path[:i] + list(replacement) + path[j + 1:]

    return candidate if route_is_valid(candidate) else path


def apply_velocity(path, velocity):
    result = path

    for edit in velocity:
        result = apply_edit(result, edit)

    return result


# ---------------- DIJKSTRA REFERENCE ----------------

start = time.perf_counter()
dijkstra_path = nx.shortest_path(
    G, SOURCE, DESTINATION, weight="cost"
)
dijkstra_runtime = time.perf_counter() - start

dijkstra_cost = route_cost(dijkstra_path)
dijkstra_distance = route_distance(dijkstra_path)

# ---------------- INITIALIZE SWARM ----------------

particles = []

for _ in range(N_PARTICLES):
    path = randomized_route()
    cost = route_cost(path)

    particles.append({
        "position": path.copy(),
        "velocity": [],
        "personal_best": path.copy(),
        "personal_best_cost": cost,
    })

best_particle = min(
    particles,
    key=lambda p: p["personal_best_cost"],
)

global_best = best_particle["personal_best"].copy()
global_best_cost = route_cost(global_best)

initial_best_cost = global_best_cost
initial_routes = {
    tuple(p["position"]) for p in particles
}

initial_gap = (
    max(0.0, (initial_best_cost - dijkstra_cost)
        / dijkstra_cost * 100)
    if dijkstra_cost > 0 else 0.0
)

# 0 means an optimal route was already present initially.
first_optimal_iteration = (
    0 if initial_best_cost <= dijkstra_cost + MIN_IMPROVEMENT
    else None
)

# ---------------- PSO LOOP ----------------

history = []
iterations_without_improvement = 0

start = time.perf_counter()

for iteration in range(N_ITERATIONS):
    improved_this_iteration = False

    for particle in particles:
        current = particle["position"]
        pbest = particle["personal_best"]

        # Inertia: retain some previous edit operations.
        velocity = [
            edit for edit in particle["velocity"]
            if rng.random() < W
        ]

        # Cognitive component.
        if rng.random() < min(1.0, C1 / 2):
            edit = make_route_edit(current, pbest)
            if edit is not None:
                velocity.append(edit)

        # Social component.
        if rng.random() < min(1.0, C2 / 2):
            edit = make_route_edit(current, global_best)
            if edit is not None:
                velocity.append(edit)

        candidate = apply_velocity(current, velocity)

        # Random exploration.
        if rng.random() < RANDOM_MUTATION_RATE:
            candidate = randomized_route()
            velocity = []

        if not route_is_valid(candidate):
            candidate = current.copy()
            velocity = []

        particle["position"] = candidate
        particle["velocity"] = velocity

        candidate_cost = route_cost(candidate)

        # Update personal best.
        if candidate_cost < (
            particle["personal_best_cost"] - MIN_IMPROVEMENT
        ):
            particle["personal_best"] = candidate.copy()
            particle["personal_best_cost"] = candidate_cost

        # Update global best.
        if candidate_cost < global_best_cost - MIN_IMPROVEMENT:
            global_best = candidate.copy()
            global_best_cost = candidate_cost
            improved_this_iteration = True

            if (
                first_optimal_iteration is None
                and global_best_cost <= dijkstra_cost + MIN_IMPROVEMENT
            ):
                first_optimal_iteration = iteration + 1

    history.append({
        "iteration": iteration + 1,
        "best_cost_minutes": global_best_cost,
        "optimality_gap_percent": (
            max(0.0, (global_best_cost - dijkstra_cost)
                / dijkstra_cost * 100)
            if dijkstra_cost > 0 else 0.0
        ),
    })

    if improved_this_iteration:
        iterations_without_improvement = 0
    else:
        iterations_without_improvement += 1

    if iterations_without_improvement >= PATIENCE:
        print(
            f"Early stopping at iteration {iteration + 1}: "
            f"no improvement for {PATIENCE} consecutive iterations."
        )
        break

pso_runtime = time.perf_counter() - start

# ---------------- FINAL RESULTS ----------------

pso_cost = route_cost(global_best)
pso_distance = route_distance(global_best)

optimality_gap = (
    max(0.0, (pso_cost - dijkstra_cost) / dijkstra_cost * 100)
    if dijkstra_cost > 0 else 0.0
)

print("\n========== DISCRETE PSO VS DIJKSTRA ==========")
print(f"Source: {SOURCE}")
print(f"Destination: {DESTINATION}")
print(f"Particles: {N_PARTICLES}")
print(f"Maximum iterations: {N_ITERATIONS}")
print(f"Iterations completed: {len(history)}")
print(f"First optimal iteration: {first_optimal_iteration}")
print(f"Distinct initial routes: {len(initial_routes)}")
print(f"Initial best cost: {initial_best_cost:.8f} minutes")
print(f"Initial optimality gap: {initial_gap:.6f}%")
print(f"Final improvement: {initial_best_cost - pso_cost:.8f} cost-minutes")

print("\nDIJKSTRA")
print(f"Cost: {dijkstra_cost:.8f} minutes")
print(f"Distance: {dijkstra_distance:.2f} m")
print(f"Runtime: {dijkstra_runtime * 1000:.4f} ms")

print("\nDISCRETE PSO")
print(f"Valid route: {route_is_valid(global_best)}")
print(f"Cost: {pso_cost:.8f} minutes")
print(f"Distance: {pso_distance:.2f} m")
print(f"Runtime: {pso_runtime * 1000:.4f} ms")
print(f"Optimality gap: {optimality_gap:.6f}%")

# ---------------- SAVE OUTPUTS ----------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

route_rows = []

for u, v in zip(global_best[:-1], global_best[1:]):
    edge = G[u][v]
    route_rows.append({
        "from_node": u,
        "to_node": v,
        "segment_id": edge["segment_id"],
        "distance_m": edge["distance"],
        "cost_minutes": edge["cost"],
    })

pd.DataFrame(route_rows).to_csv(
    OUTPUT_DIR / "discrete_pso_best_route.csv",
    index=False,
)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "discrete_pso_convergence.csv",
    index=False,
)

print("\nSaved route:", OUTPUT_DIR / "discrete_pso_best_route.csv")
print("Saved convergence:", OUTPUT_DIR / "discrete_pso_convergence.csv")
