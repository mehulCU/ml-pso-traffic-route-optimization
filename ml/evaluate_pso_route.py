
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# ---------------- CONFIGURATION ----------------
DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 316
DESTINATION = 336

N_PARTICLES = 30
N_ITERATIONS = 80

W = 0.7       # Inertia
C1 = 1.5      # Cognitive coefficient
C2 = 1.5      # Social coefficient
BETA = 1.0    # Strength of random-key edge-weight changes

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------- LOAD GRAPH ----------------
df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
G = nx.Graph()

# Match the parallel-edge reduction used by Dijkstra.
for row in df.itertuples(index=False):
    u = int(row.start_node)
    v = int(row.end_node)

    if u == v:
        continue

    cost = float(row.traffic_aware_cost_minutes)
    segment_id = str(row.segment_id)

    if G.has_edge(u, v):
        if cost < G[u][v]["base_cost"]:
            G[u][v].update(
                base_cost=cost,
                segment_id=segment_id,
                distance=float(row.distance),
            )
    else:
        G.add_edge(
            u, v,
            base_cost=cost,
            segment_id=segment_id,
            distance=float(row.distance),
        )

if SOURCE not in G or DESTINATION not in G:
    raise ValueError("Source or destination is not in the graph.")

if not nx.has_path(G, SOURCE, DESTINATION):
    raise ValueError("No connected route exists.")

# Assign a stable index to every undirected edge.
edge_list = sorted(
    (min(u, v), max(u, v))
    for u, v in G.edges()
)
edge_index = {edge: i for i, edge in enumerate(edge_list)}
N_EDGES = len(edge_list)

base_costs = np.array([
    G[u][v]["base_cost"] for u, v in edge_list
])

# ---------------- ROUTE DECODING ----------------
def decode_route(position):
    """
    Decode a particle's edge-priority vector into a valid route.

    The particle modifies edge weights for route generation.
    Fitness is always measured using the original base costs.
    """
    def modified_weight(u, v, attributes):
        key = (min(u, v), max(u, v))
        idx = edge_index[key]

        # Exponential mapping keeps modified weights positive.
        multiplier = np.exp(BETA * position[idx])
        return attributes["base_cost"] * multiplier

    path = nx.shortest_path(
        G,
        SOURCE,
        DESTINATION,
        weight=modified_weight,
    )

    return path


def evaluate_route(path):
    """Evaluate a route using the original traffic-aware costs."""
    cost = 0.0
    distance = 0.0
    segment_ids = []

    for u, v in zip(path, path[1:]):
        edge = G[u][v]
        cost += edge["base_cost"]
        distance += edge["distance"]
        segment_ids.append(edge["segment_id"])

    return cost, distance, segment_ids


# ---------------- DIJKSTRA REFERENCE ----------------
start = time.perf_counter()

dijkstra_path = nx.shortest_path(
    G, SOURCE, DESTINATION, weight="base_cost"
)
dijkstra_time = time.perf_counter() - start

dijkstra_cost, dijkstra_distance, _ = evaluate_route(dijkstra_path)

# ---------------- INITIALIZE PARTICLES ----------------
# Position: one random key per graph edge.
# Velocity: change to those keys at each iteration.
positions = rng.uniform(
    -0.35, 0.35, size=(N_PARTICLES, N_EDGES)
)
velocities = rng.uniform(
    -0.05, 0.05, size=(N_PARTICLES, N_EDGES)
)

personal_best_positions = positions.copy()
personal_best_costs = np.full(N_PARTICLES, np.inf)

global_best_position = None
global_best_cost = np.inf
global_best_path = None
global_best_distance = None
global_best_segments = None

# Evaluate initial population.
for i in range(N_PARTICLES):
    path = decode_route(positions[i])
    cost, distance, segments = evaluate_route(path)

    personal_best_costs[i] = cost

    if cost < global_best_cost:
        global_best_cost = cost
        global_best_position = positions[i].copy()
        global_best_path = path
        global_best_distance = distance
        global_best_segments = segments

# ---------------- PSO ITERATIONS ----------------
start = time.perf_counter()
history = []

for iteration in range(N_ITERATIONS):
    for i in range(N_PARTICLES):

        r1 = rng.random(N_EDGES)
        r2 = rng.random(N_EDGES)

        # Standard PSO velocity update.
        velocities[i] = (
            W * velocities[i]
            + C1 * r1 * (
                personal_best_positions[i] - positions[i]
            )
            + C2 * r2 * (
                global_best_position - positions[i]
            )
        )

        # Limit velocity and position to avoid numerical extremes.
        velocities[i] = np.clip(velocities[i], -0.5, 0.5)

        positions[i] = np.clip(
            positions[i] + velocities[i],
            -2.0,
            2.0,
        )

        # Decode into a connected route and score it.
        path = decode_route(positions[i])
        cost, distance, segments = evaluate_route(path)

        # Update personal best.
        if cost < personal_best_costs[i]:
            personal_best_costs[i] = cost
            personal_best_positions[i] = positions[i].copy()

        # Update global best.
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_position = positions[i].copy()
            global_best_path = path
            global_best_distance = distance
            global_best_segments = segments

    history.append({
        "iteration": iteration + 1,
        "best_cost_minutes": global_best_cost,
    })

pso_time = time.perf_counter() - start

# ---------------- REPORT RESULTS ----------------
gap_percent = (
    (global_best_cost - dijkstra_cost) / dijkstra_cost * 100
    if dijkstra_cost > 0 else 0.0
)

print("\n========== PSO VS DIJKSTRA ==========")
print("Source:", SOURCE)
print("Destination:", DESTINATION)
print("Particles:", N_PARTICLES)
print("Iterations:", N_ITERATIONS)

print("\nDijkstra:")
print("Route:", dijkstra_path)
print(f"Cost: {dijkstra_cost:.8f} minutes")
print(f"Distance: {dijkstra_distance:.2f} m")
print(f"Runtime: {dijkstra_time * 1000:.4f} ms")

print("\nPSO:")
print("Route:", global_best_path)
print("Segments:", global_best_segments)
print(f"Cost: {global_best_cost:.8f} minutes")
print(f"Distance: {global_best_distance:.2f} m")
print(f"Runtime: {pso_time * 1000:.4f} ms")
print(f"Optimality gap vs Dijkstra: {gap_percent:.4f}%")

# Save PSO route
route_rows = []

for u, v, segment_id in zip(
    global_best_path[:-1],
    global_best_path[1:],
    global_best_segments,
):
    edge = G[u][v]

    route_rows.append({
        "from_node": u,
        "to_node": v,
        "segment_id": segment_id,
        "distance_m": edge["distance"],
        "cost_minutes": edge["base_cost"],
    })

pd.DataFrame(route_rows).to_csv(
    OUTPUT_DIR / "pso_best_route.csv",
    index=False,
)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "pso_convergence.csv",
    index=False,
)

print("\nSaved:", OUTPUT_DIR / "pso_best_route.csv")
print("Saved:", OUTPUT_DIR / "pso_convergence.csv")
