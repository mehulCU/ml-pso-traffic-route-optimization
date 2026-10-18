
import pandas as pd
import networkx as nx
from time import perf_counter
from pathlib import Path

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_PATH = Path("ml/regional_model/dijkstra_baseline_route.csv")

SOURCE = 316
DESTINATION = 336

df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})

# Use the same parallel-edge reduction as the candidate search:
# retain the least-cost segment between each pair of nodes.
G = nx.Graph()

for row in df.itertuples(index=False):
    u = int(row.start_node)
    v = int(row.end_node)
    cost = float(row.traffic_aware_cost_minutes)

    if G.has_edge(u, v):
        if cost < G[u][v]["weight"]:
            G[u][v].update(
                weight=cost,
                segment_id=str(row.segment_id),
            )
    else:
        G.add_edge(
            u, v,
            weight=cost,
            segment_id=str(row.segment_id),
        )

# Run Dijkstra and measure execution time
start = perf_counter()
path = nx.dijkstra_path(
    G, SOURCE, DESTINATION, weight="weight"
)
elapsed = perf_counter() - start

# Extract each road segment along the route
route_rows = []
total_cost = 0.0
total_distance = 0.0

for u, v in zip(path, path[1:]):
    edge = G[u][v]
    segment_id = edge["segment_id"]

    # Retrieve distance for the selected segment
    match = df[df["segment_id"] == segment_id].iloc[0]
    cost = float(edge["weight"])
    distance = float(match["distance"])

    total_cost += cost
    total_distance += distance

    route_rows.append({
        "from_node": u,
        "to_node": v,
        "segment_id": segment_id,
        "distance_m": distance,
        "cost_minutes": cost,
    })

route_df = pd.DataFrame(route_rows)
route_df.to_csv(OUTPUT_PATH, index=False)

print("========== DIJKSTRA BASELINE ==========")
print("Source:", SOURCE)
print("Destination:", DESTINATION)
print("Route nodes:", path)
print("Road segments:", len(route_rows))
print(f"Total distance: {total_distance:.2f} m")
print(f"Total routing cost: {total_cost:.6f} minutes")
print(f"Execution time: {elapsed * 1000:.4f} ms")
print("\nSegment-by-segment route:")
print(route_df.to_string(index=False))
print("\nSaved:", OUTPUT_PATH)
