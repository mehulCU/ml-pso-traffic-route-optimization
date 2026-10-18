
import pandas as pd
import networkx as nx

PATH = "ml/regional_model/regional_routing_edges.csv"

df = pd.read_csv(PATH)

# Simple graph for connectivity and candidate route inspection.
# Keep the lowest-cost parallel segment for this initial search.
G = nx.Graph()

for row in df.itertuples(index=False):
    u, v = int(row.start_node), int(row.end_node)
    cost = float(row.traffic_aware_cost_minutes)

    if G.has_edge(u, v):
        if cost < G[u][v]["weight"]:
            G[u][v].update(
                weight=cost,
                segment_id=str(row.segment_id),
            )
    else:
        G.add_edge(
            u,
            v,
            weight=cost,
            segment_id=str(row.segment_id),
        )

print("Nodes:", G.number_of_nodes())
print("Edges (after parallel-edge reduction):", G.number_of_edges())

# Prefer junctions as origins and destinations.
junctions = [
    node for node, degree in G.degree()
    if degree >= 3
]

print("Junction candidates:", len(junctions))

# Find pairs with multiple loop-free alternatives.
found = False

for source in junctions:
    lengths = nx.single_source_dijkstra_path_length(
        G, source, weight="weight"
    )

    # Consider reachable junctions with a reasonable path cost.
    candidates = [
        node for node in junctions
        if node != source
        and node in lengths
        and lengths[node] >= 0.10
        and lengths[node] <= 0.50
    ]

    for destination in candidates:
        paths = nx.shortest_simple_paths(
            G, source, destination, weight="weight"
        )

        first_paths = []
        try:
            for path in paths:
                first_paths.append(path)
                if len(first_paths) == 3:
                    break
        except nx.NetworkXNoPath:
            continue

        if len(first_paths) >= 2:
            costs = [
                sum(G[a][b]["weight"] for a, b in zip(path, path[1:]))
                for path in first_paths
            ]

            # Ensure an alternative is meaningfully different in cost.
            if costs[1] >= costs[0] * 1.02:
                print("\nCandidate source:", source)
                print("Candidate destination:", destination)

                for i, (path, cost) in enumerate(
                    zip(first_paths, costs), start=1
                ):
                    print(
                        f"Route {i}: nodes={len(path)}, "
                        f"cost={cost:.4f} minutes"
                    )

                found = True
                break

    if found:
        break

if not found:
    print("No suitable pair found with the current thresholds.")
