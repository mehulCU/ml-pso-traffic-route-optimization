
import pandas as pd
import networkx as nx

print("-----------------------------------")
print("FINDING A ROUTING REGION")
print("-----------------------------------")

edges = pd.read_csv("ml/routing_graph_edges.csv")
nodes = pd.read_csv("ml/full_graph_nodes.csv")

# Build the full graph, retaining road-segment attributes.
G = nx.Graph()

for row in edges.itertuples(index=False):
    G.add_edge(
        int(row.start_node),
        int(row.end_node),
        segment_id=int(row.segment_id),
        distance=float(row.distance),
        speed_limit=float(row.speed_limit),
        frc=int(row.frc)
    )

# The routing graph's largest component.
largest = max(nx.connected_components(G), key=len)
G = G.subgraph(largest).copy()

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")

# Identify junctions.
junctions = [
    node for node, degree in G.degree()
    if degree >= 3
]

print(f"Junctions: {len(junctions)}")

# Inspect junctions and the number of nearby junctions
# within 1.5 km of network distance.
results = []

for source in junctions:
    distances = nx.single_source_dijkstra_path_length(
        G,
        source,
        cutoff=1500,
        weight="distance"
    )

    nearby = [
        node for node, distance in distances.items()
        if node != source
        and G.degree[node] >= 3
    ]

    results.append({
        "node_id": source,
        "degree": G.degree[source],
        "junctions_within_1500m": len(nearby)
    })

results = pd.DataFrame(results)
results = results.sort_values(
    ["junctions_within_1500m", "degree"],
    ascending=False
)

print("\nTop 20 regional junction candidates:")
print(results.head(20).to_string(index=False))

results.to_csv("ml/routing_region_candidates.csv", index=False)

print("\nSaved: ml/routing_region_candidates.csv")
