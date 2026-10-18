import pandas as pd

TARGET_NODE = 4479

edges = pd.read_csv(
    "ml/full_graph_edges.csv"
)

nodes = pd.read_csv(
    "ml/full_graph_nodes.csv"
)

print("-----------------------------------")
print("JUNCTION EDGE ANALYSIS")
print("-----------------------------------")

target_edges = edges[
    (edges["start_node"] == TARGET_NODE)
    |
    (edges["end_node"] == TARGET_NODE)
].copy()

print(
    f"Node {TARGET_NODE} has "
    f"{len(target_edges)} connected roads."
)

print("\nConnected roads:")

print(
    target_edges[
        [
            "segment_id",
            "start_node",
            "end_node",
            "street_name",
            "distance",
            "speed_limit",
            "frc"
        ]
    ].to_string(index=False)
)


print("\nJunction coordinates:")

target_node = nodes[
    nodes["node_id"] == TARGET_NODE
]

print(
    target_node.to_string(index=False)
)


print("\nNeighboring nodes:")

neighbor_ids = set(
    target_edges["start_node"]
).union(
    set(target_edges["end_node"])
)

neighbor_ids.discard(TARGET_NODE)

neighbors = nodes[
    nodes["node_id"].isin(neighbor_ids)
]

print(
    neighbors.to_string(index=False)
)
