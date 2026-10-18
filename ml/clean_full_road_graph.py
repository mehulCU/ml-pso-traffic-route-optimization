import pandas as pd


print("-----------------------------------")
print("CLEANING FULL ROAD GRAPH")
print("-----------------------------------")


edges = pd.read_csv(
    "ml/full_graph_edges.csv"
)

nodes = pd.read_csv(
    "ml/full_graph_nodes.csv"
)


print("Original edges:", len(edges))


# -----------------------------------
# Remove self-loops
# -----------------------------------

self_loops = edges[
    edges["start_node"] == edges["end_node"]
].copy()


clean_edges = edges[
    edges["start_node"] != edges["end_node"]
].copy()


print(
    "Self-loop edges removed:",
    len(self_loops)
)


if len(self_loops) > 0:

    print("\nRemoved self-loops:")

    print(
        self_loops[
            [
                "segment_id",
                "start_node",
                "end_node",
                "street_name",
                "distance"
            ]
        ].to_string(index=False)
    )


# -----------------------------------
# Recalculate node degrees
# -----------------------------------

degree_start = (
    clean_edges
    .groupby("start_node")
    .size()
)

degree_end = (
    clean_edges
    .groupby("end_node")
    .size()
)


degrees = {}

for node_id in nodes["node_id"]:

    degrees[node_id] = (
        degree_start.get(node_id, 0)
        +
        degree_end.get(node_id, 0)
    )


degree_df = pd.DataFrame({
    "node_id": list(degrees.keys()),
    "degree": list(degrees.values())
})


# -----------------------------------
# Save cleaned graph
# -----------------------------------

clean_edges.to_csv(
    "ml/routing_graph_edges.csv",
    index=False
)

degree_df.to_csv(
    "ml/routing_graph_node_degrees.csv",
    index=False
)


print("\n-----------------------------------")
print("CLEAN GRAPH RESULTS")
print("-----------------------------------")

print(
    "Original edges:",
    len(edges)
)

print(
    "Clean edges:",
    len(clean_edges)
)

print(
    "Self-loops:",
    len(self_loops)
)

print("\nDegree distribution:")

print(
    degree_df["degree"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nSaved:")

print("ml/routing_graph_edges.csv")
print("ml/routing_graph_node_degrees.csv")
