import pandas as pd


nodes = pd.read_csv(
    "ml/full_graph_nodes.csv"
)

degrees = pd.read_csv(
    "ml/full_graph_node_degrees.csv"
)


print("-----------------------------------")
print("GRAPH JUNCTION ANALYSIS")
print("-----------------------------------")


# -----------------------------------
# Degree distribution
# -----------------------------------

print("\nDegree distribution:")

print(
    degrees["degree"]
    .value_counts()
    .sort_index()
    .to_string()
)


# -----------------------------------
# High-degree junctions
# -----------------------------------

junctions = degrees[
    degrees["degree"] >= 3
].copy()

junctions = junctions.merge(
    nodes,
    on="node_id",
    how="left"
)


junctions = junctions.sort_values(
    "degree",
    ascending=False
)


print("\nNumber of junction nodes:")
print(len(junctions))


print("\nTop 30 junctions:")

print(
    junctions[
        [
            "node_id",
            "degree",
            "longitude",
            "latitude"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# -----------------------------------
# Save junctions
# -----------------------------------

junctions.to_csv(
    "ml/full_graph_junctions.csv",
    index=False
)


print("\nSaved:")
print("ml/full_graph_junctions.csv")