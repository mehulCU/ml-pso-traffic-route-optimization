
import pandas as pd
import networkx as nx

CENTER_NODE = 12885
RADIUS_METERS = 1500

edges = pd.read_csv("ml/routing_graph_edges.csv")
nodes = pd.read_csv("ml/full_graph_nodes.csv")
traffic = pd.read_csv("traffic_all.csv", usecols=["segment_id", "date", "hour"])

G = nx.Graph()

for row in edges.itertuples(index=False):
    G.add_edge(
        int(row.start_node),
        int(row.end_node),
        segment_id=int(row.segment_id),
        distance=float(row.distance)
    )

# Roads within 1.5 km of the center, measured along the graph.
distances = nx.single_source_dijkstra_path_length(
    G,
    CENTER_NODE,
    cutoff=RADIUS_METERS,
    weight="distance"
)

region_nodes = set(distances)

region_edges = edges[
    edges["start_node"].isin(region_nodes)
    & edges["end_node"].isin(region_nodes)
].copy()

# Keep only roads fully inside this connected region.
region_ids = set(region_edges["segment_id"].astype(int))

# Check hourly history coverage.
traffic["date"] = pd.to_datetime(traffic["date"])
history = (
    traffic[traffic["segment_id"].isin(region_ids)]
    .groupby("segment_id")
    .agg(
        observations=("hour", "size"),
        unique_hours=("hour", "nunique"),
        unique_dates=("date", "nunique")
    )
)

region_edges = region_edges.merge(
    history,
    left_on="segment_id",
    right_index=True,
    how="left"
)

region_edges["observations"] = (
    region_edges["observations"].fillna(0).astype(int)
)

complete = region_edges[region_edges["observations"] == 480]

print("-----------------------------------")
print("CANDIDATE REGION REPORT")
print("-----------------------------------")
print(f"Center node: {CENTER_NODE}")
print(f"Nodes within 1.5 km network distance: {len(region_nodes)}")
print(f"Roads fully inside region: {len(region_edges)}")
print(f"Roads with 480 observations: {len(complete)}")

if len(region_edges):
    print(
        f"Complete-history percentage: "
        f"{100 * len(complete) / len(region_edges):.2f}%"
    )

# Save the region for thex next step.
region_edges.to_csv("ml/candidate_region_edges.csv", index=False)

print("\nSaved: ml/candidate_region_edges.csv")
