
import pandas as pd

graph = pd.read_csv("ml/routing_graph_edges.csv")
selected = pd.read_csv("ml/top_100_segments_train_only.csv")

graph_ids = set(graph["segment_id"].astype(int))
selected_ids = set(selected["segment_id"].astype(int))

overlap = graph_ids & selected_ids

print("-----------------------------------")
print("GRAPH / LSTM ROAD OVERLAP")
print("-----------------------------------")
print(f"Routing graph edges: {len(graph_ids)}")
print(f"LSTM-selected roads: {len(selected_ids)}")
print(f"Matching segment IDs: {len(overlap)}")
print(f"Selected roads absent from graph: {len(selected_ids - graph_ids)}")
print(
    f"Percentage of graph edges with LSTM coverage: "
    f"{100 * len(overlap) / len(graph_ids):.2f}%"
)
