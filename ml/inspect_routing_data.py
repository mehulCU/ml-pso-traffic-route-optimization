
import pandas as pd

edges = pd.read_csv("ml/routing_graph_edges.csv")
nodes = pd.read_csv("ml/full_graph_nodes.csv")
degrees = pd.read_csv("ml/full_graph_node_degrees.csv")

print("-----------------------------------")
print("ROUTING DATA INSPECTION")
print("-----------------------------------")

print("\nEdges:")
print(edges.columns.tolist())
print(edges.head())

print("\nNodes:")
print(nodes.columns.tolist())
print(nodes.head())

print("\nNode degrees:")
print(degrees.columns.tolist())
print(degrees.head())

print("\nEdge count:", len(edges))
print("Unique segment IDs:", edges["segment_id"].nunique())
print("Unique start nodes:", edges["start_node"].nunique())
print("Unique end nodes:", edges["end_node"].nunique())
