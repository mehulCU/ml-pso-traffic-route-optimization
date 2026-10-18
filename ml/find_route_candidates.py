import pandas as pd
import networkx as nx


print("-----------------------------------")
print("FINDING ROUTE CANDIDATES")
print("-----------------------------------")


# --------------------------------------------------
# 1. Load cleaned routing graph
# --------------------------------------------------

edges = pd.read_csv("ml/routing_graph_edges.csv")

print(f"Routing edges: {len(edges)}")


# --------------------------------------------------
# 2. Build graph
# --------------------------------------------------

G = nx.Graph()

for _, row in edges.iterrows():

    start = int(row["start_node"])
    end = int(row["end_node"])
    distance = float(row["distance"])

    if start == end:
        continue

    if G.has_edge(start, end):

        # Keep shortest physical segment
        if distance < G[start][end]["distance"]:
            G[start][end]["distance"] = distance
            G[start][end]["segment_id"] = int(row["segment_id"])

    else:

        G.add_edge(
            start,
            end,
            distance=distance,
            segment_id=int(row["segment_id"])
        )


# --------------------------------------------------
# 3. Largest component
# --------------------------------------------------

largest_component = max(
    nx.connected_components(G),
    key=len
)

G = G.subgraph(largest_component).copy()

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")


# --------------------------------------------------
# 4. Find junctions
# --------------------------------------------------

junctions = [
    node
    for node, degree in G.degree()
    if degree >= 3
]

print(f"Junction nodes: {len(junctions)}")


# --------------------------------------------------
# 5. Search from junctions
# --------------------------------------------------

print()
print("-----------------------------------")
print("SEARCHING FOR ALTERNATIVE ROUTES")
print("-----------------------------------")


found = 0

# Only inspect first 100 junctions initially.
for source in junctions[:100]:

    # Find nearby nodes within 5 km.
    distances = nx.single_source_dijkstra_path_length(
        G,
        source,
        cutoff=5000,
        weight="distance"
    )

    # Candidate destinations must also be junctions.
    destinations = [
        node
        for node in distances
        if node != source
        and G.degree[node] >= 3
        and distances[node] >= 1000
    ]

    # Test at most 20 destinations per junction.
    destinations = destinations[:20]

    for destination in destinations:

        try:

            paths = nx.shortest_simple_paths(
                G,
                source,
                destination,
                weight="distance"
            )

            route_paths = []

            for path in paths:

                route_paths.append(path)

                if len(route_paths) == 3:
                    break

        except nx.NetworkXNoPath:
            continue


        if len(route_paths) < 2:
            continue


        # --------------------------------------------------
        # Calculate route lengths
        # --------------------------------------------------

        lengths = []

        for path in route_paths:

            length = 0

            for u, v in zip(path[:-1], path[1:]):
                length += G[u][v]["distance"]

            lengths.append(length)


        shortest = lengths[0]
        second = lengths[1]

        difference = (
            (second - shortest) / shortest
        )


        # Accept anything with a meaningful alternative.
        if difference < 0.02:
            continue


        print()
        print(f"Candidate {found + 1}")
        print("-----------------------------------")

        print(f"Source:      {source}")
        print(f"Destination: {destination}")

        print(
            f"Source degree: "
            f"{G.degree[source]}"
        )

        print(
            f"Destination degree: "
            f"{G.degree[destination]}"
        )

        print()



        for i, path in enumerate(route_paths):

            print(
                f"\nRoute {i + 1}: "
                f"{lengths[i]:.2f} m "
                f"({len(path)} nodes)"
            )

            print("Segment IDs:")

            for u, v in zip(path[:-1], path[1:]):
                edge_data = G[u][v]

                print(
                    f"  {u} -> {v} | "
                    f"segment_id: {edge_data['segment_id']} | "
                    f"distance: {edge_data['distance']:.2f} m"
                )



        found += 1

        if found >= 1:
            break

    if found >= 1:
        break


print()
print("-----------------------------------")
print(f"Candidates found: {found}")
print("-----------------------------------")
