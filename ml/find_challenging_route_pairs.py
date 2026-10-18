
import pandas as pd
import networkx as nx

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"

MIN_EDGES = 8
MAX_PAIRS_TO_PRINT = 20
MAX_ALTERNATIVES = 3

df = pd.read_csv(DATA_PATH)
G = nx.Graph()

# Keep the cheapest parallel edge, matching our existing benchmarks.
for row in df.itertuples(index=False):
    u = int(row.start_node)
    v = int(row.end_node)

    if u == v:
        continue

    cost = float(row.traffic_aware_cost_minutes)

    if not G.has_edge(u, v) or cost < G[u][v]["cost"]:
        G.add_edge(u, v, cost=cost)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected components:", nx.number_connected_components(G))

candidates = []
seen_pairs = set()

# Use junctions to make the search more likely to find route choices.
junctions = sorted(n for n, degree in G.degree() if degree >= 3)

for source in junctions:
    lengths, paths = nx.single_source_dijkstra(
        G, source, weight="cost"
    )

    for destination in junctions:
        if destination == source or destination not in paths:
            continue

        pair = (source, destination)
        unordered_pair = tuple(sorted(pair))

        if unordered_pair in seen_pairs:
            continue

        seen_pairs.add(unordered_pair)
        shortest_path = paths[destination]

        if len(shortest_path) - 1 < MIN_EDGES:
            continue

        shortest_cost = lengths[destination]

        # Require multiple meaningfully competitive alternatives.
        try:
            alternatives = []
            generator = nx.shortest_simple_paths(
                G, source, destination, weight="cost"
            )

            for path in generator:
                alternatives.append(path)
                if len(alternatives) == MAX_ALTERNATIVES:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if len(alternatives) < 2:
            continue

        alternative_costs = [
            sum(
                G[u][v]["cost"]
                for u, v in zip(path[:-1], path[1:])
            )
            for path in alternatives
        ]

        second_cost = alternative_costs[1]
        gap = (second_cost / shortest_cost - 1) * 100

        candidates.append({
            "source": source,
            "destination": destination,
            "edges_in_shortest_path": len(shortest_path) - 1,
            "shortest_cost_minutes": shortest_cost,
            "second_route_cost_minutes": second_cost,
            "second_route_gap_percent": gap,
            "routes_examined": len(alternatives),
        })

# Prefer longer routes with distinct alternatives.
candidates.sort(
    key=lambda x: (
        x["edges_in_shortest_path"],
        x["second_route_gap_percent"],
    ),
    reverse=True,
)

result = pd.DataFrame(candidates)

if result.empty:
    print("\nNo suitable pairs found. Try lowering MIN_EDGES.")
else:
    result = result.head(MAX_PAIRS_TO_PRINT)
    print("\nChallenging candidate pairs:")
    print(result.to_string(index=False))

    output = "ml/regional_model/challenging_route_pairs.csv"
    result.to_csv(output, index=False)
    print("\nSaved:", output)
