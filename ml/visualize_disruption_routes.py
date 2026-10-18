
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_PATH = Path("ml/regional_model/disruption_routes_comparison.png")

SOURCE = 11388
DESTINATION = 14979
SEED = 0
DISRUPTION_FRACTION = 0.15


def build_graph(df):
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        u = int(row.start_node)
        v = int(row.end_node)
        cost = float(row.traffic_aware_cost_minutes)

        if u == v:
            continue

        if not graph.has_edge(u, v) or cost < graph[u][v]["cost"]:
            graph.add_edge(
                u,
                v,
                cost=cost,
                segment_id=str(row.segment_id),
            )

    return graph


def route_cost(graph, path):
    return sum(
        graph[u][v]["cost"]
        for u, v in zip(path[:-1], path[1:])
    )


def route_edges(path):
    return {
        (min(u, v), max(u, v))
        for u, v in zip(path[:-1], path[1:])
    }


def main():
    df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
    base_graph = build_graph(df)

    baseline_route = nx.shortest_path(
        base_graph, SOURCE, DESTINATION, weight="cost"
    )
    baseline_cost = route_cost(base_graph, baseline_route)
    baseline_edges = route_edges(baseline_route)

    # Select one reproducible set of disrupted edges.
    edges = sorted(
        (min(u, v), max(u, v))
        for u, v in base_graph.edges()
    )

    rng = random.Random(SEED)
    rng.shuffle(edges)

    affected_count = round(len(edges) * DISRUPTION_FRACTION)
    affected_edges = set(edges[:affected_count])

    scenarios = [
        ("Baseline", None),
        ("15% disruption — 2× cost", 2.0),
        ("15% disruption — 3× cost", 3.0),
    ]

    scenario_data = []

    for title, multiplier in scenarios:
        graph = base_graph.copy()

        if multiplier is not None:
            for u, v in affected_edges:
                graph[u][v]["cost"] *= multiplier

        route = nx.shortest_path(
            graph, SOURCE, DESTINATION, weight="cost"
        )
        cost = route_cost(graph, route)

        scenario_data.append({
            "title": title,
            "graph": graph,
            "route": route,
            "cost": cost,
            "edges": route_edges(route),
            "multiplier": multiplier,
        })

    # One shared layout keeps node positions consistent across panels.
    pos = nx.spring_layout(base_graph, seed=7, iterations=60)

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    for ax, item in zip(axes, scenario_data):
        graph = item["graph"]
        route = item["route"]
        current_edges = item["edges"]
        multiplier = item["multiplier"]

        # Background road network.
        nx.draw_networkx_edges(
            base_graph, pos, ax=ax,
            edge_color="lightgray", width=0.5, alpha=0.45
        )

        # Disrupted edges, shown only for disruption scenarios.
        if multiplier is not None:
            nx.draw_networkx_edges(
                base_graph, pos, ax=ax,
                edgelist=list(affected_edges),
                edge_color="orange", width=0.8, alpha=0.65
            )

        # Baseline route.
        nx.draw_networkx_edges(
            base_graph, pos, ax=ax,
            edgelist=list(baseline_edges),
            edge_color="royalblue", width=2.0
        )

        # Scenario's optimal route.
        nx.draw_networkx_edges(
            base_graph, pos, ax=ax,
            edgelist=list(current_edges),
            edge_color="crimson", width=2.5
        )

        nx.draw_networkx_nodes(
            base_graph, pos, ax=ax,
            node_size=5, node_color="gray", alpha=0.6
        )

        nx.draw_networkx_nodes(
            base_graph, pos, ax=ax,
            nodelist=[SOURCE], node_size=85,
            node_color="green"
        )
        nx.draw_networkx_nodes(
            base_graph, pos, ax=ax,
            nodelist=[DESTINATION], node_size=85,
            node_color="purple"
        )

        ax.set_title(
            f"{item['title']}\n"
            f"Cost: {item['cost']:.4f} min | "
            f"Edges: {len(route) - 1}",
            fontsize=11
        )
        ax.axis("off")

    fig.suptitle(
        "Dynamic Route Optimization Under Simulated Disruptions",
        fontsize=16
    )

    fig.text(
        0.5, 0.025,
        "Green: source | Purple: destination | "
        "Blue: baseline route | Red: scenario route | "
        "Orange: disrupted edges\n"
        "Note: this is a schematic graph layout, not a geographic map.",
        ha="center", fontsize=10
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print("Visualization saved:", OUTPUT_PATH)
    print(f"Baseline cost: {baseline_cost:.6f} minutes")

    for item in scenario_data[1:]:
        print(
            f"{item['title']}: "
            f"{item['cost']:.6f} minutes, "
            f"{len(item['route']) - 1} edges, "
            f"route changed: "
            f"{item['edges'] != baseline_edges}"
        )


if __name__ == "__main__":
    main()
