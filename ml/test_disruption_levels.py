import random
from pathlib import Path

import networkx as nx
import pandas as pd

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 11388
DESTINATION = 14979
RANDOM_SEED = 42

SCENARIOS = [
    {"name": "5pct_2x", "fraction": 0.05, "multiplier": 2.0},
    {"name": "15pct_2x", "fraction": 0.15, "multiplier": 2.0},
    {"name": "30pct_2x", "fraction": 0.30, "multiplier": 2.0},
    {"name": "15pct_3x", "fraction": 0.15, "multiplier": 3.0},
]


def build_graph(df):
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        u, v = int(row.start_node), int(row.end_node)

        if u == v:
            continue

        cost = float(row.traffic_aware_cost_minutes)
        attrs = {
            "cost": cost,
            "segment_id": str(row.segment_id),
        }

        if not graph.has_edge(u, v) or cost < graph[u][v]["cost"]:
            graph.add_edge(u, v, **attrs)

    return graph


def route_cost(graph, path):
    return sum(
        graph[u][v]["cost"]
        for u, v in zip(path[:-1], path[1:])
    )


def main():
    df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
    base_graph = build_graph(df)

    # Compute the original optimal route once.
    baseline_route = nx.shortest_path(
        base_graph, SOURCE, DESTINATION, weight="cost"
    )
    baseline_cost = route_cost(base_graph, baseline_route)

    edge_list = sorted(
        (min(u, v), max(u, v))
        for u, v in base_graph.edges()
    )

    records = []
    route_records = []

    for scenario in SCENARIOS:
        fraction = scenario["fraction"]
        multiplier = scenario["multiplier"]

        # Identical seed means each fraction uses a reproducible prefix
        # of the same randomized edge ordering.
        rng = random.Random(RANDOM_SEED)
        shuffled_edges = edge_list.copy()
        rng.shuffle(shuffled_edges)

        affected_count = round(len(edge_list) * fraction)
        affected_edges = set(shuffled_edges[:affected_count])

        changed_graph = base_graph.copy()

        for u, v in affected_edges:
            changed_graph[u][v]["cost"] *= multiplier

        new_route = nx.shortest_path(
            changed_graph, SOURCE, DESTINATION, weight="cost"
        )

        new_cost = route_cost(changed_graph, new_route)
        old_route_cost = route_cost(changed_graph, baseline_route)

        penalty = old_route_cost - new_cost
        penalty_percent = penalty / new_cost * 100 if new_cost else 0

        baseline_edges = {
            (min(u, v), max(u, v))
            for u, v in zip(baseline_route[:-1], baseline_route[1:])
        }
        new_edges = {
            (min(u, v), max(u, v))
            for u, v in zip(new_route[:-1], new_route[1:])
        }

        overlap = (
            len(baseline_edges & new_edges)
            / len(baseline_edges | new_edges) * 100
            if baseline_edges | new_edges else 100
        )

        affected_on_baseline = len(baseline_edges & affected_edges)
        affected_on_new_route = len(new_edges & affected_edges)

        records.append({
            "scenario": scenario["name"],
            "affected_fraction": fraction,
            "cost_multiplier": multiplier,
            "graph_edges": len(edge_list),
            "affected_edges": affected_count,
            "baseline_cost_minutes": baseline_cost,
            "new_optimal_cost_minutes": new_cost,
            "old_route_cost_under_changes": old_route_cost,
            "old_route_penalty_minutes": penalty,
            "old_route_penalty_percent": penalty_percent,
            "baseline_route_edges": len(baseline_edges),
            "new_route_edges": len(new_edges),
            "affected_edges_on_baseline": affected_on_baseline,
            "affected_edges_on_new_route": affected_on_new_route,
            "edge_overlap_percent": overlap,
            "route_changed": baseline_route != new_route,
        })

        for label, graph, path in [
            ("baseline", base_graph, baseline_route),
            (scenario["name"], changed_graph, new_route),
        ]:
            for step, (u, v) in enumerate(
                zip(path[:-1], path[1:]), start=1
            ):
                route_records.append({
                    "scenario": label,
                    "step": step,
                    "from_node": u,
                    "to_node": v,
                    "segment_id": graph[u][v]["segment_id"],
                    "edge_cost_minutes": graph[u][v]["cost"],
                    "edge_disrupted": (
                        min(u, v), max(u, v)
                    ) in affected_edges if label != "baseline" else False,
                })

        print(
            f"{scenario['name']}: "
            f"affected={affected_count}/{len(edge_list)}, "
            f"new_cost={new_cost:.6f}, "
            f"old_route_penalty={penalty:.6f} "
            f"({penalty_percent:.3f}%), "
            f"route_changed={baseline_route != new_route}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "disruption_levels_summary.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False)

    routes_path = OUTPUT_DIR / "disruption_levels_routes.csv"
    pd.DataFrame(route_records).to_csv(routes_path, index=False)

    print("\n========== SUMMARY ==========")
    print(pd.DataFrame(records)[[
        "scenario",
        "affected_edges",
        "new_optimal_cost_minutes",
        "old_route_penalty_percent",
        "edge_overlap_percent",
        "route_changed",
    ]].round(4).to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved routes: {routes_path}")


if __name__ == "__main__":
    main()
