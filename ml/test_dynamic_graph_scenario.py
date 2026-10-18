import random
from pathlib import Path

import networkx as nx
import pandas as pd

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 11388
DESTINATION = 14979
RANDOM_SEED = 42
AFFECTED_FRACTION = 0.15
COST_MULTIPLIER = 2.0


def build_graph(df):
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        u = int(row.start_node)
        v = int(row.end_node)

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
    graph = build_graph(df)

    # Select edges from the graph actually used for routing.
    edge_list = sorted(
        (min(u, v), max(u, v))
        for u, v in graph.edges()
    )

    rng = random.Random(RANDOM_SEED)
    affected_count = round(len(edge_list) * AFFECTED_FRACTION)
    affected_edges = set(rng.sample(edge_list, affected_count))

    # Baseline scenario.
    baseline_route = nx.shortest_path(
        graph, SOURCE, DESTINATION, weight="cost"
    )
    baseline_cost = route_cost(graph, baseline_route)

    # Changed scenario: increase the costs of selected retained edges.
    changed_graph = graph.copy()

    for u, v in affected_edges:
        changed_graph[u][v]["cost"] *= COST_MULTIPLIER

    changed_route = nx.shortest_path(
        changed_graph, SOURCE, DESTINATION, weight="cost"
    )
    changed_cost = route_cost(changed_graph, changed_route)
    old_route_changed_cost = route_cost(changed_graph, baseline_route)

    penalty = old_route_changed_cost - changed_cost
    penalty_percent = penalty / changed_cost * 100 if changed_cost else 0

    baseline_edges = {
        (min(u, v), max(u, v))
        for u, v in zip(baseline_route[:-1], baseline_route[1:])
    }
    changed_edges = {
        (min(u, v), max(u, v))
        for u, v in zip(changed_route[:-1], changed_route[1:])
    }

    overlap = (
        len(baseline_edges & changed_edges)
        / len(baseline_edges | changed_edges) * 100
        if baseline_edges | changed_edges else 100
    )

    affected_baseline = len(baseline_edges & affected_edges)
    affected_changed = len(changed_edges & affected_edges)

    summary = pd.DataFrame([{
        "source": SOURCE,
        "destination": DESTINATION,
        "seed": RANDOM_SEED,
        "graph_edges": len(edge_list),
        "affected_edges": affected_count,
        "cost_multiplier": COST_MULTIPLIER,
        "baseline_route_edges": len(baseline_edges),
        "changed_route_edges": len(changed_edges),
        "baseline_cost_minutes": baseline_cost,
        "changed_optimal_cost_minutes": changed_cost,
        "old_route_changed_cost_minutes": old_route_changed_cost,
        "old_route_penalty_minutes": penalty,
        "old_route_penalty_percent": penalty_percent,
        "affected_edges_on_baseline": affected_baseline,
        "affected_edges_on_changed_route": affected_changed,
        "edge_overlap_percent": overlap,
        "route_changed": baseline_route != changed_route,
    }])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "dynamic_graph_scenario_summary.csv"
    summary.to_csv(summary_path, index=False)

    affected_df = pd.DataFrame([
        {
            "start_node": u,
            "end_node": v,
            "segment_id": graph[u][v]["segment_id"],
            "original_cost": graph[u][v]["cost"],
            "changed_cost": changed_graph[u][v]["cost"],
        }
        for u, v in sorted(affected_edges)
    ])
    affected_path = OUTPUT_DIR / "dynamic_graph_scenario_affected_edges.csv"
    affected_df.to_csv(affected_path, index=False)

    route_records = []
    for label, route_graph, route in [
        ("baseline", graph, baseline_route),
        ("changed_scenario", changed_graph, changed_route),
    ]:
        for step, (u, v) in enumerate(zip(route[:-1], route[1:]), start=1):
            route_records.append({
                "scenario": label,
                "step": step,
                "from_node": u,
                "to_node": v,
                "segment_id": route_graph[u][v]["segment_id"],
                "edge_cost_minutes": route_graph[u][v]["cost"],
                "edge_disrupted": (min(u, v), max(u, v)) in affected_edges,
            })

    route_path = OUTPUT_DIR / "dynamic_graph_scenario_routes.csv"
    pd.DataFrame(route_records).to_csv(route_path, index=False)

    print("\n========== GRAPH-LEVEL DYNAMIC SCENARIO ==========")
    print(f"Route: {SOURCE} -> {DESTINATION}")
    print(f"Graph edges: {len(edge_list)}")
    print(f"Affected graph edges: {affected_count}")
    print(f"Cost multiplier: {COST_MULTIPLIER}x")
    print(f"\nBaseline route edges: {len(baseline_edges)}")
    print(f"Changed route edges: {len(changed_edges)}")
    print(f"Baseline cost: {baseline_cost:.8f} minutes")
    print(f"Changed optimal cost: {changed_cost:.8f} minutes")
    print(f"Old route cost under changes: {old_route_changed_cost:.8f} minutes")
    print(f"Penalty for old route: {penalty:.8f} minutes ({penalty_percent:.4f}%)")
    print(f"Affected edges on baseline route: {affected_baseline}")
    print(f"Affected edges on changed route: {affected_changed}")
    print(f"Edge overlap: {overlap:.2f}%")
    print(f"Route changed: {baseline_route != changed_route}")
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved affected edges: {affected_path}")
    print(f"Saved routes: {route_path}")


if __name__ == "__main__":
    main()
