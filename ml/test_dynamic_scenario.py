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


def build_graph(df, cost_column):
    """Build the same undirected graph used by our benchmarks."""
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        u = int(row.start_node)
        v = int(row.end_node)

        if u == v:
            continue

        cost = float(getattr(row, cost_column))
        attrs = {
            "cost": cost,
            "segment_id": str(row.segment_id),
        }

        # Keep the cheapest parallel edge, matching earlier benchmarks.
        if not graph.has_edge(u, v) or cost < graph[u][v]["cost"]:
            graph.add_edge(u, v, **attrs)

    return graph


def path_cost(graph, path):
    return sum(
        graph[u][v]["cost"]
        for u, v in zip(path[:-1], path[1:])
    )


def route_segments(graph, path):
    return [
        graph[u][v]["segment_id"]
        for u, v in zip(path[:-1], path[1:])
    ]


def main():
    df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})

    # Scenario A: original costs.
    base_graph = build_graph(df, "traffic_aware_cost_minutes")

    baseline_route = nx.shortest_path(
        base_graph, SOURCE, DESTINATION, weight="cost"
    )
    baseline_cost = path_cost(base_graph, baseline_route)

    # Select disrupted road segments reproducibly.
    rng = random.Random(RANDOM_SEED)
    segment_ids = df["segment_id"].tolist()
    affected_count = round(len(segment_ids) * AFFECTED_FRACTION)
    affected_segments = set(rng.sample(segment_ids, affected_count))

    # Scenario B: apply the simulated disruption.
    df["scenario_cost"] = df["traffic_aware_cost_minutes"]

    affected_mask = df["segment_id"].isin(affected_segments)
    df.loc[affected_mask, "scenario_cost"] *= COST_MULTIPLIER

    scenario_graph = build_graph(df, "scenario_cost")

    scenario_route = nx.shortest_path(
        scenario_graph, SOURCE, DESTINATION, weight="cost"
    )
    scenario_cost = path_cost(scenario_graph, scenario_route)

    # What if we kept using the old route after costs changed?
    old_route_new_cost = path_cost(scenario_graph, baseline_route)

    penalty = old_route_new_cost - scenario_cost
    penalty_percent = (
        penalty / scenario_cost * 100 if scenario_cost else 0
    )

    baseline_edges = list(zip(baseline_route[:-1], baseline_route[1:]))
    scenario_edges = list(zip(scenario_route[:-1], scenario_route[1:]))
    common_edges = len(set(baseline_edges) & set(scenario_edges))
    union_edges = len(set(baseline_edges) | set(scenario_edges))
    overlap_percent = common_edges / union_edges * 100 if union_edges else 100

    # Save route-level results separately from existing benchmarks.
    route_records = []

    for label, graph, route in [
        ("baseline", base_graph, baseline_route),
        ("changed_scenario", scenario_graph, scenario_route),
    ]:
        for index, (u, v) in enumerate(zip(route[:-1], route[1:]), start=1):
            route_records.append({
                "scenario": label,
                "step": index,
                "from_node": u,
                "to_node": v,
                "segment_id": graph[u][v]["segment_id"],
                "edge_cost_minutes": graph[u][v]["cost"],
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_path = OUTPUT_DIR / "dynamic_scenario_routes.csv"
    pd.DataFrame(route_records).to_csv(route_path, index=False)

    affected_path = OUTPUT_DIR / "dynamic_scenario_affected_segments.csv"
    pd.DataFrame({
        "segment_id": sorted(affected_segments)
    }).to_csv(affected_path, index=False)

    summary = pd.DataFrame([{
        "source": SOURCE,
        "destination": DESTINATION,
        "random_seed": RANDOM_SEED,
        "affected_fraction": AFFECTED_FRACTION,
        "affected_segment_count": len(affected_segments),
        "cost_multiplier": COST_MULTIPLIER,
        "baseline_route_edges": len(baseline_route) - 1,
        "changed_route_edges": len(scenario_route) - 1,
        "baseline_cost_minutes": baseline_cost,
        "changed_route_cost_minutes": scenario_cost,
        "old_route_cost_under_changed_scenario": old_route_new_cost,
        "penalty_for_old_route_minutes": penalty,
        "penalty_for_old_route_percent": penalty_percent,
        "route_edge_overlap_percent": overlap_percent,
        "route_changed": baseline_route != scenario_route,
    }])

    summary_path = OUTPUT_DIR / "dynamic_scenario_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n========== DYNAMIC SCENARIO RESULTS ==========")
    print(f"Route: {SOURCE} -> {DESTINATION}")
    print(f"Total input segments: {len(df)}")
    print(f"Disrupted segments: {len(affected_segments)}")
    print(f"Cost multiplier: {COST_MULTIPLIER}x")
    print(f"\nBaseline route edges: {len(baseline_route) - 1}")
    print(f"Changed route edges: {len(scenario_route) - 1}")
    print(f"Baseline route cost: {baseline_cost:.8f} minutes")
    print(f"Changed-scenario optimal cost: {scenario_cost:.8f} minutes")
    print(f"Old route cost under changed scenario: {old_route_new_cost:.8f} minutes")
    print(f"Penalty for keeping old route: {penalty:.8f} minutes")
    print(f"Penalty for keeping old route: {penalty_percent:.4f}%")
    print(f"Route changed: {baseline_route != scenario_route}")
    print(f"Edge overlap: {overlap_percent:.2f}%")
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved route details: {route_path}")
    print(f"Saved disrupted segments: {affected_path}")


if __name__ == "__main__":
    main()
