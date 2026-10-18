
import random
from pathlib import Path

import networkx as nx
import pandas as pd

DATA_PATH = "ml/regional_model/regional_routing_edges.csv"
OUTPUT_DIR = Path("ml/regional_model")

SOURCE = 11388
DESTINATION = 14979

N_SEEDS = 10

SCENARIOS = [
    {"name": "5pct_2x", "fraction": 0.05, "multiplier": 2.0},
    {"name": "15pct_2x", "fraction": 0.15, "multiplier": 2.0},
    {"name": "30pct_2x", "fraction": 0.30, "multiplier": 2.0},
    {"name": "15pct_3x", "fraction": 0.15, "multiplier": 3.0},
]


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


def main():
    df = pd.read_csv(DATA_PATH, dtype={"segment_id": str})
    base_graph = build_graph(df)

    baseline_route = nx.shortest_path(
        base_graph, SOURCE, DESTINATION, weight="cost"
    )
    baseline_cost = route_cost(base_graph, baseline_route)

    edges = sorted(
        (min(u, v), max(u, v))
        for u, v in base_graph.edges()
    )

    records = []

    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            rng = random.Random(seed)
            shuffled = edges.copy()
            rng.shuffle(shuffled)

            affected_count = round(
                len(edges) * scenario["fraction"]
            )
            affected = set(shuffled[:affected_count])

            changed_graph = base_graph.copy()

            for u, v in affected:
                changed_graph[u][v]["cost"] *= (
                    scenario["multiplier"]
                )

            new_route = nx.shortest_path(
                changed_graph,
                SOURCE,
                DESTINATION,
                weight="cost",
            )

            new_cost = route_cost(changed_graph, new_route)
            old_route_cost = route_cost(
                changed_graph, baseline_route
            )

            penalty = old_route_cost - new_cost

            baseline_edges = {
                (min(u, v), max(u, v))
                for u, v in zip(
                    baseline_route[:-1], baseline_route[1:]
                )
            }
            new_edges = {
                (min(u, v), max(u, v))
                for u, v in zip(
                    new_route[:-1], new_route[1:]
                )
            }

            union = baseline_edges | new_edges
            overlap = (
                len(baseline_edges & new_edges) / len(union) * 100
                if union else 100.0
            )

            records.append({
                "scenario": scenario["name"],
                "seed": seed,
                "affected_edges": affected_count,
                "multiplier": scenario["multiplier"],
                "baseline_cost_minutes": baseline_cost,
                "new_optimal_cost_minutes": new_cost,
                "old_route_penalty_minutes": penalty,
                "old_route_penalty_percent": (
                    penalty / new_cost * 100 if new_cost else 0
                ),
                "edge_overlap_percent": overlap,
                "route_changed": baseline_edges != new_edges,
            })

        print(f"Completed: {scenario['name']}")

    results = pd.DataFrame(records)

    results.to_csv(
        OUTPUT_DIR / "disruption_robustness_trials.csv",
        index=False,
    )

    summary = results.groupby("scenario").agg(
        runs=("seed", "count"),
        mean_optimal_cost=("new_optimal_cost_minutes", "mean"),
        std_optimal_cost=("new_optimal_cost_minutes", "std"),
        mean_old_route_penalty_pct=(
            "old_route_penalty_percent", "mean"
        ),
        std_old_route_penalty_pct=(
            "old_route_penalty_percent", "std"
        ),
        mean_edge_overlap_pct=("edge_overlap_percent", "mean"),
        route_change_rate=("route_changed", "mean"),
    ).reset_index()

    summary["route_change_rate"] *= 100

    summary.to_csv(
        OUTPUT_DIR / "disruption_robustness_summary.csv",
        index=False,
    )

    print("\n========== ROBUSTNESS SUMMARY ==========")
    print(summary.round(4).to_string(index=False))
    print("\nSaved trials: ml/regional_model/disruption_robustness_trials.csv")
    print("Saved summary: ml/regional_model/disruption_robustness_summary.csv")


if __name__ == "__main__":
    main()
