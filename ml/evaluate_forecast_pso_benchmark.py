import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ml" / "evaluate_forecast_discrete_pso.py"
OUTPUT = ROOT / "ml" / "regional_model"
RESULTS = OUTPUT / "forecast_pso_benchmark_results.csv"

TIMESTAMP = "2024-08-26T10:00:00"
PAIRS = [
    (3319, 316),
    (11388, 14979),
    (11703, 6988),
    (2272, 2753),
    (2681, 15490),
]
SEEDS = [42, 123, 2026]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for pair_index, ((source, destination), seed) in enumerate(
        itertools.product(PAIRS, SEEDS), start=1
    ):
        print(
            f"\n[{pair_index}/{len(PAIRS) * len(SEEDS)}] "
            f"{source} -> {destination}, seed={seed}",
            flush=True,
        )

        command = [
            sys.executable,
            str(SCRIPT),
            "--source", str(source),
            "--destination", str(destination),
            "--timestamp", TIMESTAMP,
            "--seed", str(seed),
        ]

        run = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if run.returncode != 0:
            print(run.stdout)
            print(run.stderr, file=sys.stderr)
            rows.append({
                "source": source,
                "destination": destination,
                "timestamp": TIMESTAMP,
                "seed": seed,
                "status": "error",
                "error": (run.stderr or run.stdout)[-2000:],
            })
            continue

        # Each run writes a single-run summary; read it before the next run
        # overwrites it.
        summary_path = OUTPUT / "forecast_discrete_pso_summary.csv"
        result = pd.read_csv(summary_path).iloc[0].to_dict()
        result.update({
            "source": source,
            "destination": destination,
            "timestamp": TIMESTAMP,
            "seed": seed,
            "status": "success",
            "error": "",
        })
        rows.append(result)

        print(
            f"Gap={result['optimality_gap_percent']:.6f}%, "
            f"valid={result['valid_pso_route']}, "
            f"Dijkstra={result['dijkstra_runtime_ms']:.3f} ms, "
            f"PSO={result['pso_runtime_ms']:.3f} ms",
            flush=True,
        )

    new_results = pd.DataFrame(rows)

    # Keep earlier benchmark rows, replacing only duplicate pair/seed/time runs.
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        keys = ["source", "destination", "timestamp", "seed"]
        if all(k in old.columns for k in keys):
            old = old.merge(
                new_results[keys].drop_duplicates(),
                on=keys,
                how="left",
                indicator=True,
            )
            old = old[old["_merge"] == "left_only"].drop(
                columns=["_merge"]
            )
            new_results = pd.concat(
                [old, new_results], ignore_index=True, sort=False
            )

    new_results.to_csv(RESULTS, index=False)

    successful = new_results[new_results["status"] == "success"].copy()
    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Rows saved: {len(new_results)}")
    print(f"Successful runs in this batch: {sum(r['status'] == 'success' for r in rows)}")
    print(f"Failed runs in this batch: {sum(r['status'] != 'success' for r in rows)}")

    if not successful.empty:
        successful["optimal"] = successful["optimality_gap_percent"] <= 1e-7
        print(
            "Optimal-cost runs: "
            f"{int(successful['optimal'].sum())}/{len(successful)} "
            f"({successful['optimal'].mean() * 100:.1f}%)"
        )
        print(
            "Valid-route rate: "
            f"{successful['valid_pso_route'].astype(bool).mean() * 100:.1f}%"
        )
        print(
            "Mean initial gap: "
            f"{successful['initial_optimality_gap_percent'].mean():.6f}%"
        )
        print(
            "Mean final gap: "
            f"{successful['optimality_gap_percent'].mean():.6f}%"
        )
        print(
            "Mean Dijkstra runtime: "
            f"{successful['dijkstra_runtime_ms'].mean():.3f} ms"
        )
        print(
            "Mean PSO runtime: "
            f"{successful['pso_runtime_ms'].mean():.3f} ms"
        )
        print("\nPer-run results:")
        print(successful[
            [
                "source", "destination", "seed",
                "initial_optimality_gap_percent",
                "optimality_gap_percent",
                "first_optimal_iteration",
                "valid_pso_route",
                "dijkstra_runtime_ms", "pso_runtime_ms",
            ]
        ].to_string(index=False))

    print(f"\nCombined results: {RESULTS}")


if __name__ == "__main__":
    main()
