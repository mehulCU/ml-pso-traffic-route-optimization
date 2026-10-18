import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ml" / "regional_model"

RUN_SCRIPT = ROOT / "ml" / "evaluate_forecast_discrete_pso.py"
SINGLE_RESULT = MODEL_DIR / "forecast_discrete_pso_summary.csv"
OUTPUT = MODEL_DIR / "forecast_pso_multitime_results.csv"

TIMESTAMPS = [
    "2024-08-26T10:00:00",
    "2024-08-27T08:00:00",
    "2024-08-28T17:00:00",
    "2024-08-29T12:00:00",
    "2024-08-30T18:00:00",
]

PAIRS = [
    (3319, 316),
    (11388, 14979),
    (11703, 6988),
    (2272, 2753),
    (2681, 15490),
]

SEEDS = [42, 123, 2026]
KEYS = ["source", "destination", "timestamp", "seed"]


def main():
    if not RUN_SCRIPT.exists():
        raise FileNotFoundError(f"PSO script not found: {RUN_SCRIPT}")

    records = []
    total = len(TIMESTAMPS) * len(PAIRS) * len(SEEDS)
    run_number = 0

    for timestamp in TIMESTAMPS:
        for source, destination in PAIRS:
            for seed in SEEDS:
                run_number += 1
                print(
                    f"\n[{run_number}/{total}] "
                    f"{source} -> {destination}, "
                    f"time={timestamp}, seed={seed}",
                    flush=True,
                )

                command = [
                    sys.executable,
                    str(RUN_SCRIPT),
                    "--source", str(source),
                    "--destination", str(destination),
                    "--timestamp", timestamp,
                    "--seed", str(seed),
                ]

                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )

                    if result.returncode != 0:
                        print("FAILED:", result.stderr[-1500:], flush=True)
                        records.append({
                            "source": source,
                            "destination": destination,
                            "timestamp": timestamp,
                            "seed": seed,
                            "status": "failed",
                            "error": result.stderr[-1500:],
                        })
                        continue

                    if not SINGLE_RESULT.exists():
                        raise FileNotFoundError(
                            f"Expected result file not found: {SINGLE_RESULT}"
                        )

                    row = pd.read_csv(SINGLE_RESULT).iloc[0].to_dict()
                    row.update({
                        "source": source,
                        "destination": destination,
                        "timestamp": timestamp,
                        "seed": seed,
                        "status": "success",
                        "error": "",
                    })
                    records.append(row)

                    print(
                        f"Initial gap="
                        f"{float(row['initial_optimality_gap_percent']):.6f}%, "
                        f"final gap="
                        f"{float(row['optimality_gap_percent']):.6f}%, "
                        f"first optimal iteration="
                        f"{row.get('first_optimal_iteration')}",
                        flush=True,
                    )

                except Exception as exc:
                    print(f"FAILED: {exc}", flush=True)
                    records.append({
                        "source": source,
                        "destination": destination,
                        "timestamp": timestamp,
                        "seed": seed,
                        "status": "failed",
                        "error": str(exc),
                    })

                # Save progress after every run.
                pd.DataFrame(records).to_csv(OUTPUT, index=False)

    df = pd.DataFrame(records)
    successful = df[df["status"] == "success"].copy()

    print("\n========== MULTI-TIMESTAMP SUMMARY ==========")
    print(f"Total runs: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {(df['status'] != 'success').sum()}")

    if not successful.empty:
        initial_gap = pd.to_numeric(
            successful["initial_optimality_gap_percent"]
        )
        final_gap = pd.to_numeric(successful["optimality_gap_percent"])
        first_opt = pd.to_numeric(
            successful["first_optimal_iteration"], errors="coerce"
        )
        initially_optimal = initial_gap <= 1e-7
        final_optimal = final_gap <= 1e-7

        print(
            f"Initially optimal: {initially_optimal.sum()}/"
            f"{len(successful)} ({initially_optimal.mean() * 100:.2f}%)"
        )
        print(
            f"Final optimal: {final_optimal.sum()}/"
            f"{len(successful)} ({final_optimal.mean() * 100:.2f}%)"
        )
        print(
            f"Runs requiring improvement: "
            f"{(~initially_optimal).sum()}"
        )
        print("\nResults by forecast timestamp:")
        summary = successful.assign(
            initially_optimal=initially_optimal,
            final_optimal=final_optimal,
        ).groupby("timestamp").agg(
            runs=("seed", "count"),
            initially_optimal=("initially_optimal", "sum"),
            final_optimal=("final_optimal", "sum"),
            mean_initial_gap=("initial_optimality_gap_percent", "mean"),
            mean_final_gap=("optimality_gap_percent", "mean"),
            mean_pso_ms=("pso_runtime_ms", "mean"),
            mean_dijkstra_ms=("dijkstra_runtime_ms", "mean"),
        )
        print(summary.round(6).to_string())

        print("\nFirst optimal iteration counts:")
        print(first_opt.value_counts(dropna=False).sort_index().to_string())

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
