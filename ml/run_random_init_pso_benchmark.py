import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "ml" / "evaluate_forecast_pso_random_init.py"
OUT = ROOT / "ml" / "regional_model"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    (3319, 316),
    (11388, 14979),
    (11703, 6988),
    (2272, 2753),
    (2681, 15490),
]

TIMESTAMPS = [
    "2024-08-26T10:00:00",
    "2024-08-27T08:00:00",
    "2024-08-28T17:00:00",
    "2024-08-29T12:00:00",
    "2024-08-30T18:00:00",
]

SEEDS = [42, 123, 2026]
RESULTS = OUT / "forecast_pso_random_init_benchmark_results.csv"
LOGS = OUT / "forecast_pso_random_init_benchmark_logs.txt"

results = []
logs = []

total = len(PAIRS) * len(TIMESTAMPS) * len(SEEDS)
run_number = 0

for source, destination in PAIRS:
    for timestamp in TIMESTAMPS:
        for seed in SEEDS:
            run_number += 1
            command = [
                sys.executable,
                str(SCRIPT),
                "--source", str(source),
                "--destination", str(destination),
                "--timestamp", timestamp,
                "--particles", "30",
                "--iterations", "80",
                "--seed", str(seed),
            ]

            print(
                f"\n[{run_number}/{total}] {source}->{destination}, "
                f"{timestamp}, seed={seed}",
                flush=True,
            )

            started = time.time()
            proc = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            elapsed = time.time() - started
            output = proc.stdout + "\n" + proc.stderr
            logs.append(
                f"\n{'=' * 70}\nRUN {run_number}: {command}\n"
                f"Return code: {proc.returncode}; elapsed: {elapsed:.2f}s\n"
                f"{output}\n"
            )

            if proc.returncode != 0:
                results.append({
                    "source": source,
                    "destination": destination,
                    "timestamp": timestamp,
                    "seed": seed,
                    "status": "failed",
                    "elapsed_seconds": elapsed,
                    "output": output[-3000:],
                })
                print(f"FAILED; details saved to {LOGS}", flush=True)
                continue

            # The single-run script overwrites this file; read it immediately.
            summary_file = OUT / "forecast_pso_random_init_summary.csv"
            if not summary_file.exists():
                results.append({
                    "source": source,
                    "destination": destination,
                    "timestamp": timestamp,
                    "seed": seed,
                    "status": "missing_summary",
                    "elapsed_seconds": elapsed,
                })
                continue

            summary = pd.read_csv(summary_file).iloc[0].to_dict()
            summary.update({
                "source": source,
                "destination": destination,
                "timestamp": timestamp,
                "seed": seed,
                "status": "success",
                "elapsed_seconds": elapsed,
            })
            results.append(summary)

            # Save after every run, so interruption won't lose completed results.
            pd.DataFrame(results).to_csv(RESULTS, index=False)
            LOGS.write_text("".join(logs))

            print("Completed; results checkpointed.", flush=True)

pd.DataFrame(results).to_csv(RESULTS, index=False)
LOGS.write_text("".join(logs))

successful = sum(r.get("status") == "success" for r in results)
print(f"\nBenchmark finished: {successful}/{total} successful runs.")
print(f"Results: {RESULTS}")
print(f"Logs: {LOGS}")
