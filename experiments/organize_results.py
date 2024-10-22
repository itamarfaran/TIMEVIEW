import json
from pathlib import Path
import shutil
import pandas as pd

CLEAN_UP = False


benchmarks_dir = Path("benchmarks")

with open(benchmarks_dir.joinpath("summary.json")) as f:
    all_runs = json.load(f)


summary = []
for run in all_runs:
    run_dir = benchmarks_dir.joinpath(run["timestamp"])
    if not run_dir.is_dir():
        continue
    elif CLEAN_UP and not run_dir.joinpath("results.csv").exists():
        shutil.rmtree(run_dir)

    results: dict = run.pop("results")
    for model, statistics in results.items():
        to_append = run.copy()
        to_append["model"] = model
        to_append.update(statistics)

        tuning_file = run_dir.joinpath(model, "tuning", "hyperparameters.json")
        if tuning_file.is_file():
            with open(tuning_file, "r") as f:
                to_append.update(json.load(f))
        summary.append(to_append)

summary = pd.DataFrame(summary)
summary.to_csv("benchmarks/summary_expanded.csv", index=False)
print(summary)
