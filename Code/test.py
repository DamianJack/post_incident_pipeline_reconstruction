import csv
import json
import itertools
from pathlib import Path
import time

from train import main, configure_logging

def run_sweep(config_path="config//test_config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    sweep        = config["SWEEP"]
    datasets     = sweep["datasets"]
    models       = sweep["models"]
    results_path = sweep.get("results_path", "results.csv")

    results = []
    for dataset, model_name in itertools.product(datasets, models):
        log_path = Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)
        logfile_name = f"{log_path}/{time.strftime('%Y%m%d-%H%M%S')}-train-{dataset}-{model_name}.log"
        configure_logging(log_file=logfile_name)

        print("\n" + "=" * 60)
        print(f"RUN: dataset={dataset} | model={model_name}")
        print("=" * 60)
        try:
            metrics = main(dataset, model_name)   # returns the dict from train.py
            results.append(metrics)
        except Exception as e:                    
            print(f"[FAILED] {dataset} x {model_name}: {e}")
            results.append({"dataset": dataset, "model": model_name, "error": str(e)})

    # union of keys across all rows -> new Part 2/3 metrics become columns automatically
    fieldnames = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"Sweep complete. {len(results)} runs -> {results_path}")
    print("=" * 60)
    for row in results:
        if "error" in row:
            print(f"{row['dataset']:<10} {row['model']:<10} FAILED")
        else:
            print(f"{row['dataset']:<10} {row['model']:<10} "
                  f"acc={row.get('accuracy', float('nan')):6.2f}  "
                  f"prec={row.get('precision', float('nan')):.4f}  "
                  f"rec={row.get('recall', float('nan')):.4f}  "
                  f"f1={row.get('macro_f1', float('nan')):.4f}")

if __name__ == "__main__":
    run_sweep()