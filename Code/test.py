import csv
import json
import itertools
from pathlib import Path
import time
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders
import models
from fit import Trainer
from train import configure_logging, main
import logging

logger = logging.getLogger(__name__)

def run_sweep(config_path="config//test_config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    sweep      = config["SWEEP"]
    datasets   = sweep["datasets"]
    models_list = sweep["models"]
    results_path = sweep.get("results_path", "results.csv")

    results = []
    for dataset, model_name in itertools.product(datasets, models_list):
        log_path = Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)
        logfile_name = f"{log_path}/{time.strftime('%Y%m%d-%H%M%S')}-test-{dataset}-{model_name}.log"
        configure_logging(log_file=logfile_name)
        try:
            training_metrics, test_metrics = main(dataset, model_name)
            results.append({**training_metrics, **test_metrics})
        except Exception as e:
            logger.exception("[FAILED] %s x %s", dataset, model_name)
            results.append({"dataset": dataset, "model": model_name, "error": str(e)})

    # union of keys across all rows -> new metrics become columns automatically
    fieldnames = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 200)
    print(f"Sweep complete. {len(results)} runs -> {results_path}")
    print("=" * 200)

    print(tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))


if __name__ == "__main__":
    run_sweep()