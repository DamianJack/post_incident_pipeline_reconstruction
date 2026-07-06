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

    sweep       = config["SWEEP"]
    datasets    = sweep["datasets"]
    models_list = sweep["models"]
    results_path = sweep.get("results_path", "results.csv")

    train_results, test_results = {}, {}
    for dataset, model_name in itertools.product(datasets, models_list):
        log_path = Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)
        logfile_name = f"{log_path}/{time.strftime('%Y%m%d-%H%M%S')}-test-{dataset}-{model_name}.log"
        configure_logging(log_file=logfile_name)
        try:
            train_metrics, test_metrics = main(dataset, model_name)
        except Exception as e:
            logger.exception("[FAILED] %s x %s", dataset, model_name)
            train_metrics = {"dataset": dataset, "model": model_name, "error": str(e)}
            test_metrics  = {"dataset": dataset, "model": model_name, "error": str(e)}

        # make sure dataset/model identifiers are always present in the row
        train_metrics = {"dataset": dataset, "model": model_name, **train_metrics}
        test_metrics  = {"dataset": dataset, "model": model_name, **test_metrics}

        train_results[(dataset, model_name)] = train_metrics
        test_results[(dataset, model_name)]  = test_metrics

    _write_and_print(train_results, test_results, results_path)


def _write_and_print(train_results, test_results, results_path):
    train_rows = list(train_results.values())
    test_rows  = list(test_results.values())

    # Union of all keys across rows (handles rows with "error" vs normal metrics)
    train_fields = _ordered_union_keys(train_rows)
    test_fields  = _ordered_union_keys(test_rows)

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_fields)
        writer.writeheader()
        writer.writerows(train_rows)
        f.write("\n")
        writer = csv.DictWriter(f, fieldnames=test_fields)
        writer.writeheader()
        writer.writerows(test_rows)

    print("\n" + "=" * 200)
    print(f"Sweep complete. -> {results_path}")
    print("=" * 200)

    print("\nTrain Metrics")
    print(tabulate(train_rows, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

    print("\nTest Metrics")
    print(tabulate(test_rows, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))


def _ordered_union_keys(rows):
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


if __name__ == "__main__":
    run_sweep()