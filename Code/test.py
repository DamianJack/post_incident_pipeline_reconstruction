import csv
import json
import itertools
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders
import models
from fit import Trainer
from train import configure_logging
import logging

logger = logging.getLogger(__name__)

def test_checkpoint(dataset, model_name, checkpoint_path=None):
    with open("config//data_config.json", "r") as f:
        cfg = json.load(f)

    data_config = cfg[dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(data=dataset, data_path=data_config["DATA_PATH"], batch_size=data_config["BATCH_SIZE"])

    model_class = getattr(models, model_name)
    model = model_class(in_channels=data_config["CHANNELS"], num_classes=data_config["NUM_CLASSES"], drop_rate=data_config.get("DROP_RATE", 0.5), activation_str=data_config.get("ACTIVATION")).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=data_config.get("LEARNING_RATE", 1e-3))

    trainer = Trainer(model, criterion, optimizer, device)

    if checkpoint_path is None:
        checkpoint_path = f"best_model_green/{dataset}_{model_name}.pth"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    logfile_name = f"{log_path}/{time.strftime('%Y%m%d-%H%M%S')}-train-{dataset}-{model_name}.log"
    configure_logging(log_file=logfile_name)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    trainer.fit(train_loader, valid_loader, epochs=data_config.get("EPOCHS", 10), checkpoint_path=str(checkpoint_path))
    train_runtime_s = time.perf_counter() - t0
    peak_train_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if use_cuda else float("nan")

    trainer.load_checkpoint(str(checkpoint_path))
    precision, recall, macro_f1, accuracy = trainer.test_eval(test_loader)

    num_params = sum(p.numel() for p in model.parameters())
    infer_latency_ms, peak_infer_mem_mb = trainer.benchmark_inference(test_loader)

    metrics = {
        "dataset": dataset,
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "num_params": num_params,
        "train_runtime_s": train_runtime_s,
        "infer_latency_ms": infer_latency_ms,
        "peak_train_mem_mb": peak_train_mem_mb,
        "peak_infer_mem_mb": peak_infer_mem_mb,
    }
    return metrics

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
            metrics = test_checkpoint(dataset, model_name)
            results.append(metrics)
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
                  f"f1={row.get('macro_f1', float('nan')):.4f}  "
                  f"checkpoint={row.get('checkpoint', 'N/A')}")


if __name__ == "__main__":
    run_sweep()