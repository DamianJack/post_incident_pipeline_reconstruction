# Classification Training & Evaluation Pipeline

A PyTorch pipeline for training and evaluating CNN classifiers (ResNet18, VGG16, AlexNet)
across multiple medical datasets, with checkpointing, per-run logging,
and automated multi-dataset/multi-model sweeps.

**Author(s):**
- DAMIAN JEYAKUMAR— 10012545
- BHUVANA CHANDRA THOTA — 10001026

---

## Project Structure

```
.
├── Code/
│   ├── data.py            # Dataset loading, splitting, and preprocessing (get_loaders)
│   ├── models.py           # Model definitions: ResNet18, VGG16, AlexNet
│   ├── fit.py               # Trainer class: training loop, checkpointing, evaluation
│   ├── train.py             # Entry point for training a single (dataset, model) run
│   └── test.py              # Entry point for evaluating a checkpoint / running a sweep
├── config/
│   ├── data_config.json     # Per-dataset hyperparameters and paths
│   └── test_config.json     # Sweep definition (datasets x models to evaluate)
├── best_model/              # Saved checkpoints (created automatically)
├── logs/                    # Timestamped run logs (created automatically)
├── results.csv              # Aggregated sweep results (created automatically)
└── requirements.txt         # Python dependencies
```

> Note: `Code/data.py`, `Code/models.py`, and `Code/fit.py` are referenced by `train.py` and
> `test.py` but are not included in this document — see those files in the repo for
> implementation details.

---

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is used automatically if available (`torch.cuda.is_available()`);
otherwise the pipeline falls back to CPU.

---

## Configuration

### `config/data_config.json`
Defines per-dataset settings, keyed by dataset name (e.g. `lesions`, `cells`, `chest`).
Each entry supplies the parameters consumed by `train.py` / `test.py`, including:

| Key | Description |
|---|---|
| `DATA_PATH` | Path to the dataset on disk |
| `BATCH_SIZE` | Batch size for the data loaders |
| `CHANNELS` | Number of input image channels |
| `NUM_CLASSES` | Number of target classes |
| `DROP_RATE` | Dropout rate used in the model |
| `ACTIVATION` | Activation function identifier passed to the model |
| `LEARNING_RATE` | Adam optimizer learning rate |
| `WEIGHT_DECAY` | Adam optimizer weight decay (optional) |
| `EPOCHS` | Number of training epochs |

### `config/test_config.json`
Defines the sweep run by `test.py`:
```json
{
  "SWEEP": {
    "datasets": ["lesions", "cells", "chest"],
    "models": ["ResNet18", "VGG16", "AlexNet"],
    "results_path": "results.csv"
  }
}
```

---

## Usage

### Train a single model on a single dataset
```bash
python Code/train.py --data lesions --model ResNet18
```

**Arguments:**
| Flag | Default | Choices |
|---|---|---|
| `--data` | `lesions` | `lesions`, `cells`, `chest`, `organs`, `orgs`, `cifar100` |
| `--model` | `ResNet18` | `ResNet18`, `VGG16`, `AlexNet` |

This will:
1. Load the relevant dataset split (`train`/`valid`/`test`) via `get_loaders`.
2. Build the specified model using the dataset's `config/data_config.json` entry.
3. Train with `Trainer.fit`, saving the best checkpoint to `best_model/<data>_<model>.pth`.
4. Reload the best checkpoint and report **precision, recall, macro F1, and accuracy** on the
   held-out test set.
5. Write a timestamped log file to `logs/`.

### Evaluate a single checkpoint
From `Code/test.py`, `test_checkpoint(dataset, model_name, checkpoint_path=None)` loads (or
trains, if missing) a checkpoint and reports test-set metrics. If `checkpoint_path` is not
supplied, it defaults to `best_model/<dataset>_<model_name>.pth`; if that file doesn't exist,
the model is trained from scratch before evaluation.

### Run a full sweep
```bash
python Code/test.py
```
Iterates over every `(dataset, model)` combination defined in `config/test_config.json`,
evaluates each (training first if no checkpoint is found), and writes aggregated results to
`results.csv`. Failed runs are logged (via `logger.exception`) and recorded in the CSV with an
`error` column rather than halting the sweep. A per-run summary is also printed to the console.

---

## Outputs

- **`best_model/`** — one `.pth` checkpoint per `(dataset, model)` combination trained.
- **`logs/`** — one timestamped log file per training or sweep run.
- **`results.csv`** — one row per sweep run, with columns for `dataset`, `model`, `checkpoint`,
  `accuracy`, `precision`, `recall`, `macro_f1`, and `error` (if the run failed).

---

## Requirements

See `requirements.txt` in the project root for the full list of Python dependencies
