#----------------------
# AUTHORS:
# DAMIAN - 10012545
# BHUVAN - 10001026
#-----------------------

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logger = logging.getLogger(__name__)

def configure_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=handlers, force=True)

def main(data, data_model):   
    with open("config//data_config.json", "r") as f:
        config = json.load(f)

    data_config = config[data]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training executing on device: {device}")

    train_loader, val_loader, test_loader = get_loaders(data=data, data_path=data_config["DATA_PATH"], batch_size=data_config["BATCH_SIZE"])

    model_class = getattr(models, data_model)
    model = model_class(in_channels=data_config["CHANNELS"], num_classes=data_config["NUM_CLASSES"], drop_rate=data_config["DROP_RATE"], activation_str=data_config["ACTIVATION"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=data_config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    checkpoint_name = f"{data}_{data_model}.pth"
    best_model_dir = Path("best_model")
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{best_model_dir}/{checkpoint_name}"

    trainer.fit(train_loader, val_loader, epochs=data_config["EPOCHS"], checkpoint_path=str(checkpoint_path))
    trainer.load_checkpoint(str(checkpoint_path))

    precision, recall, macro_f1, accuracy = trainer.test_eval(test_loader)
    logger.info(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, Macro F1-Score: {macro_f1:.4f}, Accuracy: {accuracy:.4f}%")
    metrics = {
        "dataset":   data,
        "model":     data_model,
        "checkpoint": str(checkpoint_path),
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "macro_f1":  macro_f1,
    }
    return metrics

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Train a model on the specified dataset.")
    parser.add_argument("--data", type=str, default="lesions", help="Dataset to use (default: lesions)", choices=["lesions", "cells", "chest", "organs", "orgs", "cifar100"])
    parser.add_argument("--model", type=str, default="ResNet18", help="Model architecture to use (default: ResNet18)", choices=["ResNet18", "VGG16", "AlexNet"])
    args = parser.parse_args()

    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    logfileName = f"{log_path}/{time.strftime('%Y%m%d-%H%M%S')}-train-{args.data}-{args.model}.log"
    configure_logging(log_file=logfileName)
    
    main(args.data, args.model)
    