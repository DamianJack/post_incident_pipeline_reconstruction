#----------------------
# AUTHORS:
# DAMIAN - 10012545
# BHUVAN - 10001026
#-----------------------

import torch
import logging
from pathlib import Path
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

def get_loaders(data, data_path, batch_size, val_split=0.1):

    d_path    = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path,weights_only=True)
    data_dict['train_labels'] = data_dict['train_labels'].long().flatten()
    data_dict['test_labels']  = data_dict['test_labels'].long().flatten()
    logger.debug(data_dict.keys())

    total_samples = data_dict['train_images'].shape[0]
    val_size      = int(total_samples * val_split)
    train_size    = total_samples - val_size
    logger.debug(f"Validation start index: {train_size}, Validation size: {val_size}")

    torch.manual_seed(42)  # For reproducibility
    idx          = torch.randperm(total_samples)
    train_data   = data_dict['train_images'][idx[:train_size]]
    train_labels = data_dict['train_labels'][idx[:train_size]]
    val_data     = data_dict['train_images'][idx[train_size:]]
    val_labels   = data_dict['train_labels'][idx[train_size:]]

    logger.debug(f"Train samples: {train_data.shape, train_labels.shape}, Validation samples: {val_data.shape, val_labels.shape}")
    logger.debug(f"NUM CLASSES: {len(torch.unique(train_labels))}")

    train_counts = Counter(train_labels.tolist())
    logger.debug("Train class distribution:")
    for cls in sorted(train_counts):
        logger.debug(f"  Class {cls}: {train_counts[cls]} samples ({100 * train_counts[cls] / len(train_labels):.2f}%)")

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset   = TensorDataset(val_data, val_labels)
    test_dataset  = TensorDataset(data_dict['test_images'], data_dict['test_labels'])

    test_labels = data_dict['test_labels'].numpy().flatten()
    test_counts = Counter(test_labels.tolist())
    logger.debug(f"Test samples: {data_dict['test_images'].shape, data_dict['test_labels'].shape}")
    logger.debug("Test class distribution:")
    for cls in sorted(test_counts):
        logger.debug(f"  Class {cls}: {test_counts[cls]} samples ({100 * test_counts[cls] / len(test_labels):.2f}%)")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    train_loader, val_loader, test_loader = get_loaders("organs", "data", batch_size=64, val_split=0.2)