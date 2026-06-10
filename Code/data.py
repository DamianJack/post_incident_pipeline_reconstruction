#----------------------
# AUTHORS:
# DAMIAN - 10012545
# BHUVAN - 10001026
#-----------------------

import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):

    d_path    = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)
    # print(data_dict.keys())

    total_samples = data_dict['train_images'].shape[0]
    val_size      = int(total_samples * val_split)
    train_size    = total_samples - val_size
    # print(f"Validation start index: {val_start}, Validation size: {val_size}")
    
    torch.manual_seed(42)  # For reproducibility
    idx          = torch.randperm(total_samples)
    train_data   = data_dict['train_images'][idx[:train_size]]
    train_labels = data_dict['train_labels'][idx[:train_size]]
    val_data     = data_dict['train_images'][idx[train_size:]]
    val_labels   = data_dict['train_labels'][idx[train_size:]]
    # print(f"Train samples: {train_data.shape, train_labels.shape}, Validation samples: {val_data.shape, val_labels.shape}")

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset   = TensorDataset(val_data, val_labels)
    test_dataset  = TensorDataset(data_dict['test_images'], data_dict['test_labels'])
    # print(f"Test samples: {data_dict['test_images'].shape, data_dict['test_labels'].shape}")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# train_loader, val_loader, test_loader = get_loaders('cells', 'data', batch_size=64, val_split=0.2)