"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504221440
"""

from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import os

from ..common.utils import log_print

def dataloaderBuilder_from_config(
    cfg,
    train_transform,
    valid_transform=None,
):
    state_name = 'dataloaderBuilder_from_config'
    print()
    log_print(state_name, f"Building...")

    root_path = str(cfg['task'].get("root_path"))
    train_path = str(cfg['dataset'].get("train_path"))
    valid_path = str(cfg['dataset'].get("valid_path"))
    bsz = int(cfg['task'].get("batch_size"))

    if valid_transform is None:
        valid_transform = train_transform

    train_dataset = datasets.ImageFolder(
        root=os.path.join(root_path, train_path),
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        root=os.path.join(root_path, valid_path),
        transform=valid_transform
    )

    log_print(state_name, f"train_dataset={len(train_dataset)}")
    log_print(state_name, f"valid_dataset={len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=bsz, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=bsz, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    log_print(state_name, f"...Done\n")

    return train_loader, valid_loader












