"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504221440
"""

import torch
import torch.nn as nn
import timm
# import os

from ...common.utils import log_print, get_trainable_params, highlight, highlight_show


class FastViTModel(nn.Module):
    def __init__(self, 
        num_classes: int,
        model_name: str='vit_large_patch14_reg4_dinov2.lvd142m', 
        pretrain_path=None, 
        device: str="cuda", 
        torch_dtype=torch.float32
    ):
        super().__init__()
        self.state_name = 'FastViTModel'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = 518

        log_print(self.state_name, f"model_name={self.model_name}")
        log_print(self.state_name, f"num_classes={self.num_classes}")
        self.model = timm.create_model(self.model_name, pretrained=True)
        self.num_features = self.model.num_features
        self.model.head = nn.Sequential(
            nn.BatchNorm1d(self.num_features),
            nn.Dropout(0.5),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )

        log_print(self.state_name, f"model trainable params: {get_trainable_params(self)}")
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if "head" in name:
                param.requires_grad = True
        log_print(self.state_name, f"model trainable params: {get_trainable_params(self)}")
        self.to(self.device)
        log_print(self.state_name, f"...Done\n")

    def forward(self,
        input
    ):
        output = self.model(input)
        return output

    @classmethod
    def from_config(cls, cfg):
        # root_path = cfg['task'].get("root_path")
        device = str(cfg['task'].get("device"))

        model_name = str(cfg['model'].get('model_name'))
        num_classes = int(cfg['model'].get('num_classes'))

        model = cls(
            model_name=model_name, 
            num_classes=num_classes,
            device=device,
        )
        return model
    











    