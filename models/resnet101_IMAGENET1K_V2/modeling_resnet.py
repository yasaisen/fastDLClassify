"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504221440
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# import os

from ...common.utils import log_print, get_trainable_params, highlight, highlight_show


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class FastResModel(nn.Module):
    def __init__(self, 
        num_classes: int,
        model_name: str='IMAGENET1K_V2', 
        pretrain_path=None, 
        device: str="cuda", 
        torch_dtype=torch.float32
    ):
        super().__init__()
        self.state_name = 'FastResModel'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = 288

        log_print(self.state_name, f"model_name={self.model_name}")
        log_print(self.state_name, f"num_classes={self.num_classes}")
        self.model = models.resnet101(weights=self.model_name)
        self.num_features = self.model.fc.in_features
        self.model.avgpool = GeM()
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.num_features),
            nn.Dropout(0.5),
            nn.Linear(self.num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )

        log_print(self.state_name, f"model trainable params: {get_trainable_params(self)}")
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if "fc" in name:
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
    











    