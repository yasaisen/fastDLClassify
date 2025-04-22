"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504221440
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from typing import List, Tuple, Dict
from tqdm import tqdm

from ...common.utils import log_print, highlight_show, highlight


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = nn.functional.log_softmax(pred, dim=1)
        return torch.mean(torch.sum(-smooth_one_hot * log_prob, dim=1))

class imgClassifyTrainer:
    def __init__(self,
        model: nn.Module,
        device: str = "cuda",
        max_norm: float = 1.0,
        smoothing: float = 0.15,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4, 
        max_lr: float = 1e-3, 
        num_epoch: int = 30, 
        steps_per_epoch: int = 30, 
        pct_start: float = 0.2, 
        anneal_strategy: str = 'cos', 
    ):
        self.state_name = 'imgClassifyTrainer'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.max_norm = max_norm
        self.model = model.to(self.device)

        self.smoothing = smoothing
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.max_lr = max_lr
        self.num_epoch = num_epoch
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        
        self.criterion = LabelSmoothingLoss(smoothing=self.smoothing)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            epochs=self.num_epoch,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy
        )
        log_print(self.state_name, f"...Done\n")

    def init_training_stats(self,
    ):
        self.training_stats = {
            'train_steps': 1,
            'train_num_samples': 0,
            'train_num_correct': 0,
            'train_sum_loss': 0,
            'valid_steps': 1,
            'valid_num_samples': 0,
            'valid_num_correct': 0,
            'valid_sum_loss': 0,
        }

    def step_compute(self,
        inputs, 
        labels,
        training: bool = False, 
    ):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        if training:
            self.model.train()
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            _, predicted = outputs.max(1)

            num_samples = labels.size(0)
            num_correct = predicted.eq(labels).sum().item()
            accuracy = num_correct / num_samples

            self.training_stats['train_num_samples'] += num_samples
            self.training_stats['train_num_correct'] += num_correct
            self.training_stats['train_sum_loss'] += loss.item()
            avg_accuracy = self.training_stats['train_num_correct'] / self.training_stats['train_num_samples']
            avg_loss = self.training_stats['train_sum_loss'] / self.training_stats['train_steps']
            metrics = {
                'steps': self.training_stats['train_steps'], 
                'num_samples': num_samples,
                'num_correct': num_correct,
                'accuracy': accuracy,
                'avg_accuracy': avg_accuracy,
                'avg_loss': avg_loss,
                'loss': loss.item(),
                'stage': 'train',
            }
            self.training_stats['train_steps'] += 1

        else:
            self.model.eval()
            self.optimizer.zero_grad()
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = outputs.max(1)

                num_samples = labels.size(0)
                num_correct = predicted.eq(labels).sum().item()
                accuracy = num_correct / num_samples

                self.training_stats['valid_num_samples'] += num_samples
                self.training_stats['valid_num_correct'] += num_correct
                self.training_stats['valid_sum_loss'] += loss.item()
                avg_accuracy = self.training_stats['valid_num_correct'] / self.training_stats['valid_num_samples']
                avg_loss = self.training_stats['valid_sum_loss'] / self.training_stats['valid_steps']
                metrics = {
                    'steps': self.training_stats['valid_steps'], 
                    'num_samples': num_samples,
                    'num_correct': num_correct,
                    'accuracy': accuracy,
                    'avg_accuracy': avg_accuracy,
                    'avg_loss': avg_loss,
                    'loss': loss.item(),
                    'stage': 'valid',
                }
                self.training_stats['valid_steps'] += 1

        torch.cuda.empty_cache
        return metrics

    @classmethod
    def from_config(cls, 
        cfg, 
        model: nn.Module,
        steps_per_epoch: int,
    ):
        if cfg.get("task") is not None:
            trainer_cfg = cfg['task']
            device = str(trainer_cfg.get("device"))
            max_norm = float(trainer_cfg.get("max_norm"))
            smoothing = float(trainer_cfg.get("smoothing"))
            learning_rate = float(trainer_cfg.get("learning_rate"))
            weight_decay = float(trainer_cfg.get("weight_decay"))
            max_lr = float(trainer_cfg.get("max_lr"))
            num_epoch = int(trainer_cfg.get("num_epoch"))
            pct_start = float(trainer_cfg.get("pct_start"))
            anneal_strategy = str(trainer_cfg.get("anneal_strategy"))

        trainer = cls(
            model=model,
            device=device,
            max_norm=max_norm,
            smoothing=smoothing,
            learning_rate=learning_rate,
            weight_decay=weight_decay, 
            max_lr=max_lr, 
            num_epoch=num_epoch, 
            steps_per_epoch=steps_per_epoch, 
            pct_start=pct_start, 
            anneal_strategy=anneal_strategy, 
        )
        return trainer
    











