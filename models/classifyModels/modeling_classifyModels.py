"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504221440
"""

from ...common.utils import log_print, get_trainable_params, highlight, highlight_show
from ..resnet101_IMAGENET1K_V2.modeling_resnet import FastResModel
from ..vit_large_patch14_reg4_dinov2.modeling_ViT import FastViTModel


def get_classifyModel_from_config(
    cfg
):
    model_name = str(cfg['model'].get('model_name'))

    if model_name == "vit_large_patch14_reg4_dinov2.lvd142m":
        model = FastViTModel.from_config(cfg=cfg)
    if model_name == "IMAGENET1K_V2":
        model = FastResModel.from_config(cfg=cfg)

    return model












