__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
import math
import numpy as np
import torch
from typing import Any, List, Dict, Optional


class RiseRunDecay(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: Optional[int] = None,
                 constant_steps: Optional[int] = None,
                 total_steps: Optional[int] = None,
                 min_lr: float = 1e-6):
        self.warmup_steps = warmup_steps
        self.constant_steps = warmup_steps + (constant_steps or 0)
        self.total_steps = total_steps
        self.decay_interval = total_steps - self.constant_steps
        self.min_lr = min_lr
        self.lr_scales = []
        for param_group in optimizer.param_groups:
            self.lr_scales.append(param_group.get('lr_scale', 1.0))
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        lrs = []
        current_iteration = self.last_epoch
        if current_iteration <= self.warmup_steps and self.warmup_steps > 0:
            factor = current_iteration / self.warmup_steps
        elif current_iteration <= self.constant_steps:
            factor = 1.0
        else:
            if self.decay_interval == 0:
                factor = 0.0
            else:
                decay_iteration = current_iteration - self.constant_steps
                factor = 0.5 * (1 + math.cos(math.pi * decay_iteration / self.decay_interval))

        for lr, lr_scale in zip(self.base_lrs, self.lr_scales):
            scaled_min_lr = self.min_lr * lr_scale
            scaled_current_lr = lr * factor * lr_scale
            this_lr = max(scaled_min_lr, scaled_current_lr)
            lrs.append(this_lr)
        return lrs


class EMA_Scheduler:
    def __init__(self, decay_start: float = 0.9998, decay_end: float = 0.99999, ema_warmup_steps: Optional[int] = None):
        self.decays = np.linspace(decay_start, decay_end, ema_warmup_steps, dtype=np.float32).tolist() + [decay_end]
        self.max_iter = ema_warmup_steps
        self.counter = 0

    def step(self) -> float:
        w = self.decays[self.counter]
        self.counter = min(self.counter + 1, self.max_iter)
        return float(w)


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module,
               model: torch.nn.Module,
               buffers: bool = True,
               decay: Optional[float] = None) -> None:
    for p_avg, p in zip(ema_model.parameters(), model.parameters()):
        p_avg.data = decay * p_avg.data + (1. - decay) * p.data
    if buffers:
        for (n, b_avg), (n2, b) in zip(ema_model.named_buffers(), model.named_buffers()):
            if n.split('.')[-1] == 'num_batches_tracked':
                b_avg.data = b.data
            else:
                b_avg.data = decay * b_avg.data + (1. - decay) * b.data


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('head'):
        return num_layers
    else:
        return num_layers


def param_groups_lrd(model: torch.nn.Module,
                     weight_decay: Optional[float] = None,
                     no_weight_decay_list: List[str] = ['cls_token', 'pos_embed'],
                     layer_decay: Optional[float] = None) -> List[Dict[str, Any]]:
    """Generates parameter groups with layer-wise learning rate decay for standard ViT architectures."""
    param_groups = {}
    num_layers = len(model.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay, this_decay = "no_decay", 0.
        else:
            g_decay, this_decay = "decay", weight_decay
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        if group_name not in param_groups:
            param_groups[group_name] = {"lr_scale": layer_scales[layer_id], "weight_decay": this_decay, "params": []}
        param_groups[group_name]["params"].append(p)
    return list(param_groups.values())


def get_layer_id_for_beats(name: str, num_layers: int) -> int:
    """Extracts the layer ID from a parameter name specifically for the BEATs architecture."""
    if name.startswith('patch_embedding') or name.startswith('pos_conv') or name.startswith('layer_norm') or \
            name.startswith('post_extract_proj') or name.startswith('encoder.pos_conv') or name.startswith(
        'encoder.layer_norm'):
        return 0
    elif name.startswith('encoder.layers'):
        return int(name.split('.')[2]) + 1
    elif name.startswith('head') or name.startswith('vqt'):
        return num_layers
    else:
        return num_layers


def param_groups_lrd_beats(model: torch.nn.Module,
                           weight_decay: Optional[float] = None,
                           no_weight_decay_list: List[str] = ['relative_attention_bias', 'grep_a', 'vqt'],
                           layer_decay: Optional[float] = None) -> List[Dict[str, Any]]:
    """Generates parameter groups with layer-wise learning rate decay for the BEATs architecture."""
    param_groups = {}
    num_layers = len(model.encoder.layers) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or any(nd in n for nd in no_weight_decay_list):
            g_decay, this_decay = "no_decay", 0.
        else:
            g_decay, this_decay = "decay", weight_decay
        layer_id = get_layer_id_for_beats(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        if group_name not in param_groups:
            param_groups[group_name] = {"lr_scale": layer_scales[layer_id], "weight_decay": this_decay, "params": []}
        param_groups[group_name]["params"].append(p)
    return list(param_groups.values())


def get_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Initializes and returns the AdamW optimizer based on finetuning or probing configuration."""
    if not args.finetune:
        no_decay_names = ['block_attention', 'prototype_vectors', 'vqt', 'cls_token', 'pos_embed', 'registers']
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if any(nd in name for nd in no_decay_names) or param.ndim == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optim_groups = [{'params': decay_params, 'weight_decay': args.weight_decay},
                        {'params': no_decay_params, 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    else:
        if args.lr_layer_decay is not None:
            if args.model_name == 'BEATs':
                trainable_params = param_groups_lrd_beats(model=model, weight_decay=args.weight_decay,
                                                          layer_decay=args.lr_layer_decay)
            else:
                trainable_params = param_groups_lrd(model=model, weight_decay=args.weight_decay,
                                                    layer_decay=args.lr_layer_decay)
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay,
                                          betas=(args.adam_beta1, args.adam_beta2))
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                          betas=(args.adam_beta1, args.adam_beta2))
    return optimizer
