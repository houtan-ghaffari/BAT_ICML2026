__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
from typing import Dict, Any
import torch
from .vit import ViT
from .beats import BEATs

def load_bat_audioset_pretrained_state(model: torch.nn.Module, state_path: str = '') -> torch.nn.Module:
    """Loads pre-trained BAT weights from an AudioSet checkpoint into the model."""
    audioset_state = torch.load(state_path, map_location='cpu', weights_only=False)
    if 'state_dict' in audioset_state:
        audioset_state = audioset_state['state_dict']

    encoder_state_dict: Dict[str, Any] = {}
    for k, v in audioset_state.items():
        if 'student.encoder' in k:
            param_key = k.replace('student.encoder.', '')  # remove the full prefix!
            if 'pos_embed' == param_key:
                encoder_state_dict[param_key] = v[:, :model.pos_embed.shape[1]].clone()
            else:
                encoder_state_dict[param_key] = v.clone()

    msg = model.load_state_dict(encoder_state_dict, strict=False)
    print(msg)
    return model

def load_eat_audioset_pretrained_state(model: torch.nn.Module, state_path: str = '') -> torch.nn.Module:
    """Loads pre-trained EAT or SSLAM weights from an AudioSet checkpoint into the model."""
    audioset_state = torch.load(state_path, map_location='cpu', weights_only=False)
    model_state = model.state_dict()

    model_state['cls_token'] = audioset_state['model']['modality_encoders.IMAGE.extra_tokens'].clone()
    model_state['patch_embed.proj.weight'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.weight'].clone()
    model_state['patch_embed.proj.bias'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.bias'].clone()
    model_state['pos_embed'] = audioset_state['model']['modality_encoders.IMAGE.fixed_positional_encoder.positions'][:, :model.pos_embed.shape[1]].clone()
    model_state['pre_norm.weight'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.weight'].clone()
    model_state['pre_norm.bias'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.bias'].clone()

    for k in audioset_state['model'].keys():
        if k[:6] == 'blocks':
            model_state[k] = audioset_state['model'][k].clone()

    msg = model.load_state_dict(model_state, strict=False)
    print(msg)
    return model


def get_model(args: argparse.Namespace, use_gate: bool = True) -> torch.nn.Module:
    """
    Instantiates the target architecture (BEATs or ViT), loads pre-trained states if specified,
    and configures gradient requirements for probing vs. finetuning.
    """
    if args.model_name == 'BEATs':
        model = BEATs(num_classes=args.num_classes, head=args.head, num_prototypes=args.num_prototypes,
                      frame_wise_task=args.frame_wise_task, use_cls_in_head=args.use_cls_in_head,
                      head_dropout=args.head_dropout, head_norm=args.head_norm, num_vqt=args.num_vqt)

        if args.pretrained_path:
            s = torch.load(args.pretrained_path, weights_only=False)
            msg = model.load_state_dict(s['model'], strict=False)
            print(f"[Info] Loaded BEATs pretrained weights: {msg}")
        else:
            print("[Info] Initialized BEATs with random weights.")

    else:
        model = ViT(num_classes=args.num_classes, input_shape=(args.time_frame_size, args.n_mels),
                    dim=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                    drop=args.dropout, use_gate=use_gate, head=args.head, drop_path_rate=args.path_dropout_rate,
                    num_prototypes=args.num_prototypes, frame_wise_task=args.frame_wise_task,
                    use_cls_in_head=args.use_cls_in_head, head_dropout=args.head_dropout,
                    head_norm=args.head_norm, num_vqt=args.num_vqt)

        if args.pretrained_path:
            if args.model_name == 'BAT':
                model = load_bat_audioset_pretrained_state(model, args.pretrained_path)
            elif args.model_name in ['EAT', 'SSLAM']:
                model = load_eat_audioset_pretrained_state(model, args.pretrained_path)
            else:
                raise ValueError(f"Unknown model name: {args.model_name}")
        else:
            print(f"[Info] Initialized {args.model_name} with random weights.")

    model.to(args.device)

    if not args.finetune:
        model.requires_grad_(False)
        model.head.requires_grad_(True)
        if args.head == 'vqt':
            model.vqt.requires_grad_(True)

    if args.compile:
        model.compile()

    return model
