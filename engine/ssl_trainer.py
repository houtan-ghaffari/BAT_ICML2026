__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import gc
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional
import argparse

from data.bat_audio_datasets import SSLAudioSet, SSLDataTransforms
from models import MLR_Student, MLR_Teacher
from .optim import EMA_Scheduler, RiseRunDecay, ema_update
from .utils import infinite_batch_iterator


def ssl_train_step(student: torch.nn.Module,
                   teacher: torch.nn.Module,
                   x: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   data_transforms: torch.nn.Module,
                   scheduler: RiseRunDecay,
                   ema_scheduler: EMA_Scheduler,
                   clip_norm: Optional[float] = None,
                   mask_ratio: float = 0.8,
                   num_views: int = 16,
                   device: str = 'cuda',
                   accumulation_steps: int = 1,
                   is_accumulating: bool = False) -> Tuple[float, float]:
    """
    Executes a single step of Self-Supervised Learning (SSL) pretraining.
    Computes global and local representation losses between the student and teacher networks,
    performs backpropagation, and updates the teacher via Exponential Moving Average (EMA).
    """
    x = x.to(device, non_blocking=True)

    with torch.no_grad():
        # x_views.shape = (V=batch_size*views, 1, T, F)
        x_views, mask_info = data_transforms(x, num_views, mask_ratio=mask_ratio)
        x_target, _ = data_transforms(x, num_views=1, mask_ratio=0)  # Get clean target

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        student_cls_tokens, student_patch_tokens = student(x_views, mask_info)  # (V, D), (V, N, D)
        with torch.no_grad():
            _, teacher_patch_tokens = teacher(x_target)  # _, (B, N, D)

    # Convert back to float32 for loss computation
    student_cls_tokens = student_cls_tokens.float()
    student_patch_tokens = student_patch_tokens.float()
    mask = mask_info['mask'].float().to(device, non_blocking=True)

    teacher_patch_tokens = teacher_patch_tokens.repeat_interleave(num_views, dim=0).float()
    teacher_cls_tokens = teacher_patch_tokens.mean(dim=1)

    global_loss = (student_cls_tokens - teacher_cls_tokens).pow(2.).mean()
    local_loss = ((student_patch_tokens - teacher_patch_tokens).pow(2.).mean(dim=2) * mask).sum() / mask.sum()

    loss = global_loss + local_loss
    normalized_loss = loss / accumulation_steps
    normalized_loss.backward()

    if not is_accumulating:
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        decay = ema_scheduler.step()
        ema_update(teacher.encoder, student.encoder, decay=decay)

    return global_loss.item(), local_loss.item()


def run_ssl_experiment(args: argparse.Namespace) -> None:
    """
    Sets up and executes the full Self-Supervised Learning (SSL) pretraining pipeline,
    including dataset loading, student/teacher initialization, and the optimization loop.
    """
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = Path('logs/SSL')
    log_dir.mkdir(parents=True, exist_ok=True)

    history_save_path = log_dir / f'BAT_history_{time_stamp}.csv'
    state_save_path = f'BAT_state_{time_stamp}.pt'

    print(f"\n[SSL PRETRAINING] Starting on {args.device}")

    # Data
    ssl_data = np.array([f.as_posix() for f in Path(args.dataset_dir).glob("**/*.wav")]).astype(np.bytes_)
    train_dataset = SSLAudioSet(ssl_data, sr=args.sr)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=False,
                              collate_fn=train_dataset.collate_fn, persistent_workers=True, drop_last=True)

    infinite_loader = infinite_batch_iterator(train_loader)

    data_transforms = SSLDataTransforms(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
                                        n_mels=args.n_mels, time_frame_size=args.time_frame_size,
                                        num_time_patches=args.time_frame_size // args.patch_size,
                                        num_freq_patches=args.n_mels // args.patch_size).to(args.device)
    # Model
    student = MLR_Student(input_shape=(args.time_frame_size, args.n_mels),
                          patch_size=(args.patch_size, args.patch_size),
                          dim=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                          pos_trainable=False, layer_norm_first=False, pre_norm=True, use_gate=True,
                          decoder_depth=args.decoder_depth, decoder_heads=args.decoder_heads,
                          decoder_mlp_ratio=args.decoder_mlp_ratio, decoder_dim=args.decoder_dim).to(args.device)

    teacher = MLR_Teacher(input_shape=(args.time_frame_size, args.n_mels),
                          patch_size=(args.patch_size, args.patch_size),
                          dim=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                          pos_trainable=False, layer_norm_first=False, pre_norm=True, use_gate=True,
                          instance_norm_target_layer=True, layer_norm_targets=True).to(args.device)

    teacher.encoder.load_state_dict(student.encoder.state_dict())
    teacher.requires_grad_(False)

    print(f'Student params: {sum(p.numel() for p in student.parameters()):_}')
    print(f'Teacher params: {sum(p.numel() for p in teacher.parameters()):_}')

    if args.compile:
        student.compile(mode='default')
        teacher.compile(mode='default')

    ema_scheduler = EMA_Scheduler(decay_start=args.ema_decay_start, decay_end=args.ema_decay_end,
                                  ema_warmup_steps=args.ema_warmup_steps)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = RiseRunDecay(optimizer, warmup_steps=args.lr_warmup_steps, constant_steps=0,
                             total_steps=args.optimization_steps, min_lr=args.min_lr)

    # Training Loop
    total_forward_passes = args.optimization_steps * args.grad_accumulation_steps
    history = {'global_loss': [], 'local_loss': []}
    save_freq = args.save_interval if args.save_interval else args.optimization_steps
    pbar = tqdm(total=args.optimization_steps, desc="SSL Pretraining", colour='#87ceeb')
    temp_g_loss, temp_l_loss = 0.0, 0.0
    for step in range(total_forward_passes):
        x = next(infinite_loader)
        is_accumulating = (step + 1) % args.grad_accumulation_steps != 0

        g_loss, l_loss = ssl_train_step(
            student=student, teacher=teacher, x=x, optimizer=optimizer, data_transforms=data_transforms,
            scheduler=scheduler, ema_scheduler=ema_scheduler, clip_norm=args.clip_norm,
            mask_ratio=args.mask_ratio, num_views=args.num_views, device=args.device,
            accumulation_steps=args.grad_accumulation_steps, is_accumulating=is_accumulating
        )

        temp_g_loss += g_loss
        temp_l_loss += l_loss

        if not is_accumulating:
            pbar.update(1)

            temp_g_loss = temp_g_loss / args.grad_accumulation_steps
            temp_l_loss = temp_l_loss / args.grad_accumulation_steps

            history['global_loss'].append(temp_g_loss)
            history['local_loss'].append(temp_l_loss)

            pbar.set_description(
                f"Global: {temp_g_loss:.4f} | Local: {temp_l_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            temp_g_loss, temp_l_loss = 0.0, 0.0

            if pbar.n % save_freq == 0 or pbar.n == args.optimization_steps:
                current_save_path = log_dir / f'step({pbar.n})_{state_save_path}'
                torch.save({
                    'encoder': student.encoder.state_dict(),
                    'decoder': student.decoder.state_dict(),
                    'ema_encoder': teacher.encoder.state_dict(),
                    'optimized_steps': pbar.n,
                    'args': vars(args)
                }, current_save_path)

                pd.DataFrame(history).to_csv(history_save_path, index=False)

    pbar.close()

    # Cleanup
    torch._dynamo.reset()
    if hasattr(train_loader, '_iterator'):
        del train_loader._iterator
    del infinite_loader, train_loader, student, teacher, optimizer, scheduler, x
    gc.collect()
    torch.cuda.empty_cache()
