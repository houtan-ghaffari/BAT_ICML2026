__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
import gc
from datetime import datetime
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloaders
from models import get_model
from .optim import RiseRunDecay, get_optimizer
from .evaluator import evaluate_multilabel, evaluate_multiclass, evaluate_asr, evaluate_dcase_val, eval_test_set_dcase
from .utils import infinite_batch_iterator, AsymmetricLossMultiLabel


def train_step(model: torch.nn.Module,
               x: torch.Tensor,
               y: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               scheduler: Any,
               scaler: Optional[torch.amp.GradScaler] = None,
               mask_info: Optional[Dict[str, torch.Tensor]] = None,
               input_length: Optional[torch.Tensor] = None,
               target_length: Optional[torch.Tensor] = None,
               grad_clip_norm: Optional[float] = None,
               accumulate: bool = False,
               accumulation_steps: int = 1,
               lasso_weight: float = 0.0001,
               amp_dtype: Optional[torch.dtype] = None,
               loss_fn: Optional[Callable] = None) -> float:
    """
    Executes a single training step including the forward pass, loss computation, and backpropagation.
    Supports mixed precision (AMP), gradient accumulation, gradient clipping, and optional group lasso penalties.
    """
    with torch.autocast(device_type=x.device.type, dtype=amp_dtype):
        p = model(x, mask_info)

    if input_length is not None:
        loss = loss_fn(p.float(), y, input_length, target_length)
    else:
        loss = loss_fn(p.float(), y)

    if model.head_name == 'h2t':
        loss = loss + lasso_weight * model.head.compute_group_lasso_penalty()

    normalized_loss = loss / accumulation_steps

    if scaler is None:
        normalized_loss.backward()
        if not accumulate:
            if grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    else:
        scaler.scale(normalized_loss).backward()
        if not accumulate:
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            current_scale = scaler.get_scale()
            scaler.update()
            if not (current_scale > scaler.get_scale()) and scheduler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return loss.item()


def run_experiment(args: argparse.Namespace, fold_idx: Optional[int] = None) -> Tuple[
    Dict[str, float], Dict[str, list]]:
    """
    Main execution logic for setting up data, model, training loop, metric tracking, and checkpointing.
    """
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    is_legacy = args.model_name in ['EAT', 'SSLAM', 'BEATs']

    save_prefix = f"logs/{args.task}/{args.model_name}"
    if args.vit_size: save_prefix += f"({args.vit_size})"
    if fold_idx: save_prefix += f"_fold{fold_idx}"

    if args.state_save_path is None:
        mode_suffix = "finetuned" if args.finetune else f"probe({args.head})"
        args.state_save_path = f"{save_prefix}_{args.task}_{mode_suffix}_{time_stamp}.pt"

    Path(args.state_save_path).parent.mkdir(parents=True, exist_ok=True)
    temp_save_path = f"temp_state_{time_stamp}.pt"

    print(f"\n[{args.task.upper()}] Initializing {args.model_name}-{args.head} (Finetune: {args.finetune})")

    train_loader, val_loader, test_loader, data_transforms, tokenizer, vocab_df = get_dataloaders(args, is_legacy,
                                                                                                  fold_idx)
    infinite_train_loader = infinite_batch_iterator(train_loader)
    track_test = True if test_loader and getattr(args, 'test_frequency', 0) > 0 else False
    track_val = True if val_loader and getattr(args, 'val_frequency', 0) > 0 else False

    model = get_model(args, use_gate=(not is_legacy))
    optimizer = get_optimizer(model, args)
    scheduler = RiseRunDecay(optimizer, warmup_steps=args.lr_warmup_steps, constant_steps=0,
                             total_steps=args.optimization_steps, min_lr=args.min_lr)
    amp_dtype = torch.float16 if is_legacy else torch.bfloat16
    grad_scaler = torch.amp.GradScaler(args.device) if is_legacy else None

    total_forward_passes = args.optimization_steps * args.grad_accumulation_steps
    history = defaultdict(list)
    best_score = 100 if args.task == 'librispeech' else 0

    if args.use_asymmetric_loss:
        loss_fn = AsymmetricLossMultiLabel(gamma_neg=1, gamma_pos=0, clip=0.0)
    else:
        loss_fn = F.binary_cross_entropy_with_logits
    eval_fn = partial(evaluate_multilabel, data_transforms=data_transforms, device=args.device, amp_dtype=amp_dtype)
    is_best = lambda old_, new: new >= old_
    metric_to_track = None

    if args.task == 'librispeech':
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        metric_to_track = 'wer'
        is_best = lambda old_, new_: new_ <= old_
        eval_fn = partial(evaluate_asr, tokenizer=tokenizer, device=args.device, amp_dtype=amp_dtype)

    elif args.task == 'scv2':
        loss_fn = partial(F.cross_entropy, label_smoothing=args.label_smoothing if args.label_smoothing else 0.0)
        metric_to_track = 'accuracy'
        eval_fn = partial(evaluate_multiclass, data_transforms=data_transforms, device=args.device, amp_dtype=amp_dtype)

    elif args.task == 'esc50':
        loss_fn = partial(F.cross_entropy, label_smoothing=args.label_smoothing if args.label_smoothing else 0.0)
        eval_fn = partial(evaluate_multiclass, data_transforms=data_transforms, device=args.device, amp_dtype=amp_dtype)

    elif args.task == 'dcase2016_task2':
        metric_to_track = 'mAP_micro'
        eval_fn = partial(evaluate_dcase_val, device=args.device, amp_dtype=amp_dtype)

    val_msg, test_msg = "", ""
    pbar = tqdm(total=args.optimization_steps, desc=f"Fold {fold_idx}" if fold_idx else "Training")
    for step in range(total_forward_passes):
        batch = next(infinite_train_loader)
        x, y, in_len, tar_len, mask_info = (None,) * 5
        if args.task == 'librispeech':
            x, y, in_len, tar_len = [b.to(args.device, non_blocking=True) for b in batch]
        else:
            x, y = [b.to(args.device, non_blocking=True) for b in batch]
            if args.task != 'dcase2016_task2':
                with torch.no_grad():
                    x, y, mask_info = data_transforms(x, y, mask_ratio=args.mask_ratio, augment=args.augment)

        is_accumulating = (step + 1) % args.grad_accumulation_steps != 0

        loss_val = train_step(model=model, x=x, y=y, optimizer=optimizer, scheduler=scheduler, scaler=grad_scaler,
                              mask_info=mask_info, input_length=in_len, target_length=tar_len, amp_dtype=amp_dtype,
                              grad_clip_norm=args.grad_clip_norm, accumulate=is_accumulating, loss_fn=loss_fn,
                              accumulation_steps=args.grad_accumulation_steps, lasso_weight=args.lasso_weight)

        if not is_accumulating:
            pbar.update(1)

            if track_val and (pbar.n % args.val_frequency == 0 or pbar.n == args.optimization_steps):
                val_msg = ""
                model.eval()
                val_metrics = eval_fn(model, val_loader)
                model.train()

                for k, v in val_metrics.items():
                    history[f'val_{k}'].append(v)
                    val_msg += f" | val_{k}: {v:.2f}"

                if is_best(best_score, val_metrics[metric_to_track]):
                    best_score = val_metrics[metric_to_track]
                    torch.save(model.state_dict(), temp_save_path)

            if track_test and (pbar.n % args.test_frequency == 0 or pbar.n == args.optimization_steps):
                test_msg = ""
                model.eval()
                test_metrics = eval_fn(model, test_loader)
                model.train()
                for k, v in test_metrics.items():
                    key = f'fold{fold_idx}_test_{k}' if fold_idx else f'test_{k}'
                    history[key].append(v)
                    test_msg += f" | test_{k}: {v:.2f}"

            pbar.set_description(f"Loss: {loss_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}" + val_msg + test_msg)

    pbar.close()

    if Path(temp_save_path).exists():
        model.load_state_dict(torch.load(temp_save_path))
        Path(temp_save_path).unlink()

    model.eval()
    if args.task == 'dcase2016_task2':
        test_metrics = eval_test_set_dcase(data_dir=args.dataset_dir, vocab_df=vocab_df, model=model, threshold=0.3,
                                           sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
                                           legacy=is_legacy, model_name=args.model_name, amp_dtype=amp_dtype,
                                           num_temporal_patches=args.num_temporal_patches, device=args.device,
                                           patch_stride_ms=args.patch_stride_ms, filter_size=3)
    else:
        test_metrics = eval_fn(model, test_loader)

    print(f"\n[{args.task.upper()}] Test Metrics:")
    for k, v in test_metrics.items(): print(f"  {k}: {v:.2f}")

    results = {'history': history, 'test_metrics': test_metrics, 'args': vars(args)}
    if args.save_model: results['state'] = model.state_dict(),
    torch.save(results, args.state_save_path)

    # cleanup
    torch._dynamo.reset()
    if hasattr(train_loader, '_iterator'): del train_loader._iterator
    if hasattr(test_loader, '_iterator'): del test_loader._iterator
    if hasattr(val_loader, '_iterator'): del val_loader._iterator
    del infinite_train_loader, train_loader, test_loader, val_loader, data_transforms, model, optimizer, scheduler, x, y
    gc.collect()
    torch.cuda.empty_cache()

    return test_metrics, history