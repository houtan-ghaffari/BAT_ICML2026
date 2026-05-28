__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
import yaml
from pathlib import Path


def get_ssl_config() -> argparse.Namespace:
    """Parses and validates configuration arguments for SSL pretraining."""
    parser = argparse.ArgumentParser(
        prog='BAT SSL',
        description='General Audio Pretraining with Better Audio Transformer.'
    )

    sys_group = parser.add_argument_group('System & Environment')
    sys_group.add_argument('-c', '--config', type=str, help='Path to YAML config file.')
    sys_group.add_argument('--device', default='cuda', type=str)
    sys_group.add_argument('--num-workers', default=32, type=int)
    sys_group.add_argument('--save-interval', type=int, help='frequency of steps to save a checkpoint.')

    data_group = parser.add_argument_group('Data & Features')
    data_group.add_argument('--dataset-dir', type=str, default='as2m', help='path to AudioSet-2M.')
    data_group.add_argument('--sr', default=16000, type=int)
    data_group.add_argument('--n-fft', default=1024, type=int)
    data_group.add_argument('--hop-length', default=160, type=int)
    data_group.add_argument('--n-mels', default=128, type=int)
    data_group.add_argument('--time-frame-size', default=1024, type=int, help='spectrogram time frames.')

    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--patch-size', default=16, type=int, help='ViT patch size.')
    model_group.add_argument('--dim', default=768, type=int)
    model_group.add_argument('--depth', default=12, type=int)
    model_group.add_argument('--num-heads', default=12, type=int)
    model_group.add_argument('--mlp-ratio', default=4, type=int)
    model_group.add_argument('--decoder-depth', default=6, type=int)
    model_group.add_argument('--decoder-heads', default=12, type=int)
    model_group.add_argument('--decoder-mlp-ratio', default=4, type=int)
    model_group.add_argument('--decoder-dim', type=int)

    train_group = parser.add_argument_group('Training & Optimization')
    train_group.add_argument('--batch-size', default=48, type=int)
    train_group.add_argument('--optimization-steps', default=400_000, type=int)
    train_group.add_argument('--grad-accumulation-steps', default=1, type=int, help='grad accumulation steps.')
    train_group.add_argument('--lr', default=0.0005, type=float, help='learning rate.')
    train_group.add_argument('--min-lr', default=0.000001, type=float, help='minimum learning rate.')
    train_group.add_argument('--lr-warmup-steps', default=40_000, type=int)
    train_group.add_argument('--weight-decay', default=0.05, type=float)
    train_group.add_argument('--clip-norm', type=float, help='gradient clipping.')
    train_group.add_argument('--ema-warmup-steps', default=100_000, type=int)
    train_group.add_argument('--ema-decay-start', default=0.9998, type=float)
    train_group.add_argument('--ema-decay-end', default=0.99999, type=float)

    aug_group = parser.add_argument_group('Augmentation & Regularization')
    aug_group.add_argument('--mask-ratio', default=0.8, type=float)
    aug_group.add_argument('--num-views', default=16, type=int, help='number of differently masked views.')

    # parse config file first, then we use the ones from command line to override them
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                safe_config = {k.replace('-', '_'): v for k, v in yaml_config.items()}
                parser.set_defaults(**safe_config)

    final_args = parser.parse_args()
    return _validate_and_format_args(final_args)


def get_task_config() -> argparse.Namespace:
    """Parses and validates configuration arguments for downstream audio tasks."""
    parser = argparse.ArgumentParser(
        prog='Downstream Audio Tasks',
        description='Downstream Audio Tasks with Better Audio Transformer.'
    )

    sys_group = parser.add_argument_group('System & Environment')
    sys_group.add_argument('-c', '--config', type=str, help='Path to YAML config file.')
    sys_group.add_argument('--device', default='cuda', type=str)
    sys_group.add_argument('--compile', action="store_true")
    sys_group.add_argument('--state-save-path', type=str)
    sys_group.add_argument('--train-num-workers', default=8, type=int)
    sys_group.add_argument('--val-num-workers', default=1, type=int)
    sys_group.add_argument('--test-num-workers', default=1, type=int)

    data_group = parser.add_argument_group('Data & Features')
    data_group.add_argument('--task', type=str,
                            choices=['as20k', 'as2m', 'hsn', 'esc50', 'scv2', 'dcase2016_task2', 'librispeech'])
    data_group.add_argument('--dataset-dir', type=Path)
    data_group.add_argument('--sr', default=16000, type=int)
    data_group.add_argument('--n-fft', default=1024, type=int)
    data_group.add_argument('--hop-length', type=int)
    data_group.add_argument('--n-mels', default=128, type=int)
    data_group.add_argument('--f-min', default=0, type=int)
    data_group.add_argument('--time-frame-size', type=int, help='spectrogram time frames.')
    data_group.add_argument('--frame-shift', type=float, default=10.0, help='hop-length in ms')
    data_group.add_argument('--frame-length', type=float, default=25.0, help='n-fft in ms')

    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--model-name', type=str, choices=['BAT', 'SSLAM', 'EAT', 'BEATs'])
    model_group.add_argument('--pretrained-path', type=str, help='Path to pretrained model state.')
    model_group.add_argument('--vit-size', type=str, choices=['s', 'b', 'l'], help='ViT Model size.')
    model_group.add_argument('--patch-size', default=16, type=int, help='ViT patch size.')
    model_group.add_argument('--dim', default=768, type=int)
    model_group.add_argument('--depth', default=12, type=int)
    model_group.add_argument('--num-heads', default=12, type=int)
    model_group.add_argument('--mlp-ratio', default=4, type=int)
    model_group.add_argument('--head', default='linear', type=str,
                             choices=['linear', 'linear_cgp', 'cgp', 'protobin', 'asr', 'h2t', 'vqt'])
    model_group.add_argument('--num-prototypes', default=10000, type=int, help='only affects protobin and cgp.')
    model_group.add_argument('--num-vqt', default=10, type=int, help='only affects vqt.')
    model_group.add_argument('--use-cls-in-head', action="store_true")
    model_group.add_argument('--frame-wise-task', action="store_true")
    model_group.add_argument('--head-norm', action="store_true")
    model_group.add_argument('--head-dropout', default=0, type=float)

    train_group = parser.add_argument_group('Training & Optimization')
    train_group.add_argument('--finetune', action="store_true")
    train_group.add_argument('--optimization-steps', default=40_000, type=int)
    train_group.add_argument('--train-batch-size', default=48, type=int)
    train_group.add_argument('--val-batch-size', default=48, type=int)
    train_group.add_argument('--test-batch-size', default=48, type=int)
    train_group.add_argument('--val-frequency', default=1, type=int)
    train_group.add_argument('--test-frequency', default=None, type=int)
    train_group.add_argument('--lr', default=5e-5, type=float, help='learning rate.')
    train_group.add_argument('--min-lr', default=1e-6, type=float, help='minimum learning rate.')
    train_group.add_argument('--lr-layer-decay', type=float, help='layer-wise learning rate decay.')
    train_group.add_argument('--lr-warmup-steps', default=4_000, type=int)
    train_group.add_argument('--weight-decay', default=0.05, type=float)
    train_group.add_argument('--adam-beta1', default=0.9, type=float)
    train_group.add_argument('--adam-beta2', default=0.999, type=float)
    train_group.add_argument('--grad-accumulation-steps', default=1, type=int, help='grad accumulation steps.')
    train_group.add_argument('--grad-clip-norm', type=float, help='gradient clipping.')
    train_group.add_argument('--use-asymmetric-loss', action="store_true")

    aug_group = parser.add_argument_group('Augmentation & Regularization')
    aug_group.add_argument('--dropout', default=0.0, type=float)
    aug_group.add_argument('--path-dropout-rate', default=0.0, type=float)
    aug_group.add_argument('--label-smoothing', default=None, type=float)
    aug_group.add_argument('--lasso-weight', default=0.0001, type=float, help='for Head to Toe')
    aug_group.add_argument('--augment', action="store_true")
    aug_group.add_argument('--use-mixup', action="store_true")
    aug_group.add_argument('--mixup-chance', default=0.9, type=float)
    aug_group.add_argument('--mixup-beta', default=0.8, type=float)
    aug_group.add_argument('--mixup-hard-label', action="store_true")
    aug_group.add_argument('--use-random-gain', action="store_true")
    aug_group.add_argument('--use-color-noise', action="store_true")
    aug_group.add_argument('--color-noise-chance', default=0.2, type=float)
    aug_group.add_argument('--use-time-roll', action="store_true")
    aug_group.add_argument('--use-tf-mask', action="store_true")
    aug_group.add_argument('--freq-mask-param', default=16, type=int)
    aug_group.add_argument('--time-mask-param', default=64, type=int)
    aug_group.add_argument('--tf-mask-repeats', default=1, type=int)
    aug_group.add_argument('--mask-ratio', default=0.0, type=float)

    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                safe_config = {k.replace('-', '_'): v for k, v in yaml_config.items()}
                parser.set_defaults(**safe_config)

    args = parser.parse_args()
    required_args = ['task', 'dataset_dir', 'time_frame_size', 'model_name']
    missing = [arg.replace('_', '-') for arg in required_args if getattr(args, arg) is None]
    if missing:
        parser.error(f"The following arguments are required (provide in YAML or CLI): {', '.join(missing)}")
    return _validate_and_format_args(args)


def _validate_and_format_args(args: argparse.Namespace) -> argparse.Namespace:

    if getattr(args, 'hop_length', None) is None and getattr(args, 'frame_shift', None) is not None:
        args.hop_length = int(args.frame_shift * args.sr / 1000.)

    if getattr(args, 'vit_size', None) is not None:
        if args.vit_size == 's':
            args.__dict__.update({'dim': 384, 'depth': 12, 'num_heads': 6})
        elif args.vit_size == 'b':
            args.__dict__.update({'dim': 768, 'depth': 12, 'num_heads': 12})
        elif args.vit_size == 'l':
            args.__dict__.update({'dim': 1024, 'depth': 24, 'num_heads': 16})
        else:
            raise ValueError(f"Unknown vit_size: {args.vit_size}")

    if hasattr(args, 'time_frame_size') and hasattr(args, 'patch_size'):
        assert args.time_frame_size % args.patch_size == 0, f"time_frame_size ({args.time_frame_size}) must be divisible by patch_size ({args.patch_size})"
        assert args.n_mels % args.patch_size == 0, f"n_mels ({args.n_mels}) must be divisible by patch_size ({args.patch_size})"
        args.num_time_patches = args.time_frame_size // args.patch_size
        args.num_freq_patches = args.n_mels // args.patch_size

    if hasattr(args, 'task'):
        if args.finetune:
            assert args.head == 'linear', f"Finetuning requires the 'linear' head, but got: {args.head}"
        if args.augment:
            assert args.finetune, "Augmentation should only be used during finetuning, not frozen probing."
        if args.task == 'dcase2016_task2':
            assert args.head != 'vqt', "DCASE2016 Task 2 does not support the VQT head."
        if args.task == 'librispeech':
            assert args.head == 'asr', f"LibriSpeech requires the 'asr' head, but got: {args.head}"
        else:
            assert args.head != 'asr', f"The 'asr' head is only for LibriSpeech. Current task: {args.task}"

        # Pretrained Path Warning
        if args.model_name in ['BEATs', 'EAT', 'SSLAM'] and not args.pretrained_path:
            print(
                f"\n[WARNING] You are using {args.model_name} but did not provide a --pretrained-path. The model will initialize with random weights!\n")

    return args