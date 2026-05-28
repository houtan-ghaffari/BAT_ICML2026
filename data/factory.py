__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
from typing import Optional, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from .bat_audio_datasets import (AudioSet, CharTokenizer, ESC50Dataset, BirdSet, SpeechCommands, DCASE2016Task2Dataset,
                                 LibriSpeechDataset, DataTransforms, curate_audioset_dfs, prepare_esc50_dfs,
                                 prepare_hsn_dfs, prepare_speech_commands_dfs, load_dcase2016_task2_json_to_df,
                                 prepare_librispeech)

from .legacy_audio_datasets import (LegacyAudioSet, LegacyESC50Dataset, LegacyBirdSet, LegacySpeechCommands,
                                    LegacyDCASE2016Task2Dataset, LegacyLibriSpeechDataset, LegacyDataTransforms)


def get_dataloaders(args: argparse.Namespace,
                    is_legacy: bool = False,
                    fold_idx: Optional[int] = None,
                    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader],
                         Optional[torch.nn.Module], Optional[CharTokenizer], Optional[pd.DataFrame]]:

    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    tokenizer: Optional[CharTokenizer] = None
    vocab_df: Optional[pd.DataFrame] = None
    sampler: Optional[WeightedRandomSampler] = None

    if args.task in ['as20k', 'as2m']:
        args.num_classes = 527
        train_df, eval_df = curate_audioset_dfs(root_dir=args.dataset_dir, version=args.task)

        if is_legacy:
            train_dataset = LegacyAudioSet(train_df, sr=args.sr, train=True, augment=args.augment,
                                           use_time_roll=args.use_time_roll, use_color_noise=args.use_color_noise,
                                           color_noise_chance=args.color_noise_chance,
                                           time_frame_size=args.time_frame_size, n_mels=args.n_mels,
                                           frame_shift=args.frame_shift, frame_length=args.frame_length,
                                           model_name=args.model_name)
            test_dataset = LegacyAudioSet(eval_df, sr=args.sr, time_frame_size=args.time_frame_size, n_mels=args.n_mels,
                                          frame_shift=args.frame_shift, frame_length=args.frame_length,
                                          model_name=args.model_name)
        else:
            train_dataset = AudioSet(train_df, sr=args.sr, train=True)
            test_dataset = AudioSet(eval_df, sr=args.sr)

        if args.task == 'as2m':
            parsed_codes = train_df['codes'].apply(lambda x: [int(c) for c in x.decode('UTF-8').split(',') if c])
            class_counts_series = parsed_codes.explode().value_counts()
            class_counts = np.zeros(527)
            class_counts[class_counts_series.index.astype(int)] = class_counts_series.values
            class_weights = 1.0 / (class_counts + 1e-8)
            train_df['sample_weight'] = parsed_codes.apply(
                lambda labels: sum(class_weights[c] for c in labels) / len(labels) if labels else 1.0)
            sampler = WeightedRandomSampler(weights=train_df['sample_weight'].values, num_samples=200_000,
                                            replacement=False)

    elif args.task == 'esc50':
        args.num_classes = 50
        train_df, test_df = prepare_esc50_dfs(args.dataset_dir, fold_idx)
        if is_legacy:
            train_dataset = LegacyESC50Dataset(train_df, sr=args.sr, n_mels=args.n_mels, frame_shift=args.frame_shift,
                                               frame_length=args.frame_length, augment=args.augment,
                                               use_time_roll=args.use_time_roll, use_color_noise=args.use_color_noise,
                                               color_noise_chance=args.color_noise_chance, model_name=args.model_name,
                                               time_frame_size=args.time_frame_size, one_hot_labels=args.use_mixup)
            test_dataset = LegacyESC50Dataset(test_df, sr=args.sr, n_mels=args.n_mels, frame_shift=args.frame_shift,
                                              frame_length=args.frame_length, time_frame_size=args.time_frame_size,
                                              model_name=args.model_name, one_hot_labels=False)
        else:
            train_dataset = ESC50Dataset(train_df, sr=args.sr, one_hot_labels=args.use_mixup)
            test_dataset = ESC50Dataset(test_df, sr=args.sr, one_hot_labels=False)

    elif args.task == 'hsn':
        train_df, test_df, num_classes = prepare_hsn_dfs(cache_dir=args.dataset_dir)
        args.num_classes = num_classes

        class_counts = np.bincount(train_df['ebird_code_multilabel'].apply(lambda x: x[0]).values)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[train_df['ebird_code_multilabel'].apply(lambda x: x[0]).values]
        sample_weights = torch.from_numpy(sample_weights).float()
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        if is_legacy:
            train_dataset = LegacyBirdSet(train_df, sr=args.sr, duration_seconds=5, num_classes=num_classes, test=False,
                                          n_mels=args.n_mels, frame_shift=args.frame_shift,
                                          frame_length=args.frame_length, augment=args.augment, use_random_gain=False,
                                          model_name=args.model_name, use_time_roll=args.use_time_roll,
                                          use_color_noise=args.use_color_noise,
                                          color_noise_chance=args.color_noise_chance,
                                          time_frame_size=args.time_frame_size)
            test_dataset = LegacyBirdSet(test_df, sr=args.sr, duration_seconds=5, num_classes=num_classes, test=True,
                                         n_mels=args.n_mels, frame_shift=args.frame_shift,
                                         frame_length=args.frame_length, time_frame_size=args.time_frame_size,
                                         model_name=args.model_name)
        else:
            train_dataset = BirdSet(train_df, sr=args.sr, duration_seconds=5, num_classes=num_classes, test=False)
            test_dataset = BirdSet(test_df, sr=args.sr, duration_seconds=5, num_classes=num_classes, test=True)

    elif args.task == 'scv2':
        args.num_classes = 12
        train_df, val_df, test_df = prepare_speech_commands_dfs(args.dataset_dir)
        if is_legacy:
            train_dataset = LegacySpeechCommands(train_df, sr=args.sr, n_mels=args.n_mels, frame_shift=args.frame_shift,
                                                 frame_length=args.frame_length, augment=args.augment,
                                                 use_time_roll=args.use_time_roll, use_color_noise=args.use_color_noise,
                                                 color_noise_chance=args.color_noise_chance,
                                                 time_frame_size=args.time_frame_size, model_name=args.model_name,
                                                 one_hot_labels=args.use_mixup)
            val_dataset = LegacySpeechCommands(val_df, sr=args.sr, n_mels=args.n_mels, frame_shift=args.frame_shift,
                                               frame_length=args.frame_length, time_frame_size=args.time_frame_size,
                                               model_name=args.model_name)
            test_dataset = LegacySpeechCommands(test_df, sr=args.sr, n_mels=args.n_mels, frame_shift=args.frame_shift,
                                                frame_length=args.frame_length, time_frame_size=args.time_frame_size,
                                                model_name=args.model_name)
        else:
            train_dataset = SpeechCommands(train_df, sr=args.sr, one_hot_labels=args.use_mixup)
            val_dataset = SpeechCommands(val_df, sr=args.sr, one_hot_labels=False)
            test_dataset = SpeechCommands(test_df, sr=args.sr, one_hot_labels=False)

    elif args.task == 'dcase2016_task2':
        vocab_df = pd.read_csv(Path(args.dataset_dir).joinpath('labelvocabulary.csv'))
        args.num_classes = len(vocab_df)
        args.patch_stride_ms = int(1000 * (args.hop_length / args.sr) * args.patch_size)
        args.num_temporal_patches = int(120 * (args.sr / args.hop_length) / args.patch_size)
        train_df = load_dcase2016_task2_json_to_df(args.dataset_dir, split='train')
        val_df = load_dcase2016_task2_json_to_df(args.dataset_dir, split='valid')

        if is_legacy:
            train_dataset = LegacyDCASE2016Task2Dataset(train_df, vocab_df, sr=args.sr, n_mels=args.n_mels,
                                                        frame_shift=args.frame_shift, frame_length=args.frame_length,
                                                        patch_stride_ms=args.patch_stride_ms,
                                                        num_temporal_patches=args.num_temporal_patches,
                                                        augment=args.augment, use_mixup=args.use_mixup,
                                                        use_color_noise=args.use_color_noise,
                                                        use_tf_mask=args.use_tf_mask, mixup_beta=args.mixup_beta,
                                                        mixup_chance=args.mixup_chance,
                                                        mixup_hard_label=args.mixup_hard_label,
                                                        color_noise_chance=args.color_noise_chance,
                                                        freq_mask_param=args.freq_mask_param,
                                                        time_mask_param=args.time_mask_param,
                                                        time_frame_size=args.time_frame_size,
                                                        model_name=args.model_name)

            val_dataset = LegacyDCASE2016Task2Dataset(val_df, vocab_df, sr=args.sr, n_mels=args.n_mels,
                                                      frame_shift=args.frame_shift, frame_length=args.frame_length,
                                                      patch_stride_ms=args.patch_stride_ms,
                                                      num_temporal_patches=args.num_temporal_patches,
                                                      time_frame_size=args.time_frame_size, model_name=args.model_name)
        else:
            train_dataset = DCASE2016Task2Dataset(train_df, vocab_df, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft,
                                                  hop_length=args.hop_length, patch_stride_ms=args.patch_stride_ms,
                                                  num_temporal_patches=args.num_temporal_patches, augment=args.augment,
                                                  use_mixup=args.use_mixup, use_color_noise=args.use_color_noise,
                                                  use_tf_mask=args.use_tf_mask, mixup_beta=args.mixup_beta,
                                                  mixup_chance=args.mixup_chance, time_frame_size=args.time_frame_size,
                                                  mixup_hard_label=args.mixup_hard_label,
                                                  color_noise_chance=args.color_noise_chance,
                                                  freq_mask_param=args.freq_mask_param,
                                                  time_mask_param=args.time_mask_param)

            val_dataset = DCASE2016Task2Dataset(val_df, vocab_df, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft,
                                                hop_length=args.hop_length, patch_stride_ms=args.patch_stride_ms,
                                                num_temporal_patches=args.num_temporal_patches,
                                                time_frame_size=args.time_frame_size)

    elif args.task == 'librispeech':
        train_ds, val_ds, test_ds, tokenizer = prepare_librispeech(cache_dir=args.dataset_dir)
        args.num_classes = tokenizer.vocab_size

        if is_legacy:
            train_dataset = LegacyLibriSpeechDataset(train_ds, model_name=args.model_name, sr=args.sr,
                                                     n_mels=args.n_mels, frame_shift=args.frame_shift,
                                                     frame_length=args.frame_length, augment=args.augment,
                                                     tf_mask_repeats=args.tf_mask_repeats,
                                                     use_color_noise=args.use_color_noise, use_tf_mask=args.use_tf_mask,
                                                     color_noise_chance=args.color_noise_chance,
                                                     freq_mask_param=args.freq_mask_param,
                                                     time_mask_param=args.time_mask_param)

            val_dataset = LegacyLibriSpeechDataset(val_ds, model_name=args.model_name, sr=args.sr, n_mels=args.n_mels,
                                                   frame_shift=args.frame_shift, frame_length=args.frame_length)

            test_dataset = LegacyLibriSpeechDataset(test_ds, model_name=args.model_name, sr=args.sr, n_mels=args.n_mels,
                                                    frame_shift=args.frame_shift, frame_length=args.frame_length)
        else:
            train_dataset = LibriSpeechDataset(train_ds, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft,
                                               hop_length=args.hop_length, augment=args.augment,
                                               use_color_noise=args.use_color_noise, use_tf_mask=args.use_tf_mask,
                                               color_noise_chance=args.color_noise_chance,
                                               tf_mask_repeats=args.tf_mask_repeats,
                                               freq_mask_param=args.freq_mask_param,
                                               time_mask_param=args.time_mask_param)

            val_dataset = LibriSpeechDataset(val_ds, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft,
                                             hop_length=args.hop_length)

            test_dataset = LibriSpeechDataset(test_ds, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft,
                                              hop_length=args.hop_length)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=args.train_num_workers, collate_fn=train_dataset.collate_fn,
                              persistent_workers=True if args.train_num_workers > 0 else False)

    if args.task in ['scv2', 'dcase2016_task2', 'librispeech']:
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.val_num_workers, collate_fn=val_dataset.collate_fn,
                                persistent_workers=True if args.val_num_workers > 0 else False)

    if args.task != 'dcase2016_task2':
        test_loader = DataLoader(test_dataset, batch_size=1 if args.task == 'librispeech' else args.test_batch_size,
                                 shuffle=False, num_workers=args.test_num_workers,
                                 collate_fn=test_dataset.collate_fn,
                                 persistent_workers=True if args.test_num_workers > 0 else False)

    # Data Transforms
    if args.task in ['dcase2016_task2', 'librispeech']:
        data_transforms = None

    elif is_legacy:
        data_transforms = LegacyDataTransforms(num_time_patches=args.num_time_patches,
                                               num_freq_patches=args.num_freq_patches, use_mixup=args.use_mixup,
                                               mixup_beta=args.mixup_beta, mixup_chance=args.mixup_chance,
                                               mixup_hard_label=args.mixup_hard_label, use_tf_mask=args.use_tf_mask,
                                               freq_mask_param=args.freq_mask_param,
                                               time_mask_param=args.time_mask_param,
                                               tf_mask_repeats=args.tf_mask_repeats).to(args.device)
    else:
        data_transforms = DataTransforms(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels,
                                         f_min=args.f_min, time_frame_size=args.time_frame_size,
                                         use_time_roll=args.use_time_roll, num_time_patches=args.num_time_patches,
                                         num_freq_patches=args.num_freq_patches, use_mixup=args.use_mixup,
                                         mixup_chance=args.mixup_chance, mixup_beta=args.mixup_beta,
                                         mixup_hard_label=args.mixup_hard_label, use_color_noise=args.use_color_noise,
                                         color_noise_chance=args.color_noise_chance, use_tf_mask=args.use_tf_mask,
                                         freq_mask_param=args.freq_mask_param, time_mask_param=args.time_mask_param,
                                         tf_mask_repeats=args.tf_mask_repeats).to(args.device)

    return train_loader, val_loader, test_loader, data_transforms, tokenizer, vocab_df
