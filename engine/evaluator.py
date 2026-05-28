__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.signal import medfilt
from sklearn.metrics import average_precision_score, f1_score
import librosa
import jiwer
import sed_eval
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from data import load_dcase2016_task2_json_to_df, CharTokenizer


@torch.inference_mode()
def evaluate_multilabel(model: torch.nn.Module,
                        data_loader: DataLoader,
                        data_transforms: torch.nn.Module,
                        device: str,
                        amp_dtype: Optional[torch.dtype] = None) -> Dict[str, float]:
    """Evaluates the model on a multi-label classification dataset."""
    targets, predicts = [], []
    for x, y in data_loader:
        x, *_ = data_transforms(x.to(device), None, mask_ratio=0, augment=False)
        with torch.autocast(device_type=device, dtype=amp_dtype):
            logits = model(x, mask_info=None)
        targets.append(y)
        predicts.append(logits.float().sigmoid().cpu())
    targets = torch.cat(targets, dim=0).numpy()
    predicts = torch.cat(predicts, dim=0)
    return {'f1_macro': f1_score(targets, predicts.round().numpy(), average='macro', zero_division=0) * 100,
            'mAP_macro': average_precision_score(targets, predicts.numpy(), average='macro') * 100}


@torch.inference_mode()
def evaluate_multiclass(model: torch.nn.Module, data_loader: DataLoader, data_transforms: torch.nn.Module, device: str,
                        amp_dtype: Optional[torch.dtype] = None) -> Dict[str, float]:
    """Evaluates the model on a multi-class classification dataset."""
    targets, predicts = [], []
    for x, y in data_loader:
        assert y.ndim == 1, y.shape
        x, *_ = data_transforms(x.to(device), None, mask_ratio=0, augment=False)
        with torch.autocast(device_type=device, dtype=amp_dtype):
            logits = model(x, mask_info=None)
        targets.append(y.long())
        predicts.append(logits.float().softmax(dim=1).cpu())
    targets = torch.cat(targets, dim=0)
    predicts_soft = torch.cat(predicts, dim=0)
    predicts_hard = predicts_soft.argmax(dim=1).long()
    return {'accuracy': (targets == predicts_hard).float().mean().item() * 100,
            'mAP_macro': average_precision_score(targets.numpy(), predicts_soft.numpy(), average='macro') * 100}


def greedy_decode(logits: torch.Tensor, tokenizer: CharTokenizer) -> List[str]:
    """Performs greedy decoding on the output logits using the provided tokenizer."""
    preds = torch.argmax(logits, dim=-1)
    decoded_texts = []
    for seq in preds:
        collapsed = [token.item() for i, token in enumerate(seq) if i == 0 or token != seq[i - 1]]
        decoded_texts.append(tokenizer.decode(collapsed, ignore_blanks=True))
    return decoded_texts


@torch.inference_mode()
def evaluate_asr(model: torch.nn.Module,
                 data_loader: DataLoader,
                 tokenizer: CharTokenizer,
                 device: str = 'cuda',
                 amp_dtype: Optional[torch.dtype] = None) -> Dict[str, float]:
    """Evaluates the model on an Automatic Speech Recognition (ASR) task using WER and CER metrics."""
    model.eval()
    preds, refs = [], []
    for x, y, input_length, target_length in data_loader:
        with torch.autocast(device_type=device, dtype=amp_dtype):
            log_probs = model(x.to(device)).transpose(0, 1)  # T, B, C -> B, T, C
        preds.extend(greedy_decode(log_probs, tokenizer))
        refs.extend([tokenizer.decode(y[i][:target_length[i]].tolist(), ignore_blanks=True) for i in range(y.shape[0])])
    return {'wer': jiwer.wer(refs, preds) * 100, 'cer': jiwer.cer(refs, preds) * 100}


@torch.inference_mode()
def evaluate_dcase_val(model: torch.nn.Module,
                       data_loader: DataLoader,
                       device: str,
                       amp_dtype: Optional[torch.dtype] = None) -> Dict[str, float]:
    """Evaluates the model on the DCASE validation set using frame-wise temporal predictions."""
    targets, predicts = [], []
    for x, y in data_loader:
        batch_size, num_time_patches, num_classes = y.shape
        with torch.autocast(device_type=device, dtype=amp_dtype):
            logits = model(x.to(device))
        predicts.append(logits.float().sigmoid().reshape(-1, num_classes).cpu())
        targets.append(y.reshape(-1, num_classes))
    predicts = torch.cat(predicts)
    targets = torch.cat(targets).numpy()
    return {'f1_micro': f1_score(targets, predicts.round().numpy(), average='micro', zero_division=0) * 100,
            'mAP_micro': average_precision_score(targets, predicts.numpy(), average='micro') * 100}


def tokens_to_events(predictions: np.ndarray,
                     patch_stride_ms: float = 160.0,
                     threshold: float = 0.3,
                     filter_size: int = 3) -> List[Tuple[float, float]]:
    """Converts frame-wise probability predictions into discrete event timestamps (onset/offset)."""
    if filter_size > 1:
        predictions = medfilt(predictions, kernel_size=filter_size)
    binary_preds = (predictions >= threshold).astype(int)
    padded = np.pad(binary_preds, (1, 1), 'constant')
    diffs = np.diff(padded)
    onsets = np.where(diffs == 1)[0]
    offsets = np.where(diffs == -1)[0]

    num_tokens = len(predictions)
    patch_centers_ms = np.arange(num_tokens) * patch_stride_ms
    events = []
    for on, off in zip(onsets, offsets):
        off_idx = off - 1
        start_time_ms = max(0.0, patch_centers_ms[on])
        end_time_ms = patch_centers_ms[off_idx]
        events.append((start_time_ms / 1000., end_time_ms / 1000.))
    return events


def eval_test_set_dcase(data_dir: Union[str, Path],
                        vocab_df: pd.DataFrame,
                        model: torch.nn.Module,
                        device: str = 'cuda',
                        sr: int = 16000,
                        n_mels: int = 128,
                        n_fft: int = 1024,
                        hop_length: int = 160,
                        num_temporal_patches: int = 750,
                        patch_stride_ms: float = 160,
                        threshold: float = 0.3,
                        filter_size: int = 3,
                        amp_dtype: Optional[torch.dtype] = None,
                        legacy: bool = False,
                        model_name: Optional[str] = None) -> Dict[str, float]:
    """Fully evaluates the DCASE test set, including calculating standard Sound Event Detection (SED) metrics."""
    test_df = load_dcase2016_task2_json_to_df(root_path=data_dir, split='test')
    grouped_events = test_df.groupby('filename')
    filenames = list(grouped_events.groups.keys())
    num_classes = len(vocab_df)
    label_to_idx = dict(zip(vocab_df['label'], vocab_df['idx']))
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    patch_centers = torch.arange(num_temporal_patches, dtype=torch.float32) * patch_stride_ms
    sed_evaluator = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=[idx_to_label[i] for i in range(num_classes)],
        t_collar=0.200, evaluate_onset=True, evaluate_offset=False)
    model.eval()

    fbank, mel, db, minmax_normalize = None, None, None, None
    mu, std = 0.0, 1.0
    if legacy:
        if model_name in ['EAT', 'SSLAM']:
            fbank = partial(torchaudio.compliance.kaldi.fbank, htk_compat=True, sample_frequency=sr, dither=0.0,
                            window_type='hanning', num_mel_bins=n_mels, use_energy=False, frame_shift=10.0,
                            frame_length=25.0)
            mu, std = -10.685920220841732, 2.061649722538243
        else:
            fbank = partial(torchaudio.compliance.kaldi.fbank, htk_compat=False, sample_frequency=sr, dither=0.0,
                            window_type='povey', num_mel_bins=n_mels, use_energy=False, frame_shift=10.0,
                            frame_length=25.0)
            mu, std = 10.01597327655935, 3.0706585792274494
    else:
        mel = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        db = AmplitudeToDB('power', top_db=80)
        minmax_normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)

    all_yt = torch.zeros((len(filenames), num_temporal_patches, num_classes), dtype=torch.float32)
    all_yp = torch.zeros((len(filenames), num_temporal_patches, num_classes), dtype=torch.float32)

    for i, f in enumerate(filenames):
        events = grouped_events.get_group(f)
        ref_events, est_events = [], []
        for _, row in events.iterrows():
            class_idx = label_to_idx[row['label']]
            start_ms, end_ms = row['start_ms'], row['end_ms']
            ref_events.append(
                {'event_label': row['label'], 'event_onset': start_ms / 1000, 'event_offset': end_ms / 1000})
            active_frames = (patch_centers >= start_ms) & (patch_centers <= end_ms)
            all_yt[i, active_frames, class_idx] = 1.0

        x, _ = librosa.load(path=f, sr=sr, res_type='soxr_hq')
        x = torch.from_numpy(x).float()

        if legacy:
            if model_name != 'BEATs': x -= x.mean()
            x = x.unsqueeze(0)
            if model_name == 'BEATs': x = x * (2 ** 15)
            x = fbank(x).unsqueeze(0)
            x = (x - mu) / (std * 2)
            x = F.pad(x, (0, 0, 0, 12000 - x.shape[1]))
        else:
            x = minmax_normalize(db(mel(x)))
            x = x.T.unsqueeze(0)

        x = x.unsqueeze(0).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=amp_dtype):
                y_pred = model(x).float().sigmoid().cpu().squeeze(0)

        all_yp[i] = y_pred.clone()
        y_pred = y_pred.numpy().T

        for c in range(num_classes):
            y_event = tokens_to_events(y_pred[c], patch_stride_ms, threshold, filter_size)
            for start, end in y_event:
                est_events.append(
                    {'event_label': idx_to_label[c], 'event_onset': start.item(), 'event_offset': end.item()})
        sed_evaluator.evaluate(reference_event_list=ref_events, estimated_event_list=est_events)

    cache = {}
    all_yt = all_yt.reshape(-1, num_classes).numpy()
    all_yp = all_yp.reshape(-1, num_classes)
    results = sed_evaluator.results()

    cache['onset_f1_micro'] = results['overall']['f_measure']['f_measure'] * 100.0
    cache['onset_precision_micro'] = results['overall']['f_measure']['precision'] * 100.0
    cache['onset_recall_micro'] = results['overall']['f_measure']['recall'] * 100.0
    cache['f1_macro'] = f1_score(all_yt, all_yp.round().numpy(), average='macro', zero_division=0) * 100
    cache['mAP_macro'] = average_precision_score(all_yt, all_yp.numpy(), average='macro') * 100
    cache['f1_micro'] = f1_score(all_yt, all_yp.round().numpy(), average='micro', zero_division=0) * 100
    cache['mAP_micro'] = average_precision_score(all_yt, all_yp.numpy(), average='micro') * 100

    return cache
