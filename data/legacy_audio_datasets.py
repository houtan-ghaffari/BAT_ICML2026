__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from functools import partial
from pathlib import Path
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from torchaudio.compliance.kaldi import fbank as kaldi_fbank
from datasets import Dataset as HuggingFaceDataset
from .augmentations import apply_color_noise, apply_mixup
from typing import Tuple, List, Dict, Optional

LEGACY_STATS = {
    'EAT': {
        'AudioSet': {'mu': -4.59864076, 'std': 4.357353477},
        'SpeechCommands': {'mu': -9.03415996, 'std': 4.57304508},
        'ESC50': {'mu': -6.85408831, 'std': 5.39770323},
        'LibriSpeech': {'mu': -7.32925959, 'std': 4.06100490},
        'BirdSet': {'mu': -8.55048120, 'std': 4.04533479},
        'DCASE': {'mu': -10.68592022, 'std': 2.06164972},
        'htk_compat': True,
        'window_type': 'hanning',
    },
    'SSLAM': {
        'AudioSet': {'mu': -4.59864076, 'std': 4.357353477},
        'SpeechCommands': {'mu': -9.03415996, 'std': 4.57304508},
        'ESC50': {'mu': -6.85408831, 'std': 5.39770323},
        'LibriSpeech': {'mu': -7.32925959, 'std': 4.06100490},
        'BirdSet': {'mu': -8.55048120, 'std': 4.04533479},
        'DCASE': {'mu': -10.68592022, 'std': 2.06164972},
        'htk_compat': True,
        'window_type': 'hanning',
    },
    'BEATs': {
        'AudioSet': {'mu': 15.41663, 'std': 6.55582},
        'SpeechCommands': {'mu': 11.42262215, 'std': 5.65233194},
        'ESC50': {'mu': 11.72161352, 'std': 10.60385933},
        'LibriSpeech': {'mu': 13.30096596, 'std': 4.97074609},
        'BirdSet': {'mu': 11.82694931, 'std': 5.40055698},
        'DCASE': {'mu': 10.01597327, 'std': 3.07065857},
        'htk_compat': False,
        'window_type': 'povey',
    }
}


class LegacyAudioSet(Dataset):
    """Legacy AudioSet dataset utilizing Kaldi filterbanks and pre-computed statistics for backwards compatibility."""

    def __init__(self,
                 df: pd.DataFrame,
                 num_classes: int = 527,
                 sr: int = 16000,
                 augment: bool = False,
                 use_time_roll: bool = False,
                 use_color_noise: bool = False,
                 color_noise_chance: float = 0.3,
                 time_frame_size: int = 1024,
                 n_mels: int = 128,
                 frame_shift: float = 10.0,
                 frame_length: float = 25.0,
                 sample_dur: float = 10.23,
                 train: bool = False,
                 model_name: str = 'None'):

        super().__init__()
        assert model_name in ['EAT', 'SSLAM', 'BEATs']
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]

        self.num_samples = int(sample_dur * sr)
        self.mu = LEGACY_STATS[model_name]['AudioSet']['mu']
        self.std = LEGACY_STATS[model_name]['AudioSet']['std']

        window_type = LEGACY_STATS[model_name]['window_type']
        htk_compat = LEGACY_STATS[model_name]['htk_compat']

        self.fbank = partial(kaldi_fbank, htk_compat=htk_compat, sample_frequency=self.sr,
                             window_type=window_type, num_mel_bins=self.n_mels,
                             frame_shift=frame_shift, frame_length=frame_length,
                             dither=0.0, use_energy=False)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.df.iloc[idx].file_path.decode('UTF-8')
        x, sr = librosa.load(path=f, sr=self.sr, res_type="soxr_hq")

        while x.shape[0] == 0:
            idx = random.choice(range(len(self.df)))
            f = self.df.iloc[idx].file_path.decode('UTF-8')
            x, sr = librosa.load(path=f, sr=self.sr, res_type="soxr_hq")

        x = torch.from_numpy(x).float()

        codes = self.df.iloc[idx].codes.decode('UTF-8').split(',')
        y = torch.tensor([int(c) for c in codes])
        y = F.one_hot(y, num_classes=self.num_classes).amax(dim=0).float()

        if self.model_name != 'BEATs':
            x -= x.mean()

        if self.train:
            x = torch.roll(x, random.randrange(len(x)))

        num_samples = x.shape[0]
        if num_samples < self.num_samples:
            d = int(self.num_samples - num_samples)
            x = F.pad(x, (0, d))
            x = torch.roll(x, random.randint(0, d)) if self.train else x
        elif num_samples > self.num_samples:
            offset = int(random.random() * max(0, num_samples - self.num_samples)) if self.train else 0
            x = x[offset:offset + self.num_samples]

        assert x.shape[0] == self.num_samples, f"idx:{idx}, shape:{x.shape}, sr:{sr}"

        x = x.unsqueeze(0)  # 1, N

        if self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        if self.model_name == 'BEATs':
            x = x * (2 ** 15)  # 1, N

        x = self.fbank(x)  # T, F
        x = (x - self.mu) / (self.std * 2)
        x = F.pad(x, (0, 0, 0, self.time_frame_size - x.shape[0]))

        assert x.ndim == 2 and x.shape[0] == self.time_frame_size and x.shape[1] == self.n_mels, x.shape
        return x, y

    def __len__(self) -> int:
        return self.df.shape[0]

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stacks features and labels, transposing features to (B, 1, F, T)."""
        x, y = zip(*xy)
        x = torch.stack(x).float().unsqueeze(1)  # B, 1, T, F
        x = x.transpose(-2, -1)  # B, 1, F, T
        y = torch.stack(y).float()  # B, C
        return x, y


class LegacySpeechCommands(Dataset):
    """Legacy Speech Commands dataset utilizing Kaldi filterbanks and pre-computed statistics."""

    def __init__(self,
                 df: pd.DataFrame,
                 sr: int = 16000,
                 n_mels: int = 128,
                 frame_shift: float = 10.0,
                 frame_length: float = 25.0,
                 augment: bool = False,
                 use_random_gain: bool = False,
                 use_time_roll: bool = False,
                 use_color_noise: bool = False,
                 color_noise_chance: float = 0.3,
                 time_frame_size: int = 112,
                 one_hot_labels: bool = False,
                 model_name: str = 'None'):

        super().__init__()
        assert model_name in ['EAT', 'SSLAM', 'BEATs']
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]

        self.num_classes = 12
        self.mu = LEGACY_STATS[model_name]['SpeechCommands']['mu']
        self.std = LEGACY_STATS[model_name]['SpeechCommands']['std']

        window_type = LEGACY_STATS[model_name]['window_type']
        htk_compat = LEGACY_STATS[model_name]['htk_compat']

        self.fbank = partial(kaldi_fbank, htk_compat=htk_compat, sample_frequency=self.sr,
                             window_type=window_type, num_mel_bins=self.n_mels,
                             frame_shift=frame_shift, frame_length=frame_length,
                             dither=0.0, use_energy=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.df.iloc[idx]

        if self.one_hot_labels:
            y = F.one_hot(torch.tensor(entry['label']), num_classes=self.num_classes).float()
            assert y.ndim == 1 and y.shape[0] == self.num_classes, y.shape
        else:
            y = torch.tensor(entry['label'], dtype=torch.long)
            assert y.ndim == 0, y.shape

        dur = None if pd.isna(entry['duration']) else entry['duration']
        x, _ = librosa.load(entry['filepath'].decode('utf-8'), sr=self.sr, offset=entry['offset'], duration=dur, res_type="soxr_vhq")
        x = torch.from_numpy(x).float()

        if self.model_name != 'BEATs':
            x -= x.mean()

        x = F.pad(x, (0, self.sr - x.shape[0]))  # 1 second

        x = x.numpy()

        if self.augment and self.use_time_roll:
            i = random.randrange(len(x))
            x = np.roll(x, i)

        if self.augment and self.use_random_gain:
            x = x * (random.betavariate(10, 10) + 0.5)

        x = torch.from_numpy(x).float()
        x = x.unsqueeze(0)  # 1, N

        if self.augment and self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        if self.model_name == 'BEATs':
            x = x * (2 ** 15)  # 1, N

        x = self.fbank(x)  # T, F
        x = (x - self.mu) / (self.std * 2)
        x = F.pad(x, (0, 0, 0, self.time_frame_size - x.shape[0])).unsqueeze(0)

        assert x.ndim == 3 and x.shape[1] == self.time_frame_size and x.shape[2] == self.n_mels, x.shape

        return x, y

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stacks features and labels, transposing features to (B, 1, F, T)."""
        x, y = zip(*xy)
        x = torch.stack(x).float()  # B, 1, T, F
        x = x.transpose(-2, -1)  # B, 1, F, T
        y = torch.stack(y)  # B, [C]
        return x, y


class LegacyESC50Dataset(Dataset):
    """Legacy ESC-50 dataset utilizing Kaldi filterbanks and pre-computed statistics."""

    def __init__(self, df: pd.DataFrame,
                 sr: int = 16000,
                 n_mels: int = 128,
                 frame_shift: float = 10.0,
                 frame_length: float = 25.0,
                 augment: bool = False,
                 use_random_gain: bool = False,
                 use_time_roll: bool = False,
                 use_color_noise: bool = False,
                 color_noise_chance: float = 0.3,
                 time_frame_size: int = 512,
                 one_hot_labels: bool = False,
                 model_name: str = 'None'):

        super().__init__()
        assert model_name in ['EAT', 'SSLAM', 'BEATs']
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]

        self.num_classes = 50
        self.mu = LEGACY_STATS[model_name]['ESC50']['mu']
        self.std = LEGACY_STATS[model_name]['ESC50']['std']

        window_type = LEGACY_STATS[model_name]['window_type']
        htk_compat = LEGACY_STATS[model_name]['htk_compat']

        self.fbank = partial(kaldi_fbank, htk_compat=htk_compat, sample_frequency=self.sr,
                             window_type=window_type, num_mel_bins=self.n_mels,
                             frame_shift=frame_shift, frame_length=frame_length,
                             dither=0.0, use_energy=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.df.iloc[idx]

        if self.one_hot_labels:
            y = F.one_hot(torch.tensor(entry['target']), num_classes=self.num_classes).float()
            assert y.ndim == 1 and y.shape[0] == self.num_classes, y.shape
        else:
            y = torch.tensor(entry['target'], dtype=torch.long)
            assert y.ndim == 0, y.shape

        x, _ = librosa.load(path=entry['filename'], sr=self.sr, res_type="soxr_vhq")
        x = torch.from_numpy(x).float()

        if self.model_name != 'BEATs':
            x -= x.mean()

        x = F.pad(x, (0, (self.sr * 5) - x.shape[0]))  # 5 seconds
        x = x.numpy()

        if self.use_time_roll:
            i = random.randrange(len(x))
            x = np.roll(x, i)

        if self.use_random_gain:
            x = x * (random.betavariate(10, 10) + 0.5)

        x = torch.from_numpy(x).float()
        x = x.unsqueeze(0)  # 1, N

        if self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        if self.model_name == 'BEATs':
            x = x * (2 ** 15)  # 1, N

        x = self.fbank(x)  # T, F
        x = (x - self.mu) / (self.std * 2)
        x = F.pad(x, (0, 0, 0, self.time_frame_size - x.shape[0])).unsqueeze(0)

        assert x.shape[0] == 1 and x.shape[1] == self.time_frame_size and x.shape[2] == self.n_mels, x.shape
        return x, y

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stacks features and labels, transposing features to (B, 1, F, T)."""
        x, y = zip(*xy)
        x = torch.stack(x).float()  # B, 1, T, F
        x = x.transpose(-2, -1)  # B, 1, F, T
        y = torch.stack(y)  # B, [C]
        return x, y


class LegacyLibriSpeechDataset(Dataset):
    """Legacy LibriSpeech dataset utilizing Kaldi filterbanks and pre-computed statistics."""

    def __init__(self, ds: HuggingFaceDataset,
                 sr: int = 16000,
                 augment: bool = False,
                 use_color_noise: bool = False,
                 color_noise_chance: float = 0.3,
                 n_mels: int = 128,
                 frame_shift: float = 10.0,
                 frame_length: float = 25.0,
                 vit_patch_stride: int = 16,
                 upsample_factor: int = 8,
                 tf_mask_repeats: int = 1,
                 use_tf_mask: bool = False,
                 freq_mask_param: Optional[int] = None,
                 time_mask_param: Optional[int] = None,
                 model_name: str = 'None'):

        super().__init__()
        assert model_name in ['EAT', 'SSLAM', 'BEATs']
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]

        self.mu = LEGACY_STATS[model_name]['LibriSpeech']['mu']
        self.std = LEGACY_STATS[model_name]['LibriSpeech']['std']

        window_type = LEGACY_STATS[model_name]['window_type']
        htk_compat = LEGACY_STATS[model_name]['htk_compat']

        self.fbank = partial(kaldi_fbank, htk_compat=htk_compat, sample_frequency=self.sr,
                             window_type=window_type, num_mel_bins=self.n_mels,
                             frame_shift=frame_shift, frame_length=frame_length,
                             dither=0.0, use_energy=False)

        if use_tf_mask:
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        entry = self.ds[idx]
        y = torch.tensor(entry["labels"], dtype=torch.long)
        x = torch.tensor(entry['audio']['array'], dtype=torch.float32)
        assert x.ndim == 1

        if self.model_name != 'BEATs':
            x -= x.mean()

        x = x.unsqueeze(0)  # 1, N

        if self.augment and self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        if self.model_name == 'BEATs':
            x = x * (2 ** 15)  # 1, N

        x = self.fbank(x)  # T, F
        x = (x - self.mu) / (self.std * 2)

        if x.shape[0] % self.vit_patch_stride > 0:
            x = F.pad(x, (0, 0, 0, self.vit_patch_stride - x.shape[0] % self.vit_patch_stride))

        input_length = (x.shape[0] // self.vit_patch_stride) * self.upsample_factor
        target_length = y.shape[0]

        assert x.ndim == 2 and x.shape[1] == self.n_mels, x.shape
        return x, y, input_length, target_length

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads and batches sequential data, optionally applying time-frequency masking."""
        x, y, input_length, target_length = zip(*batch)
        y = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)  # B, S
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)  # B, T, F
        x = x.unsqueeze(1).float()  # B, 1, T, F

        if self.augment and self.use_tf_mask:
            x = x.transpose(2, 3)  # B, 1, F, T
            for _ in range(self.tf_mask_repeats):
                x = self.freq_masking(x)
                x = self.time_masking(x)
            x = x.transpose(2, 3)  # B, 1, T, F

        # original lengths for CTC Loss
        input_length = torch.tensor(input_length, dtype=torch.long)
        target_length = torch.tensor(target_length, dtype=torch.long)
        return x, y, input_length, target_length


class LegacyBirdSet(Dataset):
    """Legacy BirdSet dataset utilizing Kaldi filterbanks and pre-computed statistics."""

    def __init__(self, df: pd.DataFrame,
                 sr: int = 16000,
                 duration_seconds: int = 5,
                 num_classes: int = 21,
                 test: bool = False,
                 frame_shift: float = 10.0,
                 frame_length: float = 25.0,
                 n_mels: int = 128,
                 augment: bool = False,
                 use_random_gain: bool = False,
                 use_time_roll: bool = False,
                 use_color_noise: bool = False,
                 color_noise_chance: float = 0.3,
                 time_frame_size: int = 512,
                 model_name: str = 'None'):

        super().__init__()
        assert model_name in ['EAT', 'SSLAM', 'BEATs']
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]

        self.num_samples = int(duration_seconds * sr)
        self.mu = LEGACY_STATS[model_name]['BirdSet']['mu']
        self.std = LEGACY_STATS[model_name]['BirdSet']['std']

        window_type = LEGACY_STATS[model_name]['window_type']
        htk_compat = LEGACY_STATS[model_name]['htk_compat']

        self.fbank = partial(kaldi_fbank, htk_compat=htk_compat, sample_frequency=self.sr,
                             window_type=window_type, num_mel_bins=self.n_mels,
                             frame_shift=frame_shift, frame_length=frame_length,
                             dither=0.0, use_energy=False)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.test:
            return self.get_test(idx)
        return self.get_train(idx)

    def __len__(self) -> int:
        return len(self.df)

    def get_train(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads, augments, and processes training samples."""
        entry = self.df.iloc[idx]
        path = Path(entry['filepath'].decode('UTF-8'))
        audio_len = entry['length']

        y = entry['ebird_code_multilabel']
        assert isinstance(y, np.ndarray), type(y)
        y = torch.tensor(y)
        assert y.ndim == 1, y.shape
        y = F.one_hot(y, num_classes=self.num_classes).amax(dim=0)
        assert y.ndim == 1, y.shape
        assert len(y) == self.num_classes, y.shape

        if audio_len < self.duration_seconds:
            x, _ = librosa.load(path=path, sr=self.sr, res_type='soxr_hq')
            x = torch.from_numpy(x).float()
            d = int(self.num_samples - x.shape[0])
            x = F.pad(x, (0, d))
            x = torch.roll(x, random.randint(0, d))
        else:
            offset = random.random() * max(0, audio_len - self.duration_seconds)
            x, _ = librosa.load(path=path, sr=self.sr, offset=offset, duration=self.duration_seconds,
                                res_type='soxr_hq')
            x = torch.from_numpy(x).float()

        assert x.shape[0] == self.num_samples
        x += torch.randn_like(x) * 0.02

        if self.model_name != 'BEATs':
            x -= x.mean()

        x = x.numpy()

        if self.augment and self.use_time_roll:
            i = random.randrange(len(x))
            x = np.roll(x, i)

        if self.augment and self.use_random_gain:
            x = x * (random.betavariate(10, 10) + 0.5)

        x = torch.from_numpy(x).float()
        x = x.unsqueeze(0)  # 1, N

        if self.augment and self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        if self.model_name == 'BEATs':
            x = x * (2 ** 15)  # 1, N

        x = self.fbank(x)  # T, F
        x = (x - self.mu) / (self.std * 2)
        x = F.pad(x, (0, 0, 0, self.time_frame_size - x.shape[0])).unsqueeze(0)

        assert x.ndim == 3 and x.shape[1] == self.time_frame_size and x.shape[2] == self.n_mels, x.shape
        return x, y

    def get_test(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and processes fixed test samples."""
        entry = self.df.iloc[idx]
        path = Path(entry['filepath'].decode('UTF-8'))
        y = entry['ebird_code_multilabel']

        if len(y) == 0:
            y = torch.zeros(self.num_classes)
        else:
            assert isinstance(y, np.ndarray), type(y)
            y = torch.tensor(y)
            assert y.ndim == 1, y.shape
            y = F.one_hot(y, num_classes=self.num_classes).amax(dim=0)
            assert y.ndim == 1, y.shape

        assert len(y) == self.num_classes, y.shape
        x, _ = librosa.load(path=path, sr=self.sr, res_type='soxr_hq')
        x = torch.from_numpy(x).float()

        if self.model_name != 'BEATs':
            x -= x.mean()

        x = x.unsqueeze(0)  # 1, N

        if self.model_name == 'BEATs':
            x = x * (2 ** 15)  # 1, N

        x = self.fbank(x)  # T, F
        x = (x - self.mu) / (self.std * 2)
        x = F.pad(x, (0, 0, 0, self.time_frame_size - x.shape[0])).unsqueeze(0)  # 1, T, F

        assert x.ndim == 3 and x.shape[1] == self.time_frame_size and x.shape[2] == self.n_mels, x.shape
        return x, y

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stacks features and labels, transposing features to (B, 1, F, T)."""
        x, y = zip(*xy)
        x = torch.stack(x).float()  # B, 1, T, F
        x = x.transpose(-2, -1)  # B, 1, F, T
        y = torch.stack(y).float()  # B, C
        return x, y


class LegacyDCASE2016Task2Dataset(Dataset):
    """Legacy DCASE dataset utilizing Kaldi filterbanks and pre-computed statistics."""

    def __init__(self,
                 df: pd.DataFrame,
                 vocab_df: pd.DataFrame,
                 sr: int = 16_000,
                 n_mels: int = 128,
                 frame_shift: float = 10.0,
                 frame_length: float = 25.0,
                 patch_stride_ms: int = 160,
                 num_temporal_patches: int = 750,
                 augment: bool = False,
                 use_mixup: bool = False,
                 use_color_noise: bool = False,
                 use_tf_mask: bool = False,
                 mixup_beta: float = 0.8,
                 mixup_chance: float = 0.9,
                 mixup_hard_label: bool = False,
                 color_noise_chance: Optional[float] = None,
                 tf_mask_repeats: int = 1,
                 freq_mask_param: Optional[int] = None,
                 time_mask_param: Optional[int] = None,
                 time_frame_size: int = 12000,
                 model_name: str = 'None'):

        super().__init__()
        assert model_name in ['EAT', 'SSLAM', 'BEATs']
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]

        self.grouped_events = df.groupby('filename')
        self.filenames = list(self.grouped_events.groups.keys())
        self.num_classes = len(vocab_df)
        self.label_to_idx = dict(zip(vocab_df['label'], vocab_df['idx']))
        self.patch_centers = torch.arange(num_temporal_patches, dtype=torch.float32) * patch_stride_ms

        self.mu = LEGACY_STATS[model_name]['DCASE']['mu']
        self.std = LEGACY_STATS[model_name]['DCASE']['std']

        window_type = LEGACY_STATS[model_name]['window_type']
        htk_compat = LEGACY_STATS[model_name]['htk_compat']

        self.fbank = partial(kaldi_fbank, htk_compat=htk_compat, sample_frequency=self.sr,
                             window_type=window_type, num_mel_bins=self.n_mels,
                             frame_shift=frame_shift, frame_length=frame_length,
                             dither=0.0, use_energy=False)

        if use_tf_mask:
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.filenames[idx]
        events = self.grouped_events.get_group(f)

        x, sr = librosa.load(path=f, sr=self.sr, res_type='soxr_hq')
        x = torch.from_numpy(x).float()
        assert x.shape[0] > 0, f"{f} is empty!"

        if self.model_name != 'BEATs':
            x -= x.mean()

        x = x.unsqueeze(0)  # 1, N

        if self.augment and self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        if self.model_name == 'BEATs':
            x = x * (2 ** 15)  # 1, N

        x = self.fbank(x).unsqueeze(0)  # 1, T, F
        x = (x - self.mu) / (self.std * 2)

        x = F.pad(x, (0, 0, 0, self.time_frame_size - x.shape[1]))
        assert x.shape[0] == 1 and x.shape[1] == self.time_frame_size and x.shape[2] == self.n_mels, x.shape

        y = torch.zeros((self.num_temporal_patches, self.num_classes), dtype=torch.float32)
        for _, row in events.iterrows():
            class_idx = self.label_to_idx[row['label']]
            start_ms = row['start_ms']
            end_ms = row['end_ms']
            # find which ViT's temporal-patches fall inside this event's time boundaries
            active_frames = (self.patch_centers >= start_ms) & (self.patch_centers <= end_ms)
            y[active_frames, class_idx] = 1.0

        return x, y

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stacks features and labels, applying mixup and tf_masking if enabled."""
        x, y = zip(*xy)
        y = torch.stack(y)  # B, num_time_patches, C
        x = torch.stack(x)  # B, 1, T, F

        if self.augment and self.use_mixup:
            x, y = apply_mixup(x, y, self.mixup_beta, self.mixup_chance, self.mixup_hard_label)

        if self.augment and self.use_tf_mask:
            x = x.transpose(2, 3)  # B, 1, F, T
            for _ in range(self.tf_mask_repeats):
                x = self.freq_masking(x)
                x = self.time_masking(x)
            x = x.transpose(2, 3)  # B, 1, T, F

        return x, y


class LegacyDataTransforms(nn.Module):
    """Legacy feature-level transformations applied to pre-computed spectrograms."""

    def __init__(self,
                 num_time_patches: Optional[int] = None,
                 num_freq_patches: Optional[int] = None,
                 use_mixup: bool = False,
                 use_time_roll: bool = False,
                 mixup_chance: float = 0.9,
                 mixup_beta: float = .8,
                 mixup_hard_label: bool = False,
                 use_tf_mask: bool = False,
                 freq_mask_param: int = 16,
                 time_mask_param: int = 64,
                 tf_mask_repeats: int = 1,
                 ):

        super().__init__()
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]
        self.num_patches = int(num_time_patches * num_freq_patches)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask_ratio: float = 0, augment: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        args:
            x: batch of mel spectrograms (batch_size, 1, freq, time)
            y: batch of one-hot labels (batch_size, num_classes)
        """
        if augment and self.use_mixup:
            x, y = apply_mixup(x, y, self.mixup_beta, self.mixup_chance, self.mixup_hard_label)  # B, 1, F, T

        if augment and self.use_tf_mask:
            for _ in range(self.tf_mask_repeats):
                x = self.freq_masking(x)
                x = self.time_masking(x)

        mask_info = self.random_masking(batch_size=x.shape[0], mask_ratio=mask_ratio) if mask_ratio > 0 else None

        # our ViT expects B, 1, T, F shape
        x = x.transpose(2, 3)  # B, 1, T, F

        return x, y, mask_info

    def random_masking(self, batch_size: int, mask_ratio: float = 0, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Applies random patch masking for generative tasks."""
        assert 0. < mask_ratio < 1.
        len_keep = int(self.num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=device)  # noise in [0, 1]

        ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep_sorted, _ = ids_keep.sort(dim=1)

        mask = torch.ones([batch_size, self.num_patches], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return {'mask': mask, 'ids_keep_sorted': ids_keep_sorted}
