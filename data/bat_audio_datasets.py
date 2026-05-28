__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import re
from pathlib import Path
import json
import random
import numpy as np
import librosa
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torchaudio
from datasets import load_dataset, Audio
from datasets import Dataset as HuggingFaceDataset
import concurrent.futures
import soundfile as sf
from typing import Tuple, List, Dict, Optional, Union
from .augmentations import apply_mixup, apply_color_noise, apply_random_gain


# AudioSet
def curate_audioset_dfs(root_dir: Union[str, Path] = 'Datasets/AudioSet/',
                        version: str = 'as20k') -> Tuple[pd.DataFrame, pd.DataFrame]:
    root_dir = Path(root_dir)
    eval_csv_path = root_dir.joinpath('eval_segments.csv')
    eval_wav_path = root_dir.joinpath('eval_segments')
    ontology_path = root_dir.joinpath('ontology.json')

    if version == 'as20k':
        train_csv_path = root_dir.joinpath('balanced_train_segments.csv')
        train_wav_path = root_dir.joinpath('balanced_train_segments')
    elif version == 'as2m':
        train_csv_path = root_dir.joinpath('unbalanced_train_segments.csv')
        train_wav_path = root_dir.joinpath('unbalanced_train_segments')
    else:
        raise ValueError(f"Invalid AudioSet version: {version}")

    def curate(csv_path: Path, wav_path: Path, id_to_name: Dict[str, str]) -> pd.DataFrame:
        with open(csv_path, "r") as f:
            lines = f.readlines()
        df = pd.DataFrame(lines, columns=["raw_data"])
        df = df.drop([0, 1, 2])
        df.reset_index(drop=True, inplace=True)
        df = df["raw_data"].str.split(",", n=3, expand=True)
        df.columns = ["YTID", "start_seconds", "end_seconds", "positive_labels"]
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df["positive_labels"] = df["positive_labels"].str.replace('"', "", regex=False)
        df["positive_labels"] = df["positive_labels"].str.replace(" ", "", regex=False)

        paths = list(wav_path.glob('*.wav'))
        file_df = pd.DataFrame({"file_path": [f.as_posix() for f in paths], "file_name": [f.stem for f in paths]})
        file_df["file_name"] = file_df["file_name"].str.replace(r"^Y", "", regex=True)

        df = pd.merge(df, file_df, left_on="YTID", right_on="file_name", how="left")
        df = df.drop(columns=["file_name"])
        df = df[df["file_path"].notna()]

        df["positive_labels_list"] = df["positive_labels"].str.split(",")
        df["positive_labels_names"] = df["positive_labels_list"].apply(
            lambda label_list: [id_to_name.get(label, "Unknown") for label in label_list])
        return df

    with open(ontology_path, "r") as f:
        ontology = json.load(f)
    id_to_name = {obj["id"]: obj["name"] for obj in ontology}
    eval_df = curate(eval_csv_path, eval_wav_path, id_to_name)
    train_df = curate(train_csv_path, train_wav_path, id_to_name)

    eval_corrupt_files = [root_dir.joinpath('eval_segments/YmW3S0u8bj58.wav').as_posix()]
    train_corrupt_files = [root_dir.joinpath('unbalanced_train_segments/YwM5Qf5xXT8w.wav').as_posix()]
    eval_df = eval_df[~eval_df.file_path.isin(eval_corrupt_files)]
    train_df = train_df[~train_df.file_path.isin(train_corrupt_files)]
    eval_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    all_names = sorted(list(set([n for labels_names in eval_df.positive_labels_names.values for n in labels_names])))
    name_to_label = {n: i for i, n in enumerate(all_names)}

    def get_codes(name_list: List[str]) -> str:
        return ','.join([str(name_to_label[name]) for name in name_list if name in name_to_label])

    eval_df["codes"] = eval_df["positive_labels_names"].apply(get_codes)
    train_df["codes"] = train_df["positive_labels_names"].apply(get_codes)

    cols_to_drop = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels', 'positive_labels_list',
                    'positive_labels_names']
    eval_df = eval_df.drop(columns=cols_to_drop)
    train_df = train_df.drop(columns=cols_to_drop)

    eval_df['codes'] = np.array(eval_df.codes).astype(np.bytes_)
    train_df['codes'] = np.array(train_df.codes).astype(np.bytes_)
    eval_df['file_path'] = np.array(eval_df.file_path).astype(np.bytes_)
    train_df['file_path'] = np.array(train_df.file_path).astype(np.bytes_)

    return train_df, eval_df


class SSLAudioSet(Dataset):
    def __init__(self, wav_files: np.ndarray, sr: int = 16_000):
        super().__init__()
        self.wav_files = wav_files
        self.sr = sr

    def __getitem__(self, idx: int) -> torch.Tensor:
        f = self.wav_files[idx].decode('utf-8')
        x, _ = librosa.load(path=f, sr=self.sr, res_type="soxr_hq")
        while x.shape[0] == 0:  # some files are empty/corrupt
            idx = random.randrange(len(self.wav_files))
            f = self.wav_files[idx].decode('utf-8')
            x, sr = librosa.load(path=f, sr=self.sr, res_type="soxr_hq")
        x = torch.from_numpy(x).float()
        return torch.roll(x, random.randrange(len(x)))

    def __len__(self) -> int:
        return len(self.wav_files)

    def collate_fn(self, x: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(x, batch_first=True, padding_value=0.)  # batch_size, wave_length


class SSLDataTransforms(nn.Module):
    def __init__(self,
                 sr: int = 16_000,
                 n_fft: int = 1024,
                 hop_length: int = 160,
                 n_mels: int = 128,
                 time_frame_size: int = 1024,
                 num_time_patches: int = 64,
                 num_freq_patches: int = 8,
                 ):

        super().__init__()
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]
        self.mel = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.db = AmplitudeToDB('power', top_db=80)
        self.num_patches = int(num_time_patches * num_freq_patches)

    def forward(self, x: torch.Tensor, num_views: int = 1, mask_ratio: float = 0) -> Tuple[
        torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        args:
            x: batch of waveforms (batch_size, wave_length)
        """
        # if you don't add the channel, db uses batch statistics and you get non-reproducible results
        x = self.mel(x).unsqueeze(1)  # B, 1, F, T
        x = self.db(x)  # B, 1, F, T
        x = self.minmax_normalize(x)  # B, 1, F, T
        x = F.pad(x, (0, self.time_frame_size - x.shape[3]))  # B, 1, F, T=time_frames_size
        # each entry is repeated num_views times before the next entry
        x = x.repeat_interleave(num_views, dim=0)  # B * num_views, 1, F, T
        mask_info = self.inverse_block_mask(batch_size=x.shape[0], mask_ratio=mask_ratio) if mask_ratio > 0 else None
        return x.transpose(2, 3), mask_info

    def minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape  # B, 1, F, T
        x = x.flatten(1)  # B, F*T
        min_, max_ = x.aminmax(dim=1, keepdim=True)
        x = (x - min_) / (max_ - min_ + 1e-8)
        return x.reshape(shape)

    def inverse_block_mask(self, batch_size: int, mask_ratio: float = 0.8, mask_length: int = 5,
                           mask_prob_adjust: float = 0.07, device: str = 'cpu') -> Dict[str, torch.Tensor]:

        keep_ratio = 1 - mask_ratio

        # Select initial block centers
        masking_size = int(self.num_patches * (keep_ratio + mask_prob_adjust) / (mask_length ** 2))
        center_inds = torch.randint(0, self.num_patches, size=(batch_size, masking_size), device=device)

        mask = torch.zeros((batch_size, self.num_patches), device=device)
        mask.scatter_(1, center_inds, 1)
        mask = mask.view(batch_size, 1, self.num_time_patches, self.num_freq_patches)

        # Expand centers to blocks of size mask_length x mask_length using max pooling
        padding = mask_length // 2
        mask = F.max_pool2d(mask, kernel_size=mask_length, stride=1, padding=padding)
        mask = mask[:, 0, :self.num_time_patches, :self.num_freq_patches].reshape(batch_size, -1)

        # we adjust the mask density to make sure every sample has the same number of dropped/kept patches
        target_len = int(self.num_patches * keep_ratio)
        for m in mask:
            n = int(m.sum().item())
            if n > target_len:
                m[torch.multinomial(m, n - target_len, replacement=False)] = 0
            elif n < target_len:
                m[torch.multinomial(1 - m, target_len - n, replacement=False)] = 1

        mask = 1 - mask  # inverse mask (1 = remove, 0 = keep); we will use this for our loss later
        len_keep = int(self.num_patches - mask[0].sum().item())
        ids_keep_sorted, _ = mask.argsort(dim=1)[:, :len_keep].sort(dim=1)  # decoder needs the index of dropped patches
        return {'mask': mask, 'ids_keep_sorted': ids_keep_sorted}


class AudioSet(Dataset):
    """AudioSet supervised classification dataset."""

    def __init__(self,
                 df: pd.DataFrame,
                 num_classes: int = 527,
                 sr: int = 16000,
                 sample_dur: float = 10.23,
                 train: bool = False):

        super().__init__()
        self.df = df
        self.num_classes = num_classes
        self.sr = sr
        self.sample_dur = sample_dur
        self.num_samples = int(sample_dur * sr)
        self.train = train

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.df.iloc[idx].file_path.decode('UTF-8')
        x, sr = librosa.load(path=f, sr=self.sr, res_type="soxr_hq")

        while x.shape[0] == 0:  # some files are empty/corrupt
            idx = random.choice(range(len(self.df)))
            f = self.df.iloc[idx].file_path.decode('UTF-8')
            x, sr = librosa.load(path=f, sr=self.sr, res_type="soxr_hq")

        x = torch.from_numpy(x).float()

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

        codes = self.df.iloc[idx].codes.decode('UTF-8').split(',')
        y = torch.tensor([int(c) for c in codes])
        y = F.one_hot(y, num_classes=self.num_classes).amax(dim=0).float()
        return x, y

    def __len__(self) -> int:
        return self.df.shape[0]

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = zip(*xy)
        x = torch.stack(x).float()  # B, T
        y = torch.stack(y).float()  # B, C
        return x, y


# Speech Command V2
def prepare_speech_commands_dfs(root_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root_dir = Path(root_dir)
    target_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    class_mapping = {word: i for i, word in enumerate(target_words)}
    unknown_label, silence_label = 10, 11

    def load_split_file(filename: str) -> set:
        split_file = root_dir / filename
        if split_file.exists():
            with open(split_file, 'r') as f:
                return set(f.read().splitlines())
        return set()

    val_files = load_split_file('validation_list.txt')
    test_files = load_split_file('testing_list.txt')
    data = []

    for folder in root_dir.iterdir():
        if not folder.is_dir() or folder.name == '_background_noise_': continue
        word = folder.name
        label = class_mapping.get(word, unknown_label)
        for wav_file in folder.glob('*.wav'):
            relative_path = f"{word}/{wav_file.name}"
            split = 'val' if relative_path in val_files else 'test' if relative_path in test_files else 'train'
            data.append({'filepath': str(wav_file), 'label': label, 'split': split, 'is_background': False,
                         'offset': 0.0, 'duration': None})

    bg_dir = root_dir / '_background_noise_'
    if bg_dir.exists():
        for wav_file in bg_dir.glob('*.wav'):
            duration = librosa.get_duration(path=wav_file)
            num_chunks = math.floor(duration)
            for i in range(num_chunks):
                split = 'train' if i < num_chunks * 0.8 else 'val' if i < num_chunks * 0.9 else 'test'
                data.append({'filepath': str(wav_file), 'label': silence_label, 'split': split, 'is_background': True,
                             'offset': float(i), 'duration': 1.0})

    df = pd.DataFrame(data)
    df['filepath'] = np.array(df.filepath).astype(np.bytes_)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    return train_df, val_df, test_df


class SpeechCommands(Dataset):
    def __init__(self, dataframe: pd.DataFrame, sr: int = 16000, max_len: int = 16000, one_hot_labels: bool = False):
        self.df = dataframe
        self.sr = sr
        self.max_len = max_len  # 1 second
        self.num_classes = 12
        self.one_hot_labels = one_hot_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.df.iloc[idx]
        if self.one_hot_labels:
            y = F.one_hot(torch.tensor(entry['label']), num_classes=self.num_classes).float()
        else:
            y = torch.tensor(entry['label'], dtype=torch.long)

        dur = None if pd.isna(entry['duration']) else entry['duration']

        x, _ = librosa.load(entry['filepath'].decode('utf-8'), sr=self.sr, offset=entry['offset'], duration=dur,
                            res_type="soxr_vhq")
        x = torch.from_numpy(x).float()
        x = F.pad(x, (0, self.max_len - x.shape[0]))
        return x, y

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = zip(*xy)
        return torch.stack(x), torch.stack(y)


# ESC-50
def prepare_esc50_dfs(root_dir: Union[str, Path], test_fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares cross-validation folds for ESC-50."""
    root_path = Path(root_dir)
    csv_path = root_path / "meta" / "esc50.csv"
    df = pd.read_csv(csv_path)
    df['filename'] = df.filename.apply(lambda f: root_path / "audio" / f)
    test_df = df[df['fold'] == test_fold].reset_index(drop=True)
    train_df = df[df['fold'] != test_fold].reset_index(drop=True)
    return train_df, test_df


class ESC50Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, sr: int = 16000, one_hot_labels: bool = False):
        self.df = df
        self.sr = sr
        self.num_classes = 50
        self.one_hot_labels = one_hot_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.df.iloc[idx]
        if self.one_hot_labels:
            y = F.one_hot(torch.tensor(entry['target']), num_classes=self.num_classes).float()
        else:
            y = torch.tensor(entry['target'], dtype=torch.long)

        x, _ = librosa.load(path=entry['filename'], sr=self.sr, res_type="soxr_vhq")
        x = torch.from_numpy(x).float()
        return x, y

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = zip(*xy)
        return torch.stack(x), torch.stack(y)


# LibriSpeech with hugging-face datasets
class CharTokenizer:
    """Character-level tokenizer for Speech Recognition on LibriSpeech."""

    def __init__(self):
        self.chars = ["<blank>", " ", "'"] + [chr(i) for i in range(97, 123)]
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of character IDs."""
        text = re.sub(r"[^a-z ']", "", text.lower())
        return [self.char2idx[c] for c in text]

    def decode(self, ids: List[int], ignore_blanks: bool = True) -> str:
        """Decodes list of character IDs back into a string."""
        return "".join([self.idx2char[i] for i in ids if not (ignore_blanks and i == 0)])


def prepare_librispeech(cache_dir: Union[str, Path] = None) -> Tuple[
    HuggingFaceDataset, HuggingFaceDataset, HuggingFaceDataset, CharTokenizer]:
    tokenizer = CharTokenizer()
    ds = load_dataset("librispeech_asr", "clean", cache_dir=cache_dir)

    def format_ds(split: str) -> HuggingFaceDataset:
        d = ds[split].cast_column("audio", Audio(sampling_rate=16000))
        return d.map(lambda b: {"labels": tokenizer.encode(b["text"])},
                     remove_columns=["file", "speaker_id", "chapter_id", "id"])

    return format_ds("train.100"), format_ds("validation"), format_ds("test"), tokenizer


class LibriSpeechDataset(Dataset):
    def __init__(self,
                 ds: HuggingFaceDataset,
                 sr: int = 16_000,
                 n_mels: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 160,
                 vit_patch_stride: int = 16,
                 upsample_factor: int = 8,
                 augment: bool = False,
                 use_color_noise: bool = False,
                 use_tf_mask: bool = False,
                 color_noise_chance: Optional[float] = None,
                 freq_mask_param: Optional[int] = None,
                 time_mask_param: Optional[int] = None,
                 tf_mask_repeats: int = 1):

        super().__init__()
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]
        self.mel = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.db = AmplitudeToDB('power', top_db=80)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        entry = self.ds[idx]
        y = torch.tensor(entry["labels"], dtype=torch.long)
        x = torch.tensor(entry['audio']['array'], dtype=torch.float32)

        if self.augment and self.use_color_noise:
            x = apply_color_noise(x, color_noise_chance=self.color_noise_chance)

        x = self.mel(x)  # F, T
        x = self.db(x)  # F, T
        x = self.minmax_normalize(x)  # F, T

        if x.shape[1] % self.vit_patch_stride > 0:
            x = F.pad(x, (0, self.vit_patch_stride - x.shape[1] % self.vit_patch_stride))

        input_length = (x.shape[1] // self.vit_patch_stride) * self.upsample_factor
        target_length = y.shape[0]
        x = x.T  # T, F
        return x, y, input_length, target_length

    def minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_, max_ = x.aminmax()
        return (x - min_) / (max_ - min_ + 1e-8)

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


# High Sierra Nevada
def prepare_hsn_dfs(cache_dir: Union[str, Path] = None,
                    dataset_name: str = 'HSN') -> Tuple[pd.DataFrame, pd.DataFrame, int]:

    ebird_codes_path = Path(cache_dir).joinpath('HSN_ebird_codes.json')
    with open(ebird_codes_path) as f:
        ebird_codes = json.load(f)
        label2id = ebird_codes['label2id']
        id2label = {int(v): k for k, v in label2id.items()}
    num_classes = 1 + max(list(id2label.keys()))
    ds = load_dataset("DBD-research-group/BirdSet", dataset_name, trust_remote_code=True, cache_dir=cache_dir)
    df_test_5s = ds['test_5s'].to_pandas()
    df_train = ds['train'].to_pandas()

    def get_duration(path: str) -> float:
        try:
            x, sr = sf.read(path)
            return x.shape[0] / sr
        except Exception:
            raise ValueError(f"{path} is corrupt.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        durations = list(executor.map(get_duration, df_train['filepath']))

    df_train['length'] = np.array(durations)
    df_train = df_train.loc[:, ['filepath', 'ebird_code_multilabel', 'length']]
    df_test_5s = df_test_5s.loc[:, ['filepath', 'ebird_code_multilabel']]
    df_train['filepath'] = np.array(df_train.filepath).astype(np.bytes_)
    df_test_5s['filepath'] = np.array(df_test_5s.filepath).astype(np.bytes_)
    assert df_train['ebird_code_multilabel'].apply(lambda x: max(x)).unique().shape[0] == num_classes
    return df_train.reset_index(drop=True), df_test_5s.reset_index(drop=True), num_classes


class BirdSet(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 sr: int = 16_000,
                 duration_seconds: int = 5,
                 num_classes: int = 21,
                 use_events: bool = False,
                 test: bool = False):

        super().__init__()
        self.df = df
        self.sr = sr
        self.duration_seconds = duration_seconds
        self.num_classes = num_classes
        self.test = test
        self.num_samples = int(duration_seconds * sr)
        self.use_events = use_events

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.test:
            return self.get_test(idx)  # (1, T), (C)
        return self.get_train(idx)  # (1, T), (C)

    def __len__(self) -> int:
        return len(self.df)

    def get_train(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return x, y

    def get_test(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        d = int(self.num_samples - x.shape[0])
        x = F.pad(x, (0, d))
        return x, y

    def collate_fn(self, xy: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = zip(*xy)
        x = torch.stack(x).float()  # B, T
        y = torch.stack(y).float()  # B, C
        return x, y


# Dcase 2016 Task 2 Sound Event Detection
def load_dcase2016_task2_json_to_df(root_path: Union[str, Path], split: str) -> pd.DataFrame:
    json_path = Path(root_path).joinpath(f'{split}.json').as_posix()
    with open(json_path, 'r') as f:
        data = json.load(f)
    rows = []
    for filename, events in data.items():
        for event in events:
            rows.append({'filename': filename, 'label': event['label'], 'start_ms': event['start'],
                         'end_ms': event['end']})
    df = pd.DataFrame(rows)
    df['filename'] = df.filename.apply(lambda f: Path(root_path).joinpath(f"48000/{split}/{f}").as_posix())
    return df


class DCASE2016Task2Dataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 vocab_df: pd.DataFrame,
                 sr: int = 16_000,
                 n_mels: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 160,
                 patch_stride_ms: int = 160,
                 time_frame_size: int = 12000,
                 num_temporal_patches: int = 750,
                 augment: bool = False,
                 use_mixup: bool = False,
                 use_color_noise: bool = False,
                 use_tf_mask: bool = False,
                 mixup_beta: float = 0.8,
                 mixup_chance: float = 0.9,
                 mixup_hard_label: bool = False,
                 color_noise_chance: Optional[float] = None,
                 freq_mask_param: Optional[int] = None,
                 time_mask_param: Optional[int] = None,
                 tf_mask_repeats: int = 1):

        super().__init__()
        self.grouped_events = df.groupby('filename')
        self.filenames = list(self.grouped_events.groups.keys())
        self.num_classes = len(vocab_df)
        self.label_to_idx = dict(zip(vocab_df['label'], vocab_df['idx']))

        self.sr = sr
        self.mel = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.db = AmplitudeToDB('power', top_db=80)
        self.time_frame_size = time_frame_size
        # this dataset has fixed 120 sec recordings. Our ViT produces 750 time-patches with a patch-size of 16,
        # and each spectrogram frame is 10 ms (sr=16_000, hop=160), so each ViT patch is patch_stride_ms = 160 ms
        # we need this information to align the targets with the model prediction
        self.num_temporal_patches = num_temporal_patches
        self.patch_centers = torch.arange(num_temporal_patches, dtype=torch.float32) * patch_stride_ms

        self.augment = augment
        self.use_mixup = use_mixup
        self.use_color_noise = use_color_noise
        self.use_tf_mask = use_tf_mask

        if use_mixup:
            self.mixup_beta = mixup_beta
            self.mixup_chance = mixup_chance
            self.mixup_hard_label = mixup_hard_label

        if use_color_noise:
            self.color_noise_chance = color_noise_chance

        if use_tf_mask:
            self.tf_mask_repeats = tf_mask_repeats
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

        if self.augment and self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        x = self.mel(x)  # F, T
        x = self.db(x)  # F, T
        x = self.minmax_normalize(x)  # F, T
        x = F.pad(x, (0, self.time_frame_size - x.shape[1]))
        x = x.T  # T, F
        y = torch.zeros((self.num_temporal_patches, self.num_classes), dtype=torch.float32)

        for _, row in events.iterrows():
            class_idx = self.label_to_idx[row['label']]
            start_ms = row['start_ms']
            end_ms = row['end_ms']
            # find which ViT's temporal-patches fall inside this event's time boundaries
            active_frames = (self.patch_centers >= start_ms) & (self.patch_centers <= end_ms)
            y[active_frames, class_idx] = 1.0

        return x, y

    def minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_, max_ = x.aminmax()
        return (x - min_) / (max_ - min_ + 1e-8)

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = zip(*batch)
        y = torch.stack(y)  # B, T, C
        x = torch.stack(x).unsqueeze(1)  # B, 1, T, F

        if self.augment and self.use_mixup:
            x, y = apply_mixup(x, y, self.mixup_beta, self.mixup_chance, self.mixup_hard_label)

        if self.augment and self.use_tf_mask:
            x = x.transpose(2, 3)  # B, 1, F, T
            for _ in range(self.tf_mask_repeats):
                x = self.freq_masking(x)
                x = self.time_masking(x)
            x = x.transpose(2, 3)  # B, 1, T, F

        return x, y


class DataTransforms(nn.Module):
    """General preprocessing module for spectrogram extraction and augmentations."""

    def __init__(self,
                 sr: int = 16_000,
                 n_fft: int = 1024,
                 hop_length: int = 160,
                 n_mels: int = 128,
                 f_min: int = 0,
                 time_frame_size: Optional[int] = None,
                 num_time_patches: Optional[int] = None,
                 num_freq_patches: Optional[int] = None,
                 use_time_roll: bool = False,
                 use_mixup: bool = False,
                 mixup_chance: float = 0.9,
                 mixup_beta: float = .8,
                 mixup_hard_label: bool = False,
                 use_random_gain: bool = False,
                 use_color_noise: bool = False,
                 color_noise_chance: float = 0.3,
                 use_tf_mask: bool = False,
                 freq_mask_param: int = 16,
                 time_mask_param: int = 64,
                 tf_mask_repeats: int = 1,
                 ):
        super().__init__()
        [setattr(self, arg_name, arg_value) for arg_name, arg_value in locals().items() if arg_name != "self"]
        self.mel = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=f_min)
        self.num_patches = int(num_time_patches * num_freq_patches)
        self.db = AmplitudeToDB('power', top_db=80)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask_ratio: float = 0, augment: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        args:
            x: batch of waveforms (batch_size, time)
            y: batch of one-hot labels (batch_size, num_classes)
        """

        if augment and self.use_random_gain:
            x = apply_random_gain(x)

        if augment and self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)  # B, T

        x = self.mel(x).unsqueeze(1)  # B, 1, F, T
        x = self.db(x)  # B, 1, F, T
        x = self.minmax_normalize(x)  # B, 1, F, T

        if augment and self.use_time_roll:
            x = torch.roll(x, shifts=random.randint(0, x.shape[-1]), dims=-1)

        if x.shape[3] != self.time_frame_size:
            x = F.pad(x, (0, self.time_frame_size - x.shape[3]), mode='constant', value=0)

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

    def minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes features to [0, 1] using min and max values."""
        shape = x.shape  # B, *
        x = x.flatten(1)
        min_, max_ = x.aminmax(dim=1, keepdim=True)
        x = (x - min_) / (max_ - min_ + 1e-8)
        return x.reshape(shape)

    def random_masking(self, batch_size: int, mask_ratio: float = 0, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Applies random patch masking for generative tasks."""
        assert 0. < mask_ratio < 1.
        len_keep = int(self.num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=device)  # noise in [0, 1]

        ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep_sorted, _ = ids_keep.sort(dim=1)  # sort the indices

        mask = torch.ones([batch_size, self.num_patches], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return {'mask': mask, 'ids_keep_sorted': ids_keep_sorted}
