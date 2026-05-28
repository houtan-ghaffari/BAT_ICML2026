__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import numpy as np
import torch
from typing import Tuple


def apply_random_gain(x: torch.Tensor) -> torch.Tensor:
    """ x: a batch of audio waveforms of shape (B, T) """
    v = np.random.beta(10.0, 10.0, size=(x.shape[0], 1))
    v = torch.from_numpy(v).to(x) + 0.5
    return x * v


def apply_color_noise(x: torch.Tensor, color_noise_chance: float) -> torch.Tensor:
    """
    Applies power-law noise to a 1D audio waveform or a batch of waveforms.
    Expects x of shape (batch, num_samples) or (num_samples,).
    """
    squeeze_output = False
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze_output = True

    batch_size, num_samples = x.shape
    device = x.device
    dtype = x.dtype

    # f_decay: 1.0 is pink, 2.0 is brown, -1.0 is blue, -2.0 is violet
    f_decay = torch.empty((batch_size, 1), dtype=dtype, device=device).uniform_(-2.0, 2.0)
    snr_db = torch.empty((batch_size, 1), dtype=dtype, device=device).uniform_(5.0, 25.0)
    noise = torch.randn(batch_size, num_samples, dtype=dtype, device=device)

    spec = torch.fft.rfft(noise, dim=1)
    freqs = torch.arange(1, spec.shape[1] + 1, dtype=dtype, device=device).unsqueeze(0)
    mask = 1.0 / (freqs ** (f_decay / 2.0))  # divide f_decay by 2 because we are working in power not amplitude
    spec *= mask

    noise = torch.fft.irfft(spec, n=num_samples, dim=1)
    noise = noise / (1e-8 + noise.square().mean(dim=-1, keepdim=True).sqrt())

    clean_rms = x.square().mean(dim=-1, keepdim=True).sqrt()
    noise_amp = clean_rms / (10 ** (snr_db / 20.0))

    prob_mask = torch.empty((batch_size, 1), dtype=dtype, device=device).bernoulli_(color_noise_chance)
    noise_amp = noise_amp * prob_mask

    out = x + (noise_amp * noise)
    return out.squeeze(0) if squeeze_output else out


def apply_mixup(x: torch.Tensor,
                y: torch.Tensor,
                mixup_beta: float,
                mixup_chance: float,
                mixup_hard_label: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Mixup to a batch of spectrograms and labels.
    """
    batch_size = x.shape[0]
    a = torch.from_numpy(np.random.beta(mixup_beta, mixup_beta, (batch_size, 1, 1, 1))).to(x)

    if mixup_chance < 1:
        a *= torch.empty_like(a).bernoulli_(mixup_chance)

    x = (1 - a) * x + a * x.flip(0)

    if mixup_hard_label:
        y = y + y.flip(0)
    else:
        reshape_dims = [-1] + [1] * (y.ndim - 1)
        a_y = a.reshape(*reshape_dims)
        y = (1 - a_y) * y + a_y * y.flip(0)

    y = torch.clamp(y, min=0., max=1.)
    return x, y
