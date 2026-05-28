__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import os
import random
import numpy as np
import torch
from typing import Optional, Iterable, Iterator


class AsymmetricLossMultiLabel(torch.nn.Module):
    """Asymmetric Loss for multi-label classification tasks."""

    def __init__(
            self,
            gamma_neg: float = 4.0,
            gamma_pos: float = 1.0,
            clip: Optional[float] = 0.05,
            eps: float = 1e-8,
            disable_torch_grad_focal_loss: bool = True,
            reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:

            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt0 = xs_pos * y
                    pt1 = xs_neg * (1 - y)
                    pt = pt0 + pt1
                    one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            else:
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            loss *= one_sided_w

        if self.reduction == "mean":
            return -loss.mean()
        if self.reduction == "sum":
            return -loss.sum()

        return -loss


def smooth_binary_targets(targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    assert 0 <= smoothing < 1, "smoothing factor must be between 0 and 1"
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def infinite_batch_iterator(data_loader: Iterable) -> Iterator:
    """Yields batches indefinitely from the provided data loader."""
    while True:
        for batch in data_loader:
            yield batch


def seed_everything(default_seed: int = 32) -> None:
    env_seed = os.environ.get('PYTHONHASHSEED')
    if env_seed is not None:
        seed = int(env_seed)
        print(f"[Info] Found PYTHONHASHSEED in environment. Locking seed to: {seed}")
    else:
        seed = default_seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"[Info] No PYTHONHASHSEED found. Defaulting seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
