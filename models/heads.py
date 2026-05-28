__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Dict, Any, Type


class LinearHead(nn.Module):
    """Standard linear classification head for processing Transformer output tokens."""

    def __init__(self,
                 num_classes: Optional[int] = None,
                 dim: int = 768,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 frame_wise_task: bool = False,
                 use_cls: bool = True,
                 dropout=0):

        super().__init__()
        self.ln = norm_layer(dim) if norm_layer else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim, num_classes, bias=True)
        if frame_wise_task:
            self.forward_fn = self.frame_wise_forward
        else:
            if use_cls:
                self.forward_fn = self.cls_forward
            else:
                self.forward_fn = self.patch_avg_forward

    def cls_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass using only the final layer's class token."""
        x = feat_cache['cls_tokens'][-1]  # B, D
        return self.fc(self.dropout(self.ln(x)))

    def patch_avg_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass using the global average pool of the final layer's patch tokens."""
        x = feat_cache['patch_tokens'][-1]  # B, D, T, F
        x = x.flatten(2)  # B, D, N
        x = x.mean(dim=2)  # B, D
        return self.fc(self.dropout(self.ln(x)))

    def frame_wise_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass for temporal frame-wise predictions, averaging over the frequency dimension."""
        x = feat_cache['patch_tokens'][-1]  # B, D, T, F
        x = x.mean(dim=3)  # B, D, T
        x = x.transpose(1, 2)  # B, T, D
        return self.fc(self.dropout(self.ln(x)))

    def forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        return self.forward_fn(feat_cache)


class LinearCGP(nn.Module):
    """Linear head utilizing a learned weighted sum of all layer outputs."""

    def __init__(self,
                 num_classes: Optional[int] = None,
                 dim: int = 768,
                 num_layers: int = 12,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 frame_wise_task: bool = False,
                 use_cls: bool = True,
                 dropout: float = 0):

        super().__init__()
        self.num_layers = num_layers
        self.block_attention = nn.Parameter(torch.zeros(num_layers))
        self.ln = norm_layer(dim) if norm_layer else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim, num_classes, bias=True)
        if frame_wise_task:
            self.forward_fn = self.frame_wise_forward
        else:
            if use_cls:
                self.forward_fn = self.cls_forward
            else:
                self.forward_fn = self.patch_avg_forward

    def cls_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass aggregating class tokens from all layers."""
        x = feat_cache['cls_tokens']  # L, B, D
        a = torch.softmax(self.block_attention, dim=0).reshape(-1, 1, 1)  # L, 1, 1
        x = (a * x).sum(dim=0)  # B, D
        return self.fc(self.dropout(self.ln(x)))

    def patch_avg_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass aggregating globally averaged patch tokens from all layers."""
        x = feat_cache['patch_tokens']  # L, B, D, T, F
        x = x.flatten(3)  # L, B, D, N=T*F
        x = x.mean(dim=3)  # L, B, D
        a = torch.softmax(self.block_attention, dim=0).reshape(-1, 1, 1)  # L, 1, 1
        x = (a * x).sum(dim=0)  # B, D
        return self.fc(self.dropout(self.ln(x)))

    def frame_wise_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass aggregating frame-wise patch tokens from all layers."""
        x = feat_cache['patch_tokens']  # L, B, D, T, F
        x = x.mean(dim=4)  # L, B, D, T
        a = torch.softmax(self.block_attention, dim=0).reshape(-1, 1, 1, 1)  # L, 1, 1, 1
        x = (a * x).sum(dim=0)  # B, D, T
        x = x.transpose(1, 2)  # B, T, D
        return self.fc(self.dropout(self.ln(x)))

    def forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        return self.forward_fn(feat_cache)


class CGP(nn.Module):
    def __init__(self,
                 num_classes: Optional[int] = None,
                 dim: int = 768,
                 num_prototypes: int = 10000,
                 num_layers: int = 12,
                 frame_wise_task: bool = False,
                 use_cls: bool = True,
                 dropout: float = 0.2):

        super().__init__()
        self.num_layers = num_layers
        self.num_prototypes = num_prototypes
        self.prototype_vectors = nn.Parameter(torch.randn(self.num_prototypes, dim, 1, 1) * 0.02)
        self.block_attention = nn.Parameter(torch.ones(num_layers))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.num_prototypes * 3, num_classes)

        if frame_wise_task:
            self.forward_fn = self.frame_wise_forward
        else:
            if use_cls:
                self.forward_fn = self.patch_cls_forward
            else:
                self.forward_fn = self.patch_avg_forward

    def patch_cls_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Computes similarity of layer-aggregated patch and class tokens against prototypes."""
        x_patch = feat_cache['patch_tokens']  # L, B, D, T, F
        x_cls = feat_cache['cls_tokens']  # L, B, D
        num_layers, batch_size, feat_dim, time_patch, freq_patch = x_patch.shape
        assert num_layers == self.num_layers, x_patch.shape

        x_patch = F.normalize(x_patch, dim=2)  # L, B, D, T, F
        x_cls = F.normalize(x_cls, dim=2)  # L, B, D
        p = F.normalize(self.prototype_vectors, dim=1)  # P, D, 1, 1

        a = torch.softmax(self.block_attention, dim=0)

        a_patch = a.reshape(num_layers, 1, 1, 1, 1)
        x_patch = (a_patch * x_patch).sum(dim=0)  # B, D, T, F

        a_cls = a.reshape(num_layers, 1, 1)
        x_cls = (a_cls * x_cls).sum(dim=0)  # B, D

        act_patch = F.conv2d(x_patch, p)  # (B, P, T, F)
        act_patch = act_patch.reshape(batch_size, self.num_prototypes, -1)  # (B, P, T*F)
        act_patch_min = act_patch.min(dim=-1).values
        act_patch_max = act_patch.max(dim=-1).values

        p = p.squeeze()  # P, D
        act_cls = x_cls @ p.T  # B, D @ D, P -> B, P

        similarity_features = torch.stack([act_patch_min, act_cls, act_patch_max], dim=-1)  # (B, P, 3)
        similarity_features = similarity_features.reshape(batch_size, 3 * self.num_prototypes)  # (B, 3*P)

        return self.fc(self.dropout(similarity_features))

    def patch_avg_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Computes similarity using min, max, and mean pooling of patch tokens against prototypes."""
        x_patch = feat_cache['patch_tokens']
        num_layers, batch_size, feat_dim, time_patch, freq_patch = x_patch.shape

        x_patch = F.normalize(x_patch, dim=2)  # L, B, D, T, F
        p = F.normalize(self.prototype_vectors, dim=1)

        a = torch.softmax(self.block_attention, dim=0)

        a_patch = a.reshape(num_layers, 1, 1, 1, 1)
        x_patch = (a_patch * x_patch).sum(dim=0)  # B, D, T, F

        act_patch = F.conv2d(x_patch, p)
        act_patch = act_patch.reshape(batch_size, self.num_prototypes, -1)
        act_patch_min = act_patch.min(dim=-1).values
        act_patch_max = act_patch.max(dim=-1).values
        act_patch_mu = act_patch.mean(dim=-1)

        similarity_features = torch.stack([act_patch_min, act_patch_mu, act_patch_max], dim=-1)
        similarity_features = similarity_features.reshape(batch_size, 3 * self.num_prototypes)

        return self.fc(self.dropout(similarity_features))

    def frame_wise_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Computes frame-wise similarity of patch tokens against prototypes for temporal tasks."""
        x_patch = feat_cache['patch_tokens']
        num_layers, batch_size, feat_dim, time_patch, freq_patch = x_patch.shape

        x_patch = F.normalize(x_patch, dim=2)  # L, B, D, T, F
        p = F.normalize(self.prototype_vectors, dim=1)

        a = torch.softmax(self.block_attention, dim=0)

        a_patch = a.reshape(num_layers, 1, 1, 1, 1)
        x_patch = (a_patch * x_patch).sum(dim=0)

        act_patch = F.conv2d(x_patch, p)  # B, P, T, F
        act_patch_min = act_patch.min(dim=-1).values  # B, P, T
        act_patch_max = act_patch.max(dim=-1).values  # B, P, T
        act_patch_mu = act_patch.mean(dim=-1)  # B, P, T

        similarity_features = torch.stack([act_patch_min, act_patch_mu, act_patch_max], dim=-1)  # B, P, T, 3
        similarity_features = similarity_features.transpose(1, 2)  # B, T, P, 3
        similarity_features = similarity_features.reshape(batch_size, time_patch, 3 * self.num_prototypes)

        return self.fc(self.dropout(similarity_features))

    def forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        return self.forward_fn(feat_cache)


class Protobin(nn.Module):

    def __init__(self, num_classes: Optional[int] = None, dim: int = 768, num_prototypes: int = 10000,
                 frame_wise_task: bool = False, dropout: float = 0):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototype_vectors = nn.Parameter(torch.randn(self.num_prototypes, dim, 1, 1) * 0.02)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.num_prototypes, num_classes)
        self.frame_wise_task = frame_wise_task

    def _binarise(self, x: Tensor) -> Tensor:
        """Applies binarization to prototypes using a straight-through estimator for gradients."""
        x_sign = (x >= 0).float() * 2.0 - 1.0  # +1/-1
        grad_pass = x - x.detach()  # straight-through estimator
        return x_sign + grad_pass

    def forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Computes similarities against binarized prototypes."""
        x = feat_cache['patch_tokens'][-1]  # B, D, T, F
        batch_size, feat_dim, time_patch, freq_patch = x.shape
        proto_bin = self._binarise(self.prototype_vectors)
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(proto_bin, dim=1)
        act = F.conv2d(x_norm, p_norm)  # (B, P, T, F)
        if self.frame_wise_task:
            act = act.transpose(1, 2)  # (B, T, P, F)
        else:
            act = act.reshape(batch_size, self.num_prototypes, -1)  # (B, P, T*F)
        act = act.max(dim=-1).values  # (B, T, P) if frame_wise_task else (B, P)
        return self.fc(self.dropout(act))


class ASR_Head(nn.Module):
    """this is for the Automatic Speech Recognition on LibriSpeech."""

    def __init__(self,
                 num_classes: int = 29,
                 dim: int = 768,
                 num_layers: int = 12,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 upsample_factor: int = 8,
                 dropout: float = 0):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.upsample = nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=self.upsample_factor,
                                           stride=self.upsample_factor)

        self.block_attention = nn.Parameter(torch.ones(num_layers))
        self.lstm = nn.LSTM(input_size=dim, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.2)
        self.ln = norm_layer(1024) if norm_layer else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes, bias=True)

    @torch.compiler.disable
    def forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        x = feat_cache['patch_tokens']  # L, B, D, T, F
        a = torch.softmax(self.block_attention, dim=0).reshape(-1, 1, 1, 1, 1)  # (L, 1, 1, 1, 1)
        x = x * a
        x = x.sum(dim=0)  # B, D, T, F

        x = x.mean(dim=3)  # B, D, T
        x = self.upsample(x)
        x = x.transpose(1, 2)  # B, T, D

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            x, _ = self.lstm(x)

        logits = self.fc(self.dropout(self.ln(x)))  # B, T, C
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # T, B, C

        return log_probs


class Head2Toe(nn.Module):

    def __init__(self,
                 num_classes: Optional[int] = None,
                 dim: int = 768,
                 num_layers: int = 12,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 use_cls: bool = True,
                 frame_wise_task: bool = False,
                 layer_wise_lasso: bool = False,
                 dropout: float = 0):

        super().__init__()
        self.layer_wise_lasso = layer_wise_lasso
        self.group_size = dim if layer_wise_lasso else num_classes
        self.register_buffer('group_scale', torch.tensor(self.group_size, dtype=torch.float32).sqrt())
        self.ln = norm_layer(dim * num_layers) if norm_layer else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim * num_layers, num_classes, bias=True)
        if frame_wise_task:
            self.forward_fn = self.frame_wise_forward
        else:
            if use_cls:
                self.forward_fn = self.cls_forward
            else:
                self.forward_fn = self.patch_avg_forward

    def cls_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass flattening and concatenating class tokens across all layers."""
        x = feat_cache['cls_tokens']  # L, B, D
        x = x.permute(1, 0, 2).flatten(1)  # B, L*D
        return self.fc(self.dropout(self.ln(x)))

    def patch_avg_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass concatenating the global average of patch tokens across all layers."""
        x = feat_cache['patch_tokens']  # L, B, D, T, F
        x = x.flatten(3)  # L, B, D, N=T*F
        x = x.mean(dim=3)  # L, B, D
        x = x.permute(1, 0, 2).flatten(1)  # B, L*D
        return self.fc(self.dropout(self.ln(x)))

    def frame_wise_forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        """Forward pass concatenating frame-wise patch tokens across all layers."""
        x = feat_cache['patch_tokens']  # L, B, D, T, F
        x = x.mean(dim=4)  # L, B, D, T
        x = x.permute(1, 3, 0, 2)  # B, T, L, D
        x = x.flatten(2)  # B, T, L*D
        return self.fc(self.dropout(self.ln(x)))

    def forward(self, feat_cache: Dict[str, Any]) -> Tensor:
        return self.forward_fn(feat_cache)

    def compute_group_lasso_penalty(self) -> Tensor:
        """Calculates the group lasso penalty for feature selection regularization."""
        if self.layer_wise_lasso:
            groups = torch.split(self.fc.weight, self.group_size, dim=1)
            penalty = 0.0
            for w_g in groups:
                penalty += torch.linalg.matrix_norm(w_g + 1e-8)
        else:
            penalty = torch.linalg.vector_norm(self.fc.weight, dim=0).sum()
        return penalty * self.group_scale
