__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import math
from functools import partial
from typing import Optional, Tuple, Dict, Any, Type

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .pos_embed import get_2d_sincos_pos_embed_flexible
from .heads import LinearHead, LinearCGP, Protobin, CGP, ASR_Head, Head2Toe


class LayerScale(nn.Module):
    """Applies per-channel learnable scaling to inputs."""
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.s = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.s


class DropPath(nn.Module):
    """Drop paths per sample when applied in main path of residual blocks."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PatchEmbed(nn.Module):
    """Converts a 2D audio spectrogram into a sequence of flattened 1D patch embeddings."""
    def __init__(self, input_size: Tuple[int, int] = (1024, 128), patch_size: Tuple[int, int] = (16, 16), in_chans: int = 1, dim: int = 768):
        super().__init__()
        assert isinstance(input_size, tuple) and isinstance(patch_size, tuple)
        self.patch_size = patch_size
        self.num_patches = (input_size[1] // patch_size[1]) * (input_size[0] // patch_size[0])  # 512
        self.patch_ft = (input_size[1] // patch_size[1],
                         input_size[0] // patch_size[0])  # number of patches freq,time = 8,64
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # B, C=1, T=1024, F=128 -> B, 768, 64, 8
        x = x.flatten(2)  # B, 768, 64, 8 -> B, 768, 512
        x = x.transpose(1, 2)  # B, 768, 512 -> B, N=512, D=768
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: Type[nn.Module] = nn.GELU, norm_layer: Optional[Type[nn.Module]] = None,
                 bias: bool = True, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-Head Self Attention (MHSA) mechanism with optional gating and VQT support."""
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True, proj_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0., use_gate: bool = True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate = nn.Linear(dim, dim) if use_gate else None

    def forward(self, x: Tensor, num_vqt: int = 0) -> Tuple[Tensor, Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 3, B, H, N, D
        q, k, v = qkv.unbind(0)  # B, H, N, D for each
        k = k[:, :, num_vqt:, :]
        v = v[:, :, num_vqt:, :]

        attn = (q * self.scale) @ k.transpose(-2, -1)  # B, H, N, D @ B, H, D, N' -> B, H, N, N'
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = attn @ v  # B, H, N, N' @ B, H, N', D -> B, H, N, D
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        if self.gate:
            g = self.gate(x).sigmoid()
            attn_out = attn_out * g
        x = self.proj(attn_out)
        x = self.proj_drop(x)
        return x, attn


class EncoderBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 init_scale_values: Optional[float] = None,
                 layer_norm_first: bool = False,
                 use_gate: bool = True,
                 ):
        super().__init__()
        self.forward_fn = self.forward_norm_first if layer_norm_first else self.forward_norm_last
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_gate=use_gate)
        self.ls1 = LayerScale(dim, init_scale_values) if init_scale_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_scale_values) if init_scale_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor, num_vqt: int = 0) -> Tuple[Tensor, Tensor]:
        return self.forward_fn(x, num_vqt=num_vqt)

    def forward_norm_last(self, x: Tensor, num_vqt: int = 0) -> Tuple[Tensor, Tensor]:
        z, a = self.attn(x, num_vqt=num_vqt)
        z = self.ls1(z)
        z = self.drop_path1(z)
        z = x + z
        x = self.norm1(z)

        z = self.mlp(x)
        z = self.ls2(z)
        z = self.drop_path2(z)
        z = x + z
        x = self.norm2(z)
        return x, a

    def forward_norm_first(self, x: Tensor, num_vqt: int = 0) -> Tuple[Tensor, Tensor]:
        z = self.norm1(x)
        z, a = self.attn(z, num_vqt=num_vqt)
        z = self.ls1(z)
        z = self.drop_path1(z)
        x = x + z

        z = self.norm2(x)
        z = self.mlp(z)
        z = self.ls2(z)
        z = self.drop_path2(z)
        x = x + z
        return x, a


class ViT(nn.Module):
    def __init__(self,
                 num_classes: Optional[int] = None,
                 input_shape: Tuple[int, int] = (1024, 128),
                 patch_size: Tuple[int, int] = (16, 16),
                 dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path_rate: float = 0.,
                 pos_trainable: bool = False,
                 layer_norm_first: bool = False,
                 init_scale_values: Optional[float] = None,
                 pre_norm: bool = True,
                 use_gate: bool = True,
                 head: Optional[str] = None,
                 num_prototypes: int = 10000,
                 frame_wise_task: bool = False,
                 use_cls_in_head: bool = True,
                 head_dropout: float = 0,
                 head_norm: bool = False,
                 num_vqt: int = 1,
                 vqt_dropout: float = 0.0,
                 normalize_vqt: bool = True,
                 ):

        super().__init__()
        assert (input_shape[0] % patch_size[0]) == (input_shape[1] % patch_size[1]) == 0
        assert head in ['linear', 'linear_cgp', 'cgp', 'protobin', 'asr', 'h2t', 'vqt']
        self.head_name = head
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        head_layer_norm = norm_layer if head_norm else None
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_shape, patch_size, 1, dim)
        pos_embed = get_2d_sincos_pos_embed_flexible(dim, (8, 64), cls_token=False)
        self.pos_embed = nn.Parameter(torch.tensor(pos_embed).float().unsqueeze(0), requires_grad=pos_trainable)
        self.pre_norm = norm_layer(dim) if pre_norm else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02).contiguous()
        dpr = [d.item() for d in torch.linspace(0, drop_path_rate, depth)] if drop_path_rate > 0 else [0] * depth
        self.blocks = nn.ModuleList([EncoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                  drop=drop, attn_drop=attn_drop, drop_path=dpr[i], act_layer=nn.GELU,
                                                  norm_layer=norm_layer, layer_norm_first=layer_norm_first,
                                                  init_scale_values=init_scale_values, use_gate=use_gate)
                                     for i in range(depth)])

        if head == 'linear':
            self.head = LinearHead(num_classes=num_classes, dim=dim, norm_layer=head_layer_norm,
                                   use_cls=use_cls_in_head, frame_wise_task=frame_wise_task, dropout=head_dropout)

        elif head == 'linear_cgp':
            self.head = LinearCGP(num_classes=num_classes, dim=dim, norm_layer=head_layer_norm, use_cls=use_cls_in_head,
                                  frame_wise_task=frame_wise_task, num_layers=depth, dropout=head_dropout)

        elif head == 'protobin':
            self.head = Protobin(num_classes=num_classes, dim=dim, num_prototypes=num_prototypes,
                                 frame_wise_task=frame_wise_task, dropout=head_dropout)

        elif head == 'cgp':
            self.head = CGP(num_classes=num_classes, dim=dim, num_prototypes=num_prototypes, num_layers=depth,
                            frame_wise_task=frame_wise_task, use_cls=use_cls_in_head, dropout=head_dropout)

        elif head == 'asr':
            self.head = ASR_Head(dim=dim, num_classes=num_classes, num_layers=depth, norm_layer=head_layer_norm,
                                 upsample_factor=8, dropout=head_dropout)

        elif head == 'h2t':
            self.head = Head2Toe(num_classes=num_classes, dim=dim, num_layers=depth, norm_layer=head_layer_norm,
                                 use_cls=use_cls_in_head, frame_wise_task=frame_wise_task, dropout=head_dropout)

        elif head == 'vqt':
            assert num_vqt > 0
            self.num_vqt = num_vqt
            self.vqt_dropout = nn.Dropout(vqt_dropout)
            self.normalize_vqt = normalize_vqt
            patch_area = patch_size[0] * patch_size[1]
            val = math.sqrt(6.0 / float(3 * patch_area + dim))
            self.vqt = nn.Parameter(torch.zeros(depth, self.num_vqt, dim))
            nn.init.uniform_(self.vqt.data, -val, val)
            num_features = int(depth * num_vqt * dim + dim)
            if head_norm:
                self.head = nn.Sequential(norm_layer(num_features), nn.Linear(num_features, num_classes, bias=True))
            else:
                self.head = nn.Linear(num_features, num_classes, bias=True)
        else:
            raise ValueError(f"Head type {head} not supported")

        if (head in ['linear', 'linear_cgp', 'h2t']) and not frame_wise_task and use_cls_in_head:
            self.extract_features = self.extract_cls_features
        elif head == 'vqt':
            self.extract_features = self.extract_vqt_features
        else:
            self.extract_features = self.extract_features_with_patch_tokens

    def forward(self, x: Tensor, mask_info: Optional[Dict[str, Tensor]] = None) -> Tensor:
        feat_cache = self.extract_features(x, mask_info)
        return self.head(feat_cache)

    def extract_cls_features(self, x: Tensor, mask_info: Optional[Dict[str, Tensor]] = None) -> Dict[str, Any]:
        cache = {'cls_tokens': []}
        x = self.patch_embed(x)  # B, 1, T, F -> B, N, D

        max_token_size = self.pos_embed.shape[1]
        batch_size, num_tokens, emb_dim = x.shape
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        x = x.reshape(batch_size, -1, num_freq_patches, emb_dim)
        x = torch.split(x, 64, dim=1)
        chunks = []
        for xi in x:
            xi = xi.flatten(1, 2)  # B, t, D
            if xi.shape[1] < max_token_size and mask_info is not None:
                xi = F.pad(xi, (0, 0, 0, max_token_size - xi.shape[1]))
                xi = xi + self.pos_embed
                index = mask_info['ids_keep_sorted'].unsqueeze(-1).expand(-1, -1, xi.shape[2])
                index = index.to(xi.device, non_blocking=True)
                xi = torch.gather(xi, dim=1, index=index)
            else:
                xi = xi + self.pos_embed[:, :xi.shape[1], :]
            xi = torch.cat((self.cls_token.expand(batch_size, -1, -1), xi), dim=1)
            chunks.append(xi)

        for xi in chunks:
            cls_tokens = []
            xi = self.pre_norm(xi)
            for i, blk in enumerate(self.blocks):
                xi, *_ = blk(xi)
                cls_tokens.append(xi[:, 0])  # B, D
            cls_tokens = torch.stack(cls_tokens)  # L, B, D
            cache['cls_tokens'].append(cls_tokens)

        cache['cls_tokens'] = torch.stack(cache['cls_tokens']).mean(dim=0)  # L, B, D
        return cache

    def extract_features_with_patch_tokens(self, x: Tensor, mask_info: Optional[Dict[str, Tensor]] = None) -> Dict[str, Any]:
        cache = {'patch_tokens': [], 'cls_tokens': []}
        x = self.patch_embed(x)

        max_token_size = self.pos_embed.shape[1]
        batch_size, num_tokens, emb_dim = x.shape
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        x = x.reshape(batch_size, -1, num_freq_patches, emb_dim)
        x = torch.split(x, 64, dim=1)
        chunks = []
        for xi in x:
            xi = xi.flatten(1, 2)
            xi = xi + self.pos_embed[:, :xi.shape[1], :]
            xi = torch.cat((self.cls_token.expand(batch_size, -1, -1), xi), dim=1)
            chunks.append(xi)

        for xi in chunks:
            patch_tokens, cls_tokens = [], []
            xi = self.pre_norm(xi)
            for i, blk in enumerate(self.blocks):
                xi, *_ = blk(xi)
                cls_tokens.append(xi[:, 0])  # B, D
                patch_tokens.append(xi[:, 1:])  # B, N, D

            cls_tokens = torch.stack(cls_tokens)
            cache['cls_tokens'].append(cls_tokens)
            patch_tokens = torch.stack(patch_tokens)
            patch_tokens = patch_tokens.transpose(2, 3)
            patch_tokens = patch_tokens.reshape(len(self.blocks), batch_size, emb_dim, -1,
                                                num_freq_patches)
            cache['patch_tokens'].append(patch_tokens)

        cache['patch_tokens'] = torch.cat(cache['patch_tokens'], dim=3)
        assert cache['patch_tokens'].shape[3] == num_time_patches
        cache['cls_tokens'] = torch.stack(cache['cls_tokens']).mean(dim=0)

        return cache

    def extract_vqt_features(self, x: Tensor, mask_info: Optional[Dict[str, Tensor]] = None) -> Tensor:
        cache = {'cls_tokens': [], 'query_tokens': []}
        x = self.patch_embed(x)

        max_token_size = self.pos_embed.shape[1]
        batch_size, num_tokens, emb_dim = x.shape
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        x = x.reshape(batch_size, -1, num_freq_patches, emb_dim)
        x = torch.split(x, 64, dim=1)
        chunks = []
        for xi in x:
            xi = xi.flatten(1, 2)
            if xi.shape[1] < max_token_size and mask_info is not None:
                xi = F.pad(xi, (0, 0, 0, max_token_size - xi.shape[1]))
                xi = xi + self.pos_embed
                index = mask_info['ids_keep_sorted'].unsqueeze(-1).expand(-1, -1, xi.shape[2])
                index = index.to(xi.device, non_blocking=True)
                xi = torch.gather(xi, dim=1, index=index)
            else:
                xi = xi + self.pos_embed[:, :xi.shape[1], :]
            xi = torch.cat((self.cls_token.expand(batch_size, -1, -1), xi), dim=1)
            chunks.append(xi)

        for xi in chunks:
            query_tokens = []
            xi = self.pre_norm(xi)
            for i, blk in enumerate(self.blocks):
                q_states = self.vqt_dropout(self.vqt[i].expand(batch_size, -1, -1))
                xi = torch.cat((q_states, xi), dim=1)
                xi, *_ = blk(xi, num_vqt=self.num_vqt)
                query_tokens.append(xi[:, :self.num_vqt, :])  # B, q, D
                xi = xi[:, self.num_vqt:, :]
            query_tokens = torch.stack(query_tokens)
            cache['cls_tokens'].append(xi[:, 0])
            cache['query_tokens'].append(query_tokens)

        cls_tokens = torch.stack(cache['cls_tokens']).mean(dim=0)
        query_tokens = torch.stack(cache['query_tokens']).mean(dim=0)
        query_tokens = query_tokens.flatten(2)

        if self.normalize_vqt:
            cls_tokens = F.normalize(cls_tokens, dim=-1)
            query_tokens = F.normalize(query_tokens, dim=-1)

        query_tokens = query_tokens.transpose(0, 1)
        query_tokens = query_tokens.flatten(1)

        return torch.cat([cls_tokens, query_tokens], dim=1)