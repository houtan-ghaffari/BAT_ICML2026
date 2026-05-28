__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from functools import partial
from typing import Optional, Tuple, Dict, List, Union
import torch
from torch import nn
from .pos_embed import get_2d_sincos_pos_embed_flexible
from .vit import PatchEmbed, EncoderBlock


class ViT_MaskedDecoder(nn.Module):
    def __init__(self, num_time_patches: int = 64, num_freq_patches: int = 8, in_dim: int = 768, dim: Optional[int] = None, depth: int = 6, num_heads: int = 12,
                 mlp_ratio: int = 4, qkv_bias: bool = True, pos_trainable: bool = True, layer_norm_first: bool = False, init_scale_values: Optional[float] = None,
                 use_gate: bool = True):
        super().__init__()
        dim = dim or in_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_patches = int(num_time_patches * num_freq_patches)
        pos_embed = get_2d_sincos_pos_embed_flexible(dim, (num_freq_patches, num_time_patches), cls_token=False)
        self.pos_embed = nn.Parameter(torch.tensor(pos_embed).float().unsqueeze(0), requires_grad=pos_trainable)
        self.blocks = nn.ModuleList([EncoderBlock(dim=dim,
                                                  num_heads=num_heads,
                                                  mlp_ratio=mlp_ratio,
                                                  qkv_bias=qkv_bias,
                                                  act_layer=nn.GELU,
                                                  norm_layer=norm_layer,
                                                  init_scale_values=init_scale_values,
                                                  layer_norm_first=layer_norm_first,
                                                  use_gate=use_gate) for i in range(depth)])
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, dim, bias=False) if in_dim != dim else nn.Identity()
        self.out_proj = nn.Linear(dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor, mask_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.in_proj(x)
        bsz, num_kept_tokens, fsz = x.shape
        index = mask_info['ids_keep_sorted']  # B, n
        index = index.unsqueeze(2).expand(-1, -1, fsz).to(x.device, non_blocking=True)  # B, n, D
        assert index.shape[1] == num_kept_tokens
        # x_decoder = self.mask_token.expand(bsz, self.num_patches, -1)
        x_decoder = self.mask_token.repeat(bsz, self.num_patches, 1)
        x_decoder = torch.scatter(x_decoder, 1, index, x)  # B, N, D
        x = x_decoder + self.pos_embed
        for blk in self.blocks:
            x, _ = blk(x)
        return self.out_proj(x)


class ViT_MaskedEncoder(nn.Module):
    def __init__(self,
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
                 mode: str = 'student',
                 layer_norm_first: bool = False,
                 pre_norm: bool = True,
                 init_scale_values: Optional[float] = None,
                 use_gate: bool = True,
                 ):
        super().__init__()
        assert mode in ['student', 'teacher']
        assert (input_shape[0] % patch_size[0]) == (input_shape[1] % patch_size[1]) == 0
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.forward_fn = self.student_forward if mode == 'student' else self.teacher_forward
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_shape, patch_size, 1, dim)
        pos_embed = get_2d_sincos_pos_embed_flexible(dim, self.patch_embed.patch_ft, cls_token=False)
        self.pos_embed = nn.Parameter(torch.tensor(pos_embed).float().unsqueeze(0), requires_grad=pos_trainable)
        if pre_norm:
            assert not layer_norm_first, "avoid using two consecutive norm layer on the input"
            self.pre_norm = norm_layer(dim)
        else:
            self.pre_norm = nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        dpr = [d.item() for d in torch.linspace(0, drop_path_rate, depth)] if drop_path_rate > 0 else [0] * depth
        self.blocks = nn.ModuleList([EncoderBlock(dim=dim,
                                                  num_heads=num_heads,
                                                  mlp_ratio=mlp_ratio,
                                                  qkv_bias=qkv_bias,
                                                  drop=drop,
                                                  attn_drop=attn_drop,
                                                  drop_path=dpr[i],
                                                  act_layer=nn.GELU,
                                                  norm_layer=norm_layer,
                                                  init_scale_values=init_scale_values,
                                                  layer_norm_first=layer_norm_first,
                                                  use_gate=use_gate) for i in range(depth)])

    def student_forward(self, x: torch.Tensor, mask_info: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Student forward pass taking a subset of unmasked tokens."""
        x = self.patch_embed(x)  # B, 1, T, F -> B, N, D
        x = x + self.pos_embed
        if mask_info is not None:
            index = mask_info['ids_keep_sorted'].unsqueeze(-1).expand(-1, -1, x.shape[2])
            index = index.to(x.device, non_blocking=True)
            x = torch.gather(x, dim=1, index=index)  # B, N * (1 - mask_ratio), D
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pre_norm(x)
        for blk in self.blocks:
            x, _ = blk(x)
        cls_tokens = x[:, 0]
        patch_tokens = x[:, 1:]
        return cls_tokens, patch_tokens

    def teacher_forward(self, x: torch.Tensor, mask_info: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Teacher forward pass without masking, returning representations from all layers."""
        x = self.patch_embed(x)  # B, 1, T, F -> B, N, D
        x = x + self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pre_norm(x)
        features = []  # accumulate patch tokens
        for blk in self.blocks:
            x, _ = blk(x)
            features.append(x[:, 1:, :])
        cls_tokens = x[:, 0]
        return cls_tokens, features

    def forward(self, x: torch.Tensor, mask_info: Optional[Dict[str, torch.Tensor]] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        return self.forward_fn(x, mask_info)


class MLR_Student(nn.Module):

    def __init__(self,
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
                 pre_norm: bool = True,
                 init_scale_values: Optional[float] = None,
                 use_gate: bool = True,
                 decoder_depth: int = 6,
                 decoder_heads: int = 12,
                 decoder_mlp_ratio: int = 4,
                 decoder_dim: Optional[int] = None,
                 ):

        super().__init__()
        self.encoder = ViT_MaskedEncoder(input_shape=input_shape, patch_size=patch_size, dim=dim, depth=depth,
                                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                         attn_drop=attn_drop, drop_path_rate=drop_path_rate,
                                         pos_trainable=pos_trainable,
                                         mode='student', layer_norm_first=layer_norm_first, pre_norm=pre_norm,
                                         init_scale_values=init_scale_values, use_gate=use_gate)

        num_freq_patches, num_time_patches = self.encoder.patch_embed.patch_ft
        self.decoder = ViT_MaskedDecoder(num_time_patches=num_time_patches, num_freq_patches=num_freq_patches,
                                         in_dim=dim, dim=decoder_dim, depth=decoder_depth, num_heads=decoder_heads,
                                         mlp_ratio=decoder_mlp_ratio, qkv_bias=True, pos_trainable=True,
                                         layer_norm_first=False, init_scale_values=None, use_gate=use_gate)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask_info: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes partial inputs and reconstructs full representations using the decoder."""
        cls_tokens, patch_tokens = self.encoder(x, mask_info)
        patch_tokens = self.decoder(patch_tokens, mask_info)
        return cls_tokens, patch_tokens


class MLR_Teacher(nn.Module):

    def __init__(self,
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
                 pre_norm: bool = True,
                 init_scale_values: Optional[float] = None,
                 use_gate: bool = True,
                 instance_norm_target_layer: bool = True,
                 layer_norm_targets: bool = True,
                 ):

        super().__init__()
        self.encoder = ViT_MaskedEncoder(input_shape=input_shape, patch_size=patch_size, dim=dim, depth=depth,
                                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                         attn_drop=attn_drop, drop_path_rate=drop_path_rate,
                                         pos_trainable=pos_trainable,
                                         mode='teacher', layer_norm_first=layer_norm_first, pre_norm=pre_norm,
                                         init_scale_values=init_scale_values, use_gate=use_gate)
        self.instance_norm_target_layer = instance_norm_target_layer
        self.layer_norm_targets = layer_norm_targets

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes reference class and patch targets without applying a mask."""
        cls_tokens, patch_tokens = self.encoder(x)
        patch_tokens = self.make_targets(patch_tokens)
        return cls_tokens, patch_tokens

    def make_targets(self, y: List[torch.Tensor]) -> torch.Tensor:
        """Aggregates and normalizes representations across the network's layers."""
        y_stacked = torch.stack(y)  # L=layer, B=batch, N=patch, D=dim
        if self.instance_norm_target_layer:
            y_stacked = (y_stacked - y_stacked.mean(dim=2, keepdim=True)) / (y_stacked.var(dim=2, correction=0, keepdim=True) + 1e-6).sqrt()
        y_stacked = y_stacked.mean(dim=0)  # average layers' outputs: B, N, D
        if self.layer_norm_targets:
            y_stacked = (y_stacked - y_stacked.mean(dim=2, keepdim=True)) / (y_stacked.var(dim=2, correction=0, keepdim=True) + 1e-6).sqrt()
        return y_stacked