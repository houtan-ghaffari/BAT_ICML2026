"I adapted this from the original BEATs implementation: https://github.com/microsoft/unilm/blob/master/beats/BEATs.py"

__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import math
import random
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from .heads import LinearHead, LinearCGP, Protobin, CGP, ASR_Head, Head2Toe


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLULinear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super(GLULinear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2])
        else:
            x = (x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2]))

        return x


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
            0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            has_relative_attention_bias=False,
            num_buckets=32,
            max_distance=128,
            gru_rel_pos=False,
            rescale_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = nn.Linear(self.kdim, k_embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, q_embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
                torch.log(relative_positions.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position,
            bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
            self,
            query,
            key=None,
            value=None,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            before_softmax=False,
            need_head_weights=False,
            position_bias=None,
            num_vqt=0):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)

        attn_pos_bias = position_bias  # local copy for slicing
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
            if num_vqt > 0:
                k = k[num_vqt:, :, :]
                v = v[num_vqt:, :, :]
                src_len = src_len - num_vqt
                if attn_pos_bias is not None:
                    attn_pos_bias = attn_pos_bias[:, :, num_vqt:]  # safe localized slice

        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.q_head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.k_head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]) * alpha

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            attn_mask_rel_pos = attn_pos_bias
            if self.gru_rel_pos:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim) * alpha / self.scaling
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(
                    _B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * attn_pos_bias

            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())

            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(
            attn_weights, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(self, embedding_dim=768, ffn_embedding_dim=3072, num_attention_heads=12, dropout=0.,
                 attention_dropout=0., activation_dropout=0., activation_fn="gelu",
                 layer_norm_first=False, deep_norm=False, has_relative_attention_bias=False, num_buckets=0,
                 max_distance=0, rescale_init=False, gru_rel_pos=False, encoder_layers=0):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLULinear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.deep_norm = deep_norm
        if self.deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            pos_bias=None,
            num_vqt=0
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
                num_vqt=num_vqt
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
                num_vqt=num_vqt
            )

            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


class TransformerEncoder(nn.Module):
    def __init__(self, dropout=0.0, attention_dropout=0., activation_dropout=0.0, encoder_layerdrop=0.,
                 activation_fn='gelu', layer_norm_first=False, deep_norm=True, gru_rel_pos=True, encoder_layers=12,
                 encoder_embed_dim=768, conv_pos=128, conv_pos_groups=16, relative_position_embedding=True,
                 num_buckets=320, max_distance=800, encoder_ffn_embed_dim=3072, encoder_attention_heads=12):
        super().__init__()

        self.dropout = dropout
        self.embedding_dim = encoder_embed_dim
        self.pos_conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=conv_pos, padding=conv_pos // 2,
                                  groups=conv_pos_groups)
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            self.num_buckets = num_buckets
            self.max_distance = max_distance
        else:
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList([TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim,
                                                                     ffn_embedding_dim=encoder_ffn_embed_dim,
                                                                     num_attention_heads=encoder_attention_heads,
                                                                     dropout=dropout,
                                                                     attention_dropout=attention_dropout,
                                                                     activation_dropout=activation_dropout,
                                                                     activation_fn=activation_fn,
                                                                     layer_norm_first=layer_norm_first,
                                                                     deep_norm=deep_norm,
                                                                     has_relative_attention_bias=self.relative_position_embedding,
                                                                     num_buckets=self.num_buckets,
                                                                     max_distance=self.max_distance,
                                                                     gru_rel_pos=gru_rel_pos,
                                                                     encoder_layers=encoder_layers,
                                                                     ) for _ in range(encoder_layers)])
        if self.relative_position_embedding:
            for i in range(1, encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[0].self_attn.relative_attention_bias

        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = encoder_layerdrop

        self.apply(init_bert_params)

        if deep_norm:
            deep_norm_beta = math.pow(8 * encoder_layers, -1 / 4)
            for i in range(encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.out_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain=deep_norm_beta)

    def forward(self, x, num_vqt=0, vqt=None, vqt_dropout=None):
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv  # B, N, D
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # B x N x D -> N x B x D
        layer_results = []
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                if num_vqt > 0:
                    q_states = vqt_dropout(vqt[i].expand(x.shape[1], -1, -1)).transpose(0, 1)
                    x = torch.cat((q_states, x), dim=0)
                x, a, pos_bias = layer(x, pos_bias=pos_bias, num_vqt=num_vqt)
                if num_vqt > 0:
                    layer_results.append(x[:num_vqt, :, :].transpose(0, 1))
                    x = x[num_vqt:, :, :]
            else:
                if num_vqt > 0:
                    layer_results.append(torch.zeros(x.shape[1], num_vqt, x.shape[2], device=x.device))

            if num_vqt == 0:
                layer_results.append(x.transpose(0, 1))  # T x B x C -> B x T x C

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x, layer_results


class BEATs(nn.Module):
    def __init__(self,
                 num_classes=None,
                 head=None,  # linear, linear_cgp, cgp, protobin, asr (for librispeech), h2t (head to toe)
                 num_prototypes=10000,
                 frame_wise_task=False,  # for dense prediction tasks
                 use_cls_in_head=True,  # use patch average or cls in the relevant classification heads
                 head_dropout: float=0,  # this is to replicate SUPERB probing method
                 head_norm=False,
                 num_vqt=1,
                 vqt_dropout=0.0,
                 normalize_vqt=True,
                 ):
        super().__init__()
        self.encoder_embed_dim = 768
        self.embed = 512
        dropout_input = 0.
        deep_norm = True
        layer_norm_first = False
        self.post_extract_proj = nn.Linear(self.embed,
                                           self.encoder_embed_dim) if self.embed != self.encoder_embed_dim else None

        self.input_patch_size = 16
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size,
                                         bias=False)
        self.dropout_input = nn.Dropout(dropout_input)

        assert not deep_norm or not layer_norm_first
        self.encoder = TransformerEncoder()
        self.layer_norm = LayerNorm(self.embed)

        head_layer_norm = nn.LayerNorm if head_norm else None

        self.head_name = head
        depth = len(self.encoder.layers)
        dim = self.encoder_embed_dim

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

        elif head == 'asr':  # Automatic Speech Recognition probing uses this head
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
            patch_area = self.input_patch_size * self.input_patch_size
            val = math.sqrt(6.0 / float(3 * patch_area + dim))
            self.vqt = nn.Parameter(torch.zeros(depth, self.num_vqt, dim))
            nn.init.uniform_(self.vqt.data, -val, val)
            num_features = int(depth * num_vqt * dim + dim)
            if head_norm:
                self.head = nn.Sequential(head_layer_norm(num_features), nn.Dropout(head_dropout),
                                          nn.Linear(num_features, num_classes, bias=True))
            else:
                self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(num_features, num_classes, bias=True))

        elif head is None:
            self.head = nn.Identity()
        else:
            raise ValueError(f"Head type {head} not supported")

        if (head in ['linear', 'linear_cgp', 'h2t']) and not frame_wise_task and use_cls_in_head:
            self.extract_features = self.extract_cls_features
        elif head == 'vqt':
            self.extract_features = self.extract_vqt_features
        else:
            self.extract_features = self.extract_features_with_patch_tokens

    def forward(self, x, mask_info=None):
        feat_cache = self.extract_features(x, mask_info)
        return self.head(feat_cache)

    def extract_cls_features(self, x, mask_info=None):
        cache = {'cls_tokens': []}
        x = self.patch_embedding(x)
        b, c, t, f = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2)
        x = self.layer_norm(x)

        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)

        x = x.reshape(b, t, f, -1)
        chunks = torch.split(x, 64, dim=1)

        cls_chunks = []
        for xi in chunks:
            xi = xi.flatten(1, 2)
            xi, patch_tokens = self.encoder(xi)
            patch_tokens = torch.stack(patch_tokens)  # L, B, N, D
            cls_chunk = patch_tokens.mean(dim=2)  # L, B, D
            cls_chunks.append(cls_chunk)

        cache['cls_tokens'] = torch.stack(cls_chunks).mean(dim=0)  # L, B, D

        return cache

    def extract_features_with_patch_tokens(self, x, mask_info=None):
        cache = {'patch_tokens': [], 'cls_tokens': []}
        x = self.patch_embedding(x)
        b, c, t, f = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2)
        x = self.layer_norm(x)

        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)

        x = x.reshape(b, t, f, -1)
        chunks = torch.split(x, 64, dim=1)

        cls_chunks = []
        patch_chunks = []

        for xi in chunks:
            chunk_t = xi.shape[1]  # Dynamically capture time (might be <64 for last chunk)
            xi = xi.flatten(1, 2)

            xi, layer_results = self.encoder(xi)
            layer_tokens = torch.stack(layer_results)  # L, B, N, D

            # Store CLS representation
            cls_chunks.append(layer_tokens.mean(dim=2))  # L, B, D

            # Reshape patch tokens back to 2D for this specific chunk
            l, b_out, n, d = layer_tokens.shape
            spatial_chunk = layer_tokens.transpose(2, 3).reshape(l, b_out, d, chunk_t, f)
            patch_chunks.append(spatial_chunk)

        # stitch the chunks back together along the Time dimension (dim=3)
        cache['patch_tokens'] = torch.cat(patch_chunks, dim=3)  # L, B, D, T, F
        cache['cls_tokens'] = torch.stack(cls_chunks).mean(dim=0)  # L, B, D

        return cache

    def extract_vqt_features(self, x, mask_info=None):
        cache = {'cls_tokens': [], 'query_tokens': []}
        x = self.patch_embedding(x)
        b, c, t, f = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2)
        x = self.layer_norm(x)

        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)

        x = x.reshape(b, t, f, -1)
        chunks = torch.split(x, 64, dim=1)

        cls_chunks = []
        query_chunks = []

        for xi in chunks:
            xi = xi.flatten(1, 2)

            # xi output is (N, B, D), query_tokens list contains (B, q, D)
            xi, query_tokens = self.encoder(xi, num_vqt=self.num_vqt, vqt=self.vqt, vqt_dropout=self.vqt_dropout)

            q_tokens = torch.stack(query_tokens)  # L, B, q, D
            query_chunks.append(q_tokens)

            # CORRECTED: dim=0 averages across the Sequence length (N)
            cls_chunks.append(xi.mean(dim=0))

        # Average CLS and VQT across all time chunks
        cls_tokens = torch.stack(cls_chunks).mean(dim=0)  # B, D
        query_tokens = torch.stack(query_chunks).mean(dim=0)  # L, B, q, D
        query_tokens = query_tokens.flatten(2)  # L, B, q*D

        if self.normalize_vqt:
            cls_tokens = F.normalize(cls_tokens, dim=-1)
            query_tokens = F.normalize(query_tokens, dim=-1)

        query_tokens = query_tokens.transpose(0, 1)  # B, L, q*D
        query_tokens = query_tokens.flatten(1)  # B, L*q*D
        return torch.cat([cls_tokens, query_tokens], dim=1)  # B, D + L*q*D