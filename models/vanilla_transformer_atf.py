r"""Vanilla Transformer without positional embeddings.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.factory import build_dropout_layer, AttentionOutput


class TransformerLayer(nn.Module):

    def __init__(self, config):

        super().__init__()

        d_model = config['feature_dim']
        nhead =  config['n_head']

        self.dim = d_model // nhead
        self.nhead = nhead
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.attention = Attention() #LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, attn_factor, x_mask=None, source_mask=None):

        bs = x.size(0)
        q, k, v = x, source, source
        q_mask, kv_mask = x_mask, source_mask

        qw = self.q_proj(q).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        kw = self.k_proj(k).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        vw = self.v_proj(v).view(bs, -1, self.nhead, self.dim)

        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)
        if kv_mask is not None:
            a.masked_fill_( q_mask[:, :, None, None] * (~kv_mask[:, None, :, None]), float('-inf'))
            
            # if ((q_mask.sum(1)==0).any() and (kv_mask.sum(1)==0).any()):
            #     pass
            
        a =  a / qw.size(3) **0.5
        a = torch.einsum("bnmh,bnm->bnmh", a, attn_factor)
        a = torch.softmax(a, dim=2)
        o = torch.einsum("nlsh,nshd->nlhd", a, vw).contiguous()  # [N, L, (H, D)]

        message = self.merge(o.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        e = x + message

        return e
