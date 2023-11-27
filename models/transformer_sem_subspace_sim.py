import copy
import math
import torch
from torch import nn
from torch.nn import Module, Dropout
from models.position_encoding import VolumetricPositionEncoding as VolPE
from models.matching import Matching
from models.procrustes import SoftProcrustesLayer
import numpy as np
import random
from scipy.spatial.transform import Rotation

from models.semantic_subspace_attention import SemanticSubspaceAttentionLayer
from lib.utils import pad_sequence, unpad_sequences
from models.vanilla_transformer_atf import TransformerLayer
from einops import rearrange


class GeometryAttentionLayer(nn.Module):

    def __init__(self, config):

        super(GeometryAttentionLayer, self).__init__()

        d_model = config['feature_dim']
        nhead =  config['n_head']

        self.dim = d_model // nhead
        self.nhead = nhead
        self.pe_type = config['pe_type']
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

    def forward(self, x, source, x_pe, source_pe, x_mask=None, source_mask=None):

        bs = x.size(0)
        q, k, v = x, source, source
        qp, kvp  = x_pe, source_pe
        q_mask, kv_mask = x_mask, source_mask

        if self.pe_type == 'sinusoidal':
            #w(x+p), attention is all you need : https://arxiv.org/abs/1706.03762
            if qp is not None: # disentangeld
                q = q + qp
                k = k + kvp
            qw = self.q_proj(q).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            kw = self.k_proj(k).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            vw = self.v_proj(v).view(bs, -1, self.nhead, self.dim)

        elif self.pe_type == 'rotary':
            #Rwx roformer : https://arxiv.org/abs/2104.09864

            qw = self.q_proj(q)
            kw = self.k_proj(k)
            vw = self.v_proj(v)

            if qp is not None: # disentangeld
                q_cos, q_sin = qp[...,0] ,qp[...,1]
                k_cos, k_sin = kvp[...,0],kvp[...,1]
                qw = VolPE.embed_rotary(qw, q_cos, q_sin)
                kw = VolPE.embed_rotary(kw, k_cos, k_sin)

            qw = qw.view(bs, -1, self.nhead, self.dim)
            kw = kw.view(bs, -1, self.nhead, self.dim)
            vw = vw.view(bs, -1, self.nhead, self.dim)

        else:
            raise KeyError()

        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)
        if kv_mask is not None:
            a.masked_fill_( q_mask[:, :, None, None] * (~kv_mask[:, None, :, None]), float('-inf'))
        a =  a / qw.size(3) **0.5
        a = torch.softmax(a, dim=2)
        o = torch.einsum("nlsh,nshd->nlhd", a, vw).contiguous()  # [N, L, (H, D)]

        message = self.merge(o.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        e = x + message

        return e




class SemanticTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.d_model = config['feature_dim']
        self.nhead = config['n_head']
        self.layer_types = config['layer_types']


        encoder_layer = GeometryAttentionLayer(config)
        semantic_encoder_layer = TransformerLayer(config)

        self.dino_feat_proj = nn.Linear(384, self.d_model, bias=False)
        
        self.layers = nn.ModuleList()

        for l_type in self.layer_types:

            if l_type in ['self']:

                self.layers.append( copy.deepcopy(encoder_layer))

            elif l_type == "cross":
                self.layers.append( copy.deepcopy(semantic_encoder_layer))

            else:
                raise KeyError()

        self._reset_parameters()



    def forward(self, 
                src_feat, tgt_feat, 
                s_pcd, t_pcd, 
                src_mask, tgt_mask, src_2dfeat, tgt_2dfeat,
                data, timers = None):

        self.timers = timers

        assert self.d_model == src_feat.size(2), "the feature number of src and transformer must be equal"

        # src_pe = self.positional_encoding(s_pcd)
        # tgt_pe = self.positional_encoding(t_pcd)

        # src_feat = VolPE.embed_pos(self.pe_type, src_feat, src_pe)
        # tgt_feat = VolPE.embed_pos(self.pe_type, tgt_feat, tgt_pe)
        feat_sim = torch.einsum("bnd,bmd->bnm", src_2dfeat, tgt_2dfeat)
            
        src_2dfeat = self.dino_feat_proj(src_2dfeat)
        tgt_2dfeat = self.dino_feat_proj(tgt_2dfeat)
        
        for layer, name in zip(self.layers, self.layer_types):
            if name == 'self':
                src_feat = src_feat + src_2dfeat
                tgt_feat = tgt_feat + tgt_2dfeat
                if self.timers: self.timers.tic('self atten')
                src_feat = layer(src_feat, src_feat, None, None, src_mask, src_mask, )
                tgt_feat = layer(tgt_feat, tgt_feat, None, None, tgt_mask, tgt_mask)
                if self.timers: self.timers.toc('self atten')
            elif name == 'cross':
                if self.timers: self.timers.tic('cross atten')
                src_feat = layer(src_feat, tgt_feat, feat_sim, src_mask, tgt_mask)
                tgt_feat = layer(tgt_feat, src_feat, feat_sim.transpose(1,2), tgt_mask, src_mask)
                if self.timers: self.timers.toc('cross atten')

        return src_feat, tgt_feat, feat_sim


    def forward(self, 
                src_feat, tgt_feat, 
                s_pcd, t_pcd, 
                src_mask, tgt_mask, src_2dfeat, tgt_2dfeat,
                data, timers = None):

        self.timers = timers

        assert self.d_model == src_feat.size(2), "the feature number of src and transformer must be equal"

        # src_pe = self.positional_encoding(s_pcd)
        # tgt_pe = self.positional_encoding(t_pcd)

        # src_feat = VolPE.embed_pos(self.pe_type, src_feat, src_pe)
        # tgt_feat = VolPE.embed_pos(self.pe_type, tgt_feat, tgt_pe)
        feat_sim = torch.einsum("bnd,bmd->bnm", src_2dfeat, tgt_2dfeat)
        feat_sim_thr = 0.0
        s2t_close_feat = feat_sim
        s2t_close_feat[feat_sim < feat_sim_thr] = 0
        src_dists = torch.exp(100*-torch.cdist(s_pcd, s_pcd))
        tgt_dists = torch.exp(100*-torch.cdist(t_pcd, t_pcd))
        
        s2t_subspace = torch.einsum("bst,btm->bsm", s2t_close_feat, tgt_dists)
        s2t_subspace = s2t_subspace / (s2t_subspace.max(-1).values.T+1e-12)
        data["s2t_subspace"] = s2t_subspace

        t2s_subspace = torch.einsum("bst,bsm->btm", s2t_close_feat, src_dists)
        t2s_subspace = t2s_subspace / (t2s_subspace.max(-1).values.T+1e-12)
        data["t2s_subspace"] = t2s_subspace
            
        src_2dfeat = self.dino_feat_proj(src_2dfeat)
        tgt_2dfeat = self.dino_feat_proj(tgt_2dfeat)
        
        for layer, name in zip(self.layers, self.layer_types):
            if name == 'self':
                src_feat = src_feat + src_2dfeat
                tgt_feat = tgt_feat + tgt_2dfeat
                if self.timers: self.timers.tic('self atten')
                src_feat = layer(src_feat, src_feat, None, None, src_mask, src_mask, )
                tgt_feat = layer(tgt_feat, tgt_feat, None, None, tgt_mask, tgt_mask)
                if self.timers: self.timers.toc('self atten')
            elif name == 'cross':
                if self.timers: self.timers.tic('cross atten')
                src_feat = layer(src_feat, tgt_feat, s2t_subspace, src_mask, tgt_mask)
                tgt_feat = layer(tgt_feat, src_feat, t2s_subspace, tgt_mask, src_mask)
                if self.timers: self.timers.toc('cross atten')

        return src_feat, tgt_feat, feat_sim

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)