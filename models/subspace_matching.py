import torch
import torch.nn as nn
import torch.nn.functional as F
from models.position_encoding import VolumetricPositionEncoding as VolPE
from einops import rearrange


def log_optimal_transport(scores, alpha, iters, src_mask, tgt_mask ):

    b, m, n = scores.shape

    if src_mask is None:
        ms = torch.tensor(m)[None, None, ...].to(scores.device)
        ns = torch.tensor(n)[None, None, ...].to(scores.device)
    else :
        ms = src_mask.sum(dim=1, keepdim=True)
        ns = tgt_mask.sum(dim=1, keepdim=True)

    ms = ms.float()
    ns = ns.float()
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    Z = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log() # [b, 1]

    log_mu = torch.cat([norm.repeat(1, m), ns.log() + norm], dim=1)
    log_nu = torch.cat([norm.repeat(1, n), ms.log() + norm], dim=1)

    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp( Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    Z=  Z + u.unsqueeze(2) + v.unsqueeze(1)

    Z = Z - norm.view(-1,1,1)

    return Z


class SubspaceMatching(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.match_type = config['match_type']

        self.confidence_threshold = config['confidence_threshold']

        d_model = config['feature_dim']

        self.src_proj = nn.Linear(d_model, d_model, bias=False)
        self.tgt_proj = nn.Linear(d_model, d_model, bias=False)

        self.entangled= config['entangled']


        if self.match_type == "dual_softmax":
            self.temperature = config['dsmax_temperature']
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
        elif self.match_type == 'sinkhorn':
            #sinkhorn algorithm
            self.skh_init_bin_score = config['skh_init_bin_score']
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
            self.bin_score = nn.Parameter(
                torch.tensor( self.skh_init_bin_score,  requires_grad=True))
        else:
            raise NotImplementedError()


    @staticmethod
    @torch.no_grad()
    def get_match( conf_matrix, thr, mutual=True):

        mask = conf_matrix > thr

        #mutual nearest
        if mutual and conf_matrix.nelement():
            mask = mask \
                   * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                   * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        #find all valid coarse matches
        index = torch.nonzero(mask, as_tuple=False)
        b_ind, src_ind, tgt_ind = index[:,0], index[:,1], index[:,2]
        mconf = conf_matrix[b_ind, src_ind, tgt_ind]

        return index, mconf, mask

    @staticmethod
    @torch.no_grad()
    def get_topk_match( conf_matrix, thr, mutual=True):

        mask = conf_matrix > thr

        #mutual nearest
        if mutual:
            mask = mask \
                   * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                   * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        #find all valid coarse matches
        index = (mask==True).nonzero()
        b_ind, src_ind, tgt_ind = index[:,0], index[:,1], index[:,2]
        mconf = conf_matrix[b_ind, src_ind, tgt_ind]

        return index, mconf, mask


    def forward(self, src_feats, tgt_feats, src_mask, tgt_mask):
        '''
        @param src_feats: [B, S, M, C]
        @param tgt_feats: [B, S, N, C]
        @param src_mask: [B, S, M]
        @param tgt_mask: [B, S, N]
        @return:
        '''

        # Create hard semantic masks
        src_mask_hard = src_mask > 0.3
        tgt_mask_hard = tgt_mask > 0.3

        cross_semantic_mask = torch.einsum("bsm,bsn->bsmn", src_mask_hard.long(), tgt_mask_hard.long())

        src_mask_flat = rearrange(src_mask_hard, 'b s m -> (b s) m')
        tgt_mask_flat = rearrange(tgt_mask_hard, 'b s n -> (b s) n')
        cross_semantic_mask_flat = rearrange(cross_semantic_mask, 'b s m n -> (b s) m n')
        valid_sim_mask = cross_semantic_mask_flat.flatten(1).sum(1) > 0

        src_feats = self.src_proj(src_feats)
        tgt_feats = self.src_proj(tgt_feats)

        src_feats, tgt_feats = map(lambda feat: feat / feat.shape[-1] ** .5,
                                   [src_feats, tgt_feats])

        sim_matrix = torch.einsum("bsmc,bsnc->bsmn", src_feats, tgt_feats)


        if self.match_type == "dual_softmax":
            # dual softmax matching
            sim_matrix_1 = sim_matrix / self.temperature

            if src_mask is not None:
                sim_matrix_2 = sim_matrix_1.clone()
                sim_matrix_1.masked_fill_(~src_mask_hard[..., None], float('-inf'))
                sim_matrix_2.masked_fill_(~tgt_mask_hard[:, :, None, :], float('-inf'))
                conf_matrix = F.softmax(sim_matrix_1, 1) * F.softmax(sim_matrix_2, 2)
            else :
                conf_matrix = F.softmax(sim_matrix_1, 1) * F.softmax(sim_matrix_1, 2)

        elif self.match_type == "sinkhorn" :
            #optimal transport sinkhoron
            if src_mask is not None:
                sim_matrix.masked_fill_(cross_semantic_mask==0, float('-inf'))

            sim_matrix = rearrange(sim_matrix, 'b s m n -> (b s) m n')

            log_assign_matrix = log_optimal_transport(
                sim_matrix[valid_sim_mask], 
                self.bin_score, self.skh_iters, 
                src_mask_flat[valid_sim_mask], tgt_mask_flat[valid_sim_mask])

            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1].contiguous()

        coarse_match, _, _ = self.get_match(conf_matrix, self.confidence_threshold)
        
        # if conf_matrix.shape[0] != src_feats.shape[0]:
        #     conf_matrix = rearrange(conf_matrix, '(b s) m n -> b s m n', b=src_feats.shape[0])
            
        return conf_matrix, coarse_match, cross_semantic_mask, valid_sim_mask



class SubspaceMatchingV2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.match_type = config['match_type']

        self.confidence_threshold = config['confidence_threshold']

        d_model = config['feature_dim']

        self.src_proj = nn.Linear(d_model, d_model, bias=False)
        self.tgt_proj = nn.Linear(d_model, d_model, bias=False)

        self.entangled= config['entangled']


        if self.match_type == "dual_softmax":
            self.temperature = config['dsmax_temperature']
            self.bin_score = nn.Parameter(
                torch.tensor(1.0, requires_grad=True))
        elif self.match_type == 'sinkhorn':
            #sinkhorn algorithm
            self.skh_init_bin_score = config['skh_init_bin_score']
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
            self.bin_score = nn.Parameter(
                torch.tensor( self.skh_init_bin_score,  requires_grad=True))
        else:
            raise NotImplementedError()


    @staticmethod
    @torch.no_grad()
    def get_match( conf_matrix, thr, mutual=True):

        mask = conf_matrix > thr

        #mutual nearest
        if mutual and conf_matrix.nelement():
            mask = mask \
                   * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                   * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        #find all valid coarse matches
        index = torch.nonzero(mask, as_tuple=False)
        b_ind, src_ind, tgt_ind = index[:,0], index[:,1], index[:,2]
        mconf = conf_matrix[b_ind, src_ind, tgt_ind]

        return index, mconf, mask

    @staticmethod
    @torch.no_grad()
    def get_topk_match( conf_matrix, thr, mutual=True):

        mask = conf_matrix > thr

        #mutual nearest
        if mutual:
            mask = mask \
                   * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                   * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        #find all valid coarse matches
        index = (mask==True).nonzero()
        b_ind, src_ind, tgt_ind = index[:,0], index[:,1], index[:,2]
        mconf = conf_matrix[b_ind, src_ind, tgt_ind]

        return index, mconf, mask


    def forward(self, src_feats, tgt_feats, src_mask, tgt_mask, valid_matching_mask):
        '''
        @param src_feats: [B, S, C]
        @param tgt_feats: [B, T, C]
        @param src_mask: [B, S]
        @param tgt_mask: [B, T]
        @return:
        '''
        src_mask = src_mask[valid_matching_mask]
        tgt_mask = tgt_mask[valid_matching_mask]
        matching_mask = torch.einsum("bm,bn->bmn", src_mask, tgt_mask)

        src_feats = self.src_proj(src_feats)
        tgt_feats = self.src_proj(tgt_feats)

        src_feats, tgt_feats = map(lambda feat: feat / feat.shape[-1] ** .5,
                                   [src_feats, tgt_feats])

        if self.match_type == "dual_softmax":
            # dual softmax matching
            sim_matrix_1 = torch.einsum("bsc,btc->bst", src_feats, tgt_feats) / self.temperature

            if src_mask is not None:
                sim_matrix_2 = sim_matrix_1.clone()
                sim_matrix_1.masked_fill_(~src_mask[:, :, None], float('-inf'))
                sim_matrix_2.masked_fill_(~tgt_mask[:, None, :], float('-inf'))
                conf_matrix = F.softmax(sim_matrix_1, 1) * F.softmax(sim_matrix_2, 2)
            else :
                conf_matrix = F.softmax(sim_matrix_1, 1) * F.softmax(sim_matrix_1, 2)

        elif self.match_type == "sinkhorn" :
            #optimal transport sinkhoron
            sim_matrix = torch.einsum("bsc,btc->bst", src_feats, tgt_feats)

            if src_mask is not None:
                sim_matrix.masked_fill_(~(matching_mask).bool(), float('-inf'))

            log_assign_matrix = log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters, src_mask, tgt_mask)

            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1].contiguous()

        coarse_match, _, _ = self.get_match(conf_matrix, self.confidence_threshold)
        return conf_matrix, coarse_match


def subspace_matches_remap(subspace_matches, src_sub_ind_padded, tgt_sub_ind_padded):
    matches = []
    for i in range(src_sub_ind_padded.shape[0]):
        subspace_matches_batch = subspace_matches[subspace_matches[:, 0]==i]
        src_ind_sub = subspace_matches_batch[:, 1]
        tgt_ind_sub = subspace_matches_batch[:, 2]
        
        matches.append(
            torch.stack([
                torch.zeros_like(src_ind_sub),
                src_sub_ind_padded[i, src_ind_sub],
                tgt_sub_ind_padded[i, tgt_ind_sub]
            ]).transpose(0,1)
        )
    return torch.cat(matches)
        