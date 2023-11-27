from models.blocks import *
from models.backbone import KPFCN
# from models.transformer import RepositioningTransformer
from models.transformer_sem_subspace import SemanticTransformer
from models.matching import Matching
from models.subspace_matching import SubspaceMatchingV2
from models.procrustes import SoftProcrustesLayer
from pointscope import PointScopeClient as PSC
from lib.tictok import Timers
from lib.utils import pad_sequence, unpad_sequences
from einops import rearrange


class Pipeline(nn.Module):

    def __init__(self, config):
        super(Pipeline, self).__init__()
        self.config = config
        self.backbone = KPFCN(config['kpfcn_config'])
        self.pe_type = config['coarse_transformer']['pe_type']
        self.positioning_type = config['coarse_transformer']['positioning_type']
        self.coarse_transformer = SemanticTransformer(config['coarse_transformer'])
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])

        if self.config.use_lepard_pretrained:
            lepard_weights = torch.load(self.config.lepard_pretrained_path)["state_dict"]
            self.load_state_dict(lepard_weights, strict=False)

        self.coarse_subspace_matching = SubspaceMatchingV2(config['coarse_matching'])
        self.timers = Timers()


    def forward(self, data,  timers=None):


        if self.timers: self.timers.tic('kpfcn backbone encode')
        coarse_feats = self.backbone(data, phase="coarse")
        if self.timers: self.timers.toc('kpfcn backbone encode')

        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, \
            src_semantic_masks, tgt_semantic_masks, \
            src_sub_ind_padded, tgt_sub_ind_padded, \
            src_sub_ind_padded_mask, tgt_sub_ind_padded_mask, \
            semantic_masks_valid, src_2dfeat, tgt_2dfeat = self.split_feats(coarse_feats, data)
        data.update({'s_pcd': s_pcd, 't_pcd': t_pcd})
        if self.timers: self.timers.toc('coarse_preprocess')

        if self.timers: self.timers.tic('coarse feature transformer')
        src_feats, tgt_feats, src_semantic_feat, tgt_semantic_feat = self.coarse_transformer(
            src_feats, tgt_feats, 
            s_pcd, t_pcd, 
            src_mask, tgt_mask, 
            src_sub_ind_padded, tgt_sub_ind_padded,
            src_sub_ind_padded_mask, tgt_sub_ind_padded_mask, semantic_masks_valid,
            src_2dfeat, tgt_2dfeat,
            data, timers=timers)
            
            
        if self.timers: self.timers.toc('coarse feature transformer')

        if self.timers: self.timers.tic('match feature coarse')
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(
            src_feats, tgt_feats, src_mask, tgt_mask, data, pe_type = self.pe_type)
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')

        if self.timers: self.timers.tic('procrustes_layer')
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')
        
        
        if self.timers: self.timers.tic('semantic match feature coarse')
        if semantic_masks_valid.any():
            subspace_conf_matrix_pred, subspace_coarse_match_pred = self.coarse_subspace_matching(
                    src_semantic_feat, tgt_semantic_feat, 
                    ~src_sub_ind_padded_mask, ~tgt_sub_ind_padded_mask, 
                    semantic_masks_valid
            )
            data.update({
                'subspace_conf_matrix_pred': subspace_conf_matrix_pred, 
                'subspace_coarse_match_pred': subspace_coarse_match_pred,
            })
        if self.timers: self.timers.toc('semantic match feature coarse')

        return data

    def split_feats(self, geo_feats, data):

        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        src_ind_coarse_split = data[ 'src_ind_coarse_split']
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']
        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]

        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

        src_semantic_masks = data['src_semantic_masks']
        tgt_semantic_masks = data['tgt_semantic_masks']
        
        src_semantic_masks = rearrange(src_semantic_masks, 'b s m -> (b s) m')
        tgt_semantic_masks = rearrange(tgt_semantic_masks, 'b s n -> (b s) n')
        
        src_semantic_masks_hard = src_semantic_masks > self.config.semantic_mask_thr
        tgt_semantic_masks_hard = tgt_semantic_masks > self.config.semantic_mask_thr

        semantic_masks_valid = ((src_semantic_masks_hard.sum(dim=-1) > 0) & (tgt_semantic_masks_hard.sum(dim=-1) > 0))
        
        if semantic_masks_valid.any():
            src_semantic_masks = src_semantic_masks
            tgt_semantic_masks = tgt_semantic_masks
            
            src_range = torch.arange(src_semantic_masks_hard.shape[-1])
            src_sub_ind = [src_range[b] for b in src_semantic_masks_hard]
            
            tgt_range = torch.arange(tgt_semantic_masks_hard.shape[-1])
            tgt_sub_ind = [tgt_range[b] for b in tgt_semantic_masks_hard]

            src_sub_ind_padded, src_sub_ind_padded_mask, _ = \
                pad_sequence(src_sub_ind, batch_first=True, 
                             require_padding_mask=True, padding_value=-1)
            tgt_sub_ind_padded, tgt_sub_ind_padded_mask, _ = \
                pad_sequence(tgt_sub_ind, batch_first=True, 
                             require_padding_mask=True, padding_value=-1)

            src_sub_ind_padded = src_sub_ind_padded.to(src_semantic_masks.device)
            tgt_sub_ind_padded = tgt_sub_ind_padded.to(tgt_semantic_masks.device)
            
            src_sub_ind_padded_mask = src_sub_ind_padded_mask.to(src_semantic_masks.device)
            tgt_sub_ind_padded_mask = tgt_sub_ind_padded_mask.to(tgt_semantic_masks.device)
        
        else:
            src_sub_ind_padded, tgt_sub_ind_padded = None, None
            src_sub_ind_padded_mask, tgt_sub_ind_padded_mask = None, None

        data['src_semantic_masks'] = src_semantic_masks
        data['tgt_semantic_masks'] = tgt_semantic_masks

        data['src_sub_ind_padded'] = src_sub_ind_padded
        data['tgt_sub_ind_padded'] = tgt_sub_ind_padded
        
        data['src_sub_ind_padded_mask'] = src_sub_ind_padded_mask
        data['tgt_sub_ind_padded_mask'] = tgt_sub_ind_padded_mask

        data['semantic_masks_valid'] = semantic_masks_valid

        src_2dfeat = data["src_2dfeat"]
        tgt_2dfeat = data["tgt_2dfeat"]

        return src_feats.view( b_size , src_pts_max , -1), \
                tgt_feats.view( b_size , tgt_pts_max , -1), \
                src_pcd.view( b_size , src_pts_max , -1), \
                tgt_pcd.view( b_size , tgt_pts_max , -1), \
                src_mask, \
                tgt_mask, \
                src_semantic_masks, \
                tgt_semantic_masks, \
                src_sub_ind_padded, \
                tgt_sub_ind_padded, \
                src_sub_ind_padded_mask, \
                tgt_sub_ind_padded_mask, \
                semantic_masks_valid, \
                src_2dfeat, \
                tgt_2dfeat