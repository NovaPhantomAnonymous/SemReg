from models.blocks import *
from models.backbone import KPFCN
# from models.transformer import RepositioningTransformer
from models.transformer_sem_subspace_sim import SemanticTransformer
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

        # self.coarse_subspace_matching = SubspaceMatchingV2(config['coarse_matching'])
        self.timers = Timers()


    def forward(self, data,  timers=None):


        if self.timers: self.timers.tic('kpfcn backbone encode')
        coarse_feats = self.backbone(data, phase="coarse")
        if self.timers: self.timers.toc('kpfcn backbone encode')

        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, \
            src_2dfeat, tgt_2dfeat = self.split_feats(coarse_feats, data)
        data.update({'s_pcd': s_pcd, 't_pcd': t_pcd})
        if self.timers: self.timers.toc('coarse_preprocess')

        if self.timers: self.timers.tic('coarse feature transformer')
        src_feats, tgt_feats, feat_2d_sim = self.coarse_transformer(
            src_feats, tgt_feats, 
            s_pcd, t_pcd, 
            src_mask, tgt_mask,
            src_2dfeat, tgt_2dfeat,
            data, timers=timers)
        data.update({'feat_2d_sim': feat_2d_sim, 
                     "src_feats": src_feats, 
                     "tgt_feats": tgt_feats})
        if self.timers: self.timers.toc('coarse feature transformer')

        if self.timers: self.timers.tic('match feature coarse')
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(
            src_feats, tgt_feats, src_mask, tgt_mask, data, pe_type = self.pe_type)
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')

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

        src_2dfeat = data["src_2dfeat"]
        tgt_2dfeat = data["tgt_2dfeat"]

        return src_feats.view( b_size , src_pts_max , -1), \
                tgt_feats.view( b_size , tgt_pts_max , -1), \
                src_pcd.view( b_size , src_pts_max , -1), \
                tgt_pcd.view( b_size , tgt_pts_max , -1), \
                src_mask, \
                tgt_mask, \
                src_2dfeat, \
                tgt_2dfeat