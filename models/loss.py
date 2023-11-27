import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd
import nibabel.quaternions as nq
from sklearn.metrics import precision_recall_fscore_support
from datasets.utils import blend_scene_flow, multual_nn_correspondence, knn_point_np
from models.matching import Matching as CM
from einops import rearrange
from lib.utils import pad_sequence, unpad_sequences
from pointscope import PointScopeClient as PSC
from lib.utils import apply_transform
from models.circle_loss import weighted_circle_loss

def ransac_pose_estimation(src_pcd, tgt_pcd, corrs, distance_threshold=0.05, ransac_n=3):
    src_pcd = to_o3d_pcd(src_pcd)
    tgt_pcd = to_o3d_pcd(tgt_pcd)
    corrs = o3d.utility.Vector2iVector(np.array(corrs).T)
    
    result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
        source=src_pcd, target=tgt_pcd, corres=corrs,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.registration.RANSACConvergenceCriteria(50000, 5000))
    return result_ransac.transformation


def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html
    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,4,4]
    Returns:
    p (float): transformation error
    """

    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]

    return p.item()


class MatchMotionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.focal_alpha = config['focal_alpha']
        self.focal_gamma = config['focal_gamma']
        self.pos_w = config['pos_weight']
        self.neg_w = config['neg_weight']
        self.mot_w = config['motion_weight']
        self.mat_w = config['match_weight']
        self.semantic_w = config['semantic_loss_w']
        self.motion_loss_type = config['motion_loss_type']

        self.match_type = config['match_type']
        self.positioning_type = config['positioning_type']


        self.registration_threshold = config['registration_threshold']

        self.confidence_threshold_metric = config['confidence_threshold_metric']
        self.inlier_thr = config['inlier_thr']
        self.fmr_thr = config['fmr_thr']
        self.mutual_nearest = config['mutual_nearest']
        self.dataset = config['dataset']


    def forward(self, data):
        loss_info = {}
        loss = self.ge_coarse_loss_sim(data, loss_info)
        loss_info.update({ 'loss': loss })
        return loss_info

    def ge_coarse_loss_circle(self, data, loss_info, eval_metric=False):


        if self.dataset == "4dmatch":
            s2t_flow = torch.zeros_like(data['s_pcd'])
            for i, cflow in enumerate(data['coarse_flow']):
                s2t_flow[i][: len(cflow)] = cflow

        loss = 0.

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        conf_matrix_pred = data['conf_matrix_pred']
        match_gt = data['coarse_matches']
        R_s2t_gt = data['batched_rot']
        t_s2t_gt = data['batched_trn']
        
        feat_2d_sim = data["feat_2d_sim"]
        s2t_subspace = data["s2t_subspace"]
        t2s_subspace = data["t2s_subspace"]

        #get the overlap mask, for dense motion loss
        s_overlap_mask = torch.zeros_like(src_mask).bool()
        for bi, corr in enumerate (match_gt):
            s_overlap_mask[bi][ corr[0] ] = True
        # compute focal loss
        c_weight = (src_mask[:, :, None] * tgt_mask[:, None, :]).float()
        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        
        
        # s_pcd = data["s_pcd"]
        # t_pcd = data["t_pcd"]
            
        # tsfms = data["tsfms"][0]
        # s_pcd_gt = apply_transform(s_pcd, tsfms)
        # point_wise_dist = torch.cdist(s_pcd_gt, t_pcd)
        # pos_mask = point_wise_dist < 0.1
        # s2t_neg_mask = ((point_wise_dist > 0.2).long() * (s2t_subspace > 0.1).long()).bool()
        # t2s_neg_mask = ((point_wise_dist > 0.2).long() * (t2s_subspace.transpose(1, 2) > 0.1).long()).bool()
        
        # c_loss = weighted_circle_loss(
        #     pos_masks=pos_mask,
        #     neg_masks=neg_mask,
        #     feat_dists=conf_matrix_pred,
        #     pos_margin=0.1,
        #     neg_margin=1.4,
        #     pos_optimal=0.1,
        #     neg_optimal=1.4,
        #     log_scale=16,
        #     pos_scales=None,
        #     neg_scales=None,
        # )
        
        # focal_coarse_s2t = self.compute_correspondence_loss_sim(conf_matrix_pred, conf_matrix_gt, weight=s2t_subspace > 0.1)
        # focal_coarse_t2s = self.compute_correspondence_loss_sim(conf_matrix_pred, conf_matrix_gt, weight=t2s_subspace.transpose(1, 2) > 0.1)
        # focal_coarse = (focal_coarse_s2t + focal_coarse_t2s) / 2
        c_weight = ((s2t_subspace > 0.2) + (t2s_subspace.transpose(1, 2) > 0.2)) > 0
        focal_coarse = self.compute_correspondence_loss_sim(conf_matrix_pred, conf_matrix_gt, weight=c_weight)
        
        recall, precision = self.compute_match_recall( conf_matrix_gt, data['coarse_match_pred'])
        loss_info.update( {"focal_coarse": focal_coarse, "recall_coarse": recall, "precision_coarse": precision } )
        # loss = loss + self.mat_w * focal_coarse
        loss = loss + focal_coarse
        
        
        
        VIS_GAUSSIAN_MIXTURE_SS = False
        if VIS_GAUSSIAN_MIXTURE_SS:
            idx = 1
            s_pcd = data["s_pcd"][0].cpu()
            t_pcd = data["t_pcd"][0].cpu()+torch.Tensor([3,0,0])
            tsfm = data["tsfms"][0].cpu()
            s_pcd = apply_transform(s_pcd, tsfm)
            psc = PSC().vedo().add_pcd(torch.cat([s_pcd, t_pcd]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            src_color, tgt_color = torch.zeros_like(s_pcd), torch.zeros_like(t_pcd)
            src_color[idx] = torch.tensor([1,0,0]).float()
            
            ss = s2t_subspace[0][idx].cpu()
            tgt_color = torch.tensor([1,0,0]).float() * ss[..., None]
            
            tgt_color[ss<0.2] = torch.tensor([0, 1, 0]).float()
            
            psc.add_color(torch.cat([src_color, tgt_color]))
            psc.show()
            
        VIS_CIRCLE_POSITIVE = False
        if VIS_CIRCLE_POSITIVE:
            idx = 100
            s_pcd = data["s_pcd"]
            t_pcd = data["t_pcd"]
                
            tsfms = data["tsfms"][0]
            s_pcd_gt = apply_transform(s_pcd, tsfms)
            
            s_ = s_pcd_gt[0].cpu()
            t_ = t_pcd[0].cpu()
            psc = PSC().vedo().add_pcd(torch.cat([s_, t_+torch.Tensor([3,0,0])]))
            src_color, tgt_color = torch.zeros_like(s_), torch.zeros_like(t_)
            src_color[idx] = torch.tensor([1,0,0]).float()
            
            p_ = pos_mask[0][idx].cpu()
            neg_ = s2t_neg_mask[0][idx].cpu()
            tgt_color[p_] = torch.tensor([0, 1, 0]).float()
            tgt_color[neg_] = torch.tensor([0, 0, 1]).float()
            psc.add_color(torch.cat([src_color, tgt_color]))
            psc.show()
            
        return loss


    def ge_coarse_loss_sim(self, data, loss_info, eval_metric=False):


        if self.dataset == "4dmatch":
            s2t_flow = torch.zeros_like(data['s_pcd'])
            for i, cflow in enumerate(data['coarse_flow']):
                s2t_flow[i][: len(cflow)] = cflow

        loss = 0.

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        conf_matrix_pred = data['conf_matrix_pred']
        match_gt = data['coarse_matches']
        R_s2t_gt = data['batched_rot']
        t_s2t_gt = data['batched_trn']
        
        feat_2d_sim = data["feat_2d_sim"]
        s2t_subspace = data["s2t_subspace"]
        t2s_subspace = data["t2s_subspace"]

        #get the overlap mask, for dense motion loss
        s_overlap_mask = torch.zeros_like(src_mask).bool()
        for bi, corr in enumerate (match_gt):
            s_overlap_mask[bi][ corr[0] ] = True
        # compute focal loss
        c_weight = (src_mask[:, :, None] * tgt_mask[:, None, :]).float()
        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        
        
        c_weight = ((feat_2d_sim*c_weight) > 0.4)
        focal_coarse = self.compute_correspondence_loss_sim(conf_matrix_pred, conf_matrix_gt, weight=c_weight)
        
        recall, precision = self.compute_match_recall( conf_matrix_gt, data['coarse_match_pred'])
        loss_info.update({"focal_coarse": focal_coarse, "recall_coarse": recall, "precision_coarse": precision})
        # loss = loss + self.mat_w * focal_coarse
        loss = loss + focal_coarse

        if False:
            idx = 1
            s_pcd = data["s_pcd"][0].cpu()
            t_pcd = data["t_pcd"][0].cpu()+torch.Tensor([3,0,0])
            tsfm = data["tsfms"][0].cpu()
            s_pcd = apply_transform(s_pcd, tsfm)
            psc = PSC().vedo().add_pcd(torch.cat([s_pcd, t_pcd]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            src_color, tgt_color = torch.zeros_like(s_pcd), torch.zeros_like(t_pcd)
            src_color[idx] = torch.tensor([1,0,0]).float()
            tgt_color = torch.tensor([1,0,0]).float() * s2t_subspace[0][idx, None].T.cpu()
            psc.add_color(torch.cat([src_color, tgt_color]))
            psc.show()

        return loss

    def compute_correspondence_loss_sim(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = (conf_gt == 1) & weight
        neg_mask = (conf_gt == 0) & weight

        

        pos_w, neg_w = self.pos_w, self.neg_w

        #corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.
            
        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        # no supervision on dustbin row & column.
        loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
        loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
        # if weight is not None:
        #     loss_pos = loss_pos * weight[pos_mask]
        #     loss_neg = loss_neg * weight[neg_mask]
            
        loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
        return loss

    def ge_coarse_loss(self, data, loss_info, eval_metric=False):


        if self.dataset == "4dmatch":
            s2t_flow = torch.zeros_like(data['s_pcd'])
            for i, cflow in enumerate(data['coarse_flow']):
                s2t_flow[i][: len(cflow)] = cflow

        loss = 0.

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        conf_matrix_pred = data['conf_matrix_pred']
        match_gt = data['coarse_matches']
        R_s2t_gt = data['batched_rot']
        t_s2t_gt = data['batched_trn']

        #get the overlap mask, for dense motion loss
        s_overlap_mask = torch.zeros_like(src_mask).bool()
        for bi, corr in enumerate (match_gt):
            s_overlap_mask[bi][ corr[0] ] = True
        # compute focal loss
        c_weight = (src_mask[:, :, None] * tgt_mask[:, None, :]).float()
        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        focal_coarse = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=c_weight)
        recall, precision = self.compute_match_recall( conf_matrix_gt, data['coarse_match_pred'])
        loss_info.update( { "focal_coarse": focal_coarse, "recall_coarse": recall, "precision_coarse": precision } )
        # loss = loss + self.mat_w * focal_coarse
        loss = loss + 0 * focal_coarse


        # Semantic Subspace matching 
        semantic_masks_valid = data['semantic_masks_valid']
        if semantic_masks_valid.any():
            subspace_conf_matrix_pred = data['subspace_conf_matrix_pred']
            
            src_sub_ind_padded = data["src_sub_ind_padded"][semantic_masks_valid]
            tgt_sub_ind_padded = data["tgt_sub_ind_padded"][semantic_masks_valid]

            src_sub_ind_padded_mask = data['src_sub_ind_padded_mask'][semantic_masks_valid]
            tgt_sub_ind_padded_mask = data['tgt_sub_ind_padded_mask'][semantic_masks_valid]
            
            subspace_num = int(semantic_masks_valid.shape[0] / conf_matrix_gt.shape[0])
            
            conf_matrix_sub_gt = conf_matrix_gt.unsqueeze(dim=1).repeat(1, subspace_num, 1, 1)
            conf_matrix_sub_gt = rearrange(conf_matrix_sub_gt, 'b s m n -> (b s) m n')[semantic_masks_valid]
            
            s_conf_matrix_gt = torch.stack([
                conf_matrix_sub_gt[i][src_sub_ind_padded[i]][:, tgt_sub_ind_padded[i]] for i in range(conf_matrix_sub_gt.shape[0])
            ])
            
            
            # src_matches_gt = conf_matrix_gt.sum(dim=2)
            # tgt_matches_gt = conf_matrix_gt.sum(dim=1)
            
            
            # src_matches_gt_expand = src_matches_gt.unsqueeze(dim=1).repeat(1, semantic_masks_valid.shape[0], 1)
            # tgt_matches_gt_expand = tgt_matches_gt.unsqueeze(dim=1).repeat(1, semantic_masks_valid.shape[0], 1)
            
            # src_matches_gt_expand = rearrange(src_matches_gt_expand, 'b s m -> (b s) m')[semantic_masks_valid]
            # tgt_matches_gt_expand = rearrange(tgt_matches_gt_expand, 'b s n -> (b s) n')[semantic_masks_valid]

            # src_matches_gt_expand = torch.cat([src_matches_gt_expand, torch.zeros_like(src_matches_gt_expand[:, :1])], dim=1)
            # tgt_matches_gt_expand = torch.cat([tgt_matches_gt_expand, torch.zeros_like(tgt_matches_gt_expand[:, :1])], dim=1)
                    
            # src_matches_gt_subspace = torch.stack([f[i] for f, i in zip(src_matches_gt_expand, src_sub_ind_padded)])
            # tgt_matches_gt_subspace = torch.stack([f[i] for f, i in zip(tgt_matches_gt_expand, tgt_sub_ind_padded)])

            s_c_weight = torch.einsum("bn,bm->bnm", (~src_sub_ind_padded_mask).long(), (~tgt_sub_ind_padded_mask).long())
            # s_conf_matrix_gt = torch.einsum("bn,bm->bnm", src_matches_gt_subspace, tgt_matches_gt_subspace)

            s_focal_coarse = self.compute_subspace_correspondence_loss(
                subspace_conf_matrix_pred, s_conf_matrix_gt, 
                weight=s_c_weight)
            
            loss_info.update({"s_focal_coarse": s_focal_coarse})
            loss = loss + s_focal_coarse * self.semantic_w

        else:
            print("fuck")

            # VIS_SUBSPACE_CORR = False
            # if VIS_SUBSPACE_CORR:
            #     offset = np.array([0, 0, 0])
            #     s_pcd = data['s_pcd'][0].clone().cpu()
            #     t_pcd = data['t_pcd'][0].clone().cpu()+offset
            #     tsfm = data["tsfms"][0].clone().cpu()
            #     s_pcd_gt = apply_transform(s_pcd, tsfm)
                
            #     psc = PSC().add_pcd(s_pcd_gt).add_pcd(t_pcd)
                
            #     src_sub_ind_padded = src_sub_ind_padded.clone().cpu()
            #     tgt_sub_ind_padded = tgt_sub_ind_padded.clone().cpu()
                
            #     c = s_conf_matrix_gt.long().bool()
                
            #     for ind in range(c.shape[0]):
            #         corr = c[ind].nonzero()
            #         if corr.shape[0] > 0:
            #             psc.add_lines(s_pcd_gt[src_sub_ind_padded[ind][corr[:, 0]]], 
            #                           t_pcd[tgt_sub_ind_padded[ind][corr[:, 1]]], 
            #                           color=[1,1,1])
                
            #     psc.show()


        return loss

        # if recall > 0.01 and self.mot_w > 0:
        #     R_s2t_pred = data["R_s2t_pred"]
        #     t_s2t_pred = data["t_s2t_pred"]

        #     #compute predicted flow. Note, if 4dmatch, the R_pred,t_pred try to find the best rigid fit of deformation
        #     src_pcd_wrapped_pred = (torch.matmul(R_s2t_pred, data['s_pcd'].transpose(1, 2)) + t_s2t_pred).transpose(1, 2)
        #     sflow_pred = src_pcd_wrapped_pred - data['s_pcd']


        #     if self.dataset == '4dmatch':
        #         spcd_deformed = data['s_pcd'] + s2t_flow
        #         src_pcd_wrapped_gt = (torch.matmul(R_s2t_gt, spcd_deformed.transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
        #     else : # 3dmatch
        #         src_pcd_wrapped_gt = (torch.matmul(R_s2t_gt, data['s_pcd'].transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
        #     sflow_gt = src_pcd_wrapped_gt - data['s_pcd']

        #     e1 = torch.sum(torch.abs(sflow_pred - sflow_gt), 2)
        #     e1 = e1[s_overlap_mask] # [data['src_mask']]
        #     l1_loss = torch.mean(e1)
        #     loss = loss + self.mot_w * l1_loss


        #
        # if eval_metric :
        #
        #     match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=self.confidence_threshold_metric, mutual=self.mutual_nearest)
        #
        #     '''Inlier Ratio (IR)'''
        #     ir = self.compute_inlier_ratio(match_pred, data, self.inlier_thr,
        #                                    s2t_flow=s2t_flow if self.dataset == "4dmatch" else None)
        #     loss_info.update({"Inlier Ratio": ir.mean()})
        #
        #     if self.dataset == '3dmatch':
        #
        #         '''Feature Matching Recall (FMR)'''
        #         fmr = (ir > self.fmr_thr).float().sum() / len(ir)
        #         loss_info.update({"Feature Matching Recall": fmr})
        #
        #         '''Registration Recall (RR)'''
        #         rot_, trn_ = self.ransac_regist_coarse(data['s_pcd'], data['t_pcd'], src_mask, tgt_mask , match_pred)
        #         rot, trn = rot_.to(data['s_pcd']) , trn_.to(data['s_pcd'])
        #         rr = self.compute_registration_recall(rot, trn, data, self.registration_threshold)
        #         loss_info.update({'Registration_Recall': rr})



        # if self.positioning_type == "procrustes":

        #     for layer_ind in data["position_layers"]:
        #         # compute focal loss
        #         rpe_conf_matrix = data["position_layers"][layer_ind]["conf_matrix"]
        #         focal_rpe = self.compute_correspondence_loss(rpe_conf_matrix, conf_matrix_gt, weight=c_weight)
        #         recall, precision = self.compute_match_recall(conf_matrix_gt,
        #                                                       data["position_layers"][layer_ind]['match_pred'])
        #         # loss_info.update({'focal_layer_%d' % layer_ind: focal_rpe, 'recall_layer_%d' % layer_ind: recall,
        #         #                   'precision_layer_%d' % layer_ind: precision})
        #         loss = loss + self.mat_w * focal_rpe

        #         if recall >0.01 and self.mot_w > 0:
        #             R_s2t_pred = data["position_layers"][layer_ind]["R_s2t_pred"]
        #             t_s2t_pred = data["position_layers"][layer_ind]["t_s2t_pred"]

        #             src_pcd_wrapped_pred = (torch.matmul(R_s2t_pred, data['s_pcd'].transpose(1, 2)) + t_s2t_pred).transpose(1, 2)
        #             sflow_pred = src_pcd_wrapped_pred - data['s_pcd']


        #             if self.dataset == '4dmatch':
        #                 spcd_deformed = data['s_pcd'] + s2t_flow
        #                 src_pcd_wrapped_gt = ( torch.matmul(R_s2t_gt, spcd_deformed.transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
        #             else:  # 3dmatch
        #                 src_pcd_wrapped_gt = ( torch.matmul(R_s2t_gt, data['s_pcd'].transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
        #             sflow_gt = src_pcd_wrapped_gt - data['s_pcd']

        #             e1 = torch.sum(torch.abs(sflow_pred - sflow_gt), 2) #[data['src_mask']]
        #             e1 = e1[s_overlap_mask]  # [data['src_mask']]
        #             l1_loss = torch.mean(e1)
        #             loss = loss + self.mot_w * l1_loss

        return loss


    @staticmethod
    def compute_nrfmr(match_pred, data, recall_thr=0.04):

        s_pcd, t_pcd = data['s_pcd'], data['t_pcd']

        s_pcd_raw = data['src_pcd_list']
        sflow_list = data['sflow_list']
        metric_index_list = data['metric_index_list']

        batched_rot = data['batched_rot']  # B,3,3
        batched_trn = data['batched_trn']

        nrfmr = 0.

        for i in range(len(s_pcd_raw)):

            # use the match prediction as the motion anchor
            match_pred_i = match_pred[match_pred[:, 0] == i]
            s_id, t_id = match_pred_i[:, 1], match_pred_i[:, 2]
            s_pcd_matched = s_pcd[i][s_id]
            t_pcd_matched = t_pcd[i][t_id]
            motion_pred = t_pcd_matched - s_pcd_matched

            if len(s_pcd_matched) >= 3 :

                # get the wrapped metric points
                metric_index = metric_index_list[i]
                sflow = sflow_list[i]
                s_pcd_raw_i = s_pcd_raw[i]
                metric_pcd = s_pcd_raw_i[metric_index]
                metric_sflow = sflow[metric_index]
                metric_pcd_deformed = metric_pcd + metric_sflow
                metric_pcd_wrapped_gt = (torch.matmul(batched_rot[i], metric_pcd_deformed.T) + batched_trn[i]).T

                # blend the motion for metric points
                try:
                    metric_motion_pred, valid_mask = MatchMotionLoss.blend_anchor_motion(
                        metric_pcd.cpu().numpy(), s_pcd_matched.cpu().numpy(), motion_pred.cpu().numpy(), knn=3,
                        search_radius=0.1)
                    metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)
                    dist = torch.sqrt(torch.sum((metric_pcd_wrapped_pred - metric_pcd_wrapped_gt) ** 2, dim=1))
                    r = (dist < recall_thr).float().sum() / len(dist)
                except :
                    r = 0

                nrfmr = nrfmr + r


                debug = False
                if debug:
                    import mayavi.mlab as mlab
                    c_red = (224. / 255., 0 / 255., 125 / 255.)
                    c_pink = (224. / 255., 75. / 255., 232. / 255.)
                    c_blue = (0. / 255., 0. / 255., 255. / 255.)
                    scale_factor = 0.013
                    metric_pcd_wrapped_gt = metric_pcd_wrapped_gt.cpu()
                    metric_pcd_wrapped_pred = metric_pcd_wrapped_pred.cpu()
                    err = metric_pcd_wrapped_pred - metric_pcd_wrapped_gt
                    mlab.points3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2],
                                  scale_factor=scale_factor, color=c_pink)
                    mlab.points3d(metric_pcd_wrapped_pred[:, 0], metric_pcd_wrapped_pred[:, 1],
                                  metric_pcd_wrapped_pred[:, 2], scale_factor=scale_factor, color=c_blue)
                    mlab.quiver3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2],
                                  err[:, 0], err[:, 1], err[:, 2],
                                  scale_factor=1, mode='2ddash', line_width=1.)
                    mlab.show()


        nrfmr = nrfmr / len(s_pcd_raw)

        return nrfmr

    @staticmethod
    def blend_anchor_motion(query_loc, reference_loc, reference_flow, knn=3, search_radius=0.1):
        '''approximate flow on query points
        this function assume query points are sub- or un-sampled from reference locations
        @param query_loc:[m,3]
        @param reference_loc:[n,3]
        @param reference_flow:[n,3]
        @param knn:
        @return:
            blended_flow:[m,3]
        '''
        dists, idx = knn_point_np(knn, reference_loc, query_loc)
        dists[dists < 1e-10] = 1e-10
        mask = dists > search_radius
        dists[mask] = 1e+10
        weight = 1.0 / dists
        weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
        blended_flow = np.sum(reference_flow[idx] * weight.reshape([-1, knn, 1]), axis=1, keepdims=False)

        mask = mask.sum(axis=1) < 3

        return blended_flow, mask

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        #corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.
            
        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        if self.match_type == "dual_softmax":
            pos_conf = conf[pos_mask]
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
            loss =  pos_w * loss_pos.mean()
            return loss

        elif self.match_type == "sinkhorn":
            # no supervision on dustbin row & column.
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
                
            loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
            return loss
        
    def compute_subspace_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        # Select subspace with both possitive and negative points
        # valid_mask = weight.flatten(start_dim=1).sum(dim=1) > 0
        # conf = conf[valid_mask]
        # conf_gt = conf_gt[valid_mask]
        # weight = weight[valid_mask]
        
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        #corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.
            
        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        if self.match_type == "dual_softmax":
            pos_conf = conf[pos_mask]
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
            loss =  pos_w * loss_pos.mean()
            return loss

        elif self.match_type == "sinkhorn":
            # no supervision on dustbin row & column.
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            # if weight is not None:
            #     loss_pos = loss_pos[weight[pos_mask]]
            #     loss_neg = loss_neg[weight[neg_mask]]
                
            loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
            return loss

    def match_2_conf_matrix(self, matches_gt, matrix_pred):
        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate (matches_gt) :
            matrix_gt [ b][ match[0],  match[1] ] = 1
        return matrix_gt


    @staticmethod
    def compute_match_recall(conf_matrix_gt, match_pred) : #, s_pcd, t_pcd, search_radius=0.3):
        '''
        @param conf_matrix_gt:
        @param match_pred:
        @return:
        '''

        pred_matrix = torch.zeros_like(conf_matrix_gt)

        b_ind, src_ind, tgt_ind = match_pred[:, 0], match_pred[:, 1], match_pred[:, 2]
        pred_matrix[b_ind, src_ind, tgt_ind] = 1.

        true_positive = (pred_matrix == conf_matrix_gt) * conf_matrix_gt

        recall = true_positive.sum() / conf_matrix_gt.sum()

        precision = true_positive.sum() / max(len(match_pred), 1)

        return recall, precision


    @staticmethod
    def ransac_regist_coarse(batched_src_pcd, batched_tgt_pcd, src_mask, tgt_mask, match_pred ):
        # s_len = src_mask.sum(dim=1).int()
        # t_len = tgt_mask.sum(dim=1).int()
        bsize = len(batched_src_pcd)

        batched_src_pcd = MatchMotionLoss.tensor2numpy( batched_src_pcd)
        batched_tgt_pcd = MatchMotionLoss.tensor2numpy( batched_tgt_pcd)
        match_pred = MatchMotionLoss.tensor2numpy(match_pred)

        rot = []
        trn = []

        for i in range(bsize):
            # s_pcd = batched_src_pcd[i][:s_len[i]]
            # t_pcd = batched_tgt_pcd[i][:t_len[i]]

            s_pcd = batched_src_pcd[i]
            t_pcd = batched_tgt_pcd[i]

            pair_i = match_pred[:, 0] == i
            n_pts = pair_i.sum()

            if n_pts < 3 :
                rot.append(torch.eye(3))
                trn.append(torch.zeros((3,1)))
                continue

            ind = match_pred[pair_i]
            s_ind, t_ind = ind[:, 1], ind[:, 2]

            pose = ransac_pose_estimation(s_pcd, t_pcd, [s_ind, t_ind], distance_threshold=0.05)
            pose = pose.copy()
            rot.append(torch.from_numpy(pose[:3,:3]))
            trn.append(torch.from_numpy(pose[:3,3:]))

        return  torch.stack(rot, dim=0 ), torch.stack(trn , dim=0)#ndarray


    @staticmethod
    def compute_inlier_ratio(match_pred, data, inlier_thr, s2t_flow=None):
        s_pcd, t_pcd = data['s_pcd'], data['t_pcd'] #B,N,3
        batched_rot = data['batched_rot'] #B,3,3
        batched_trn = data['batched_trn']

        if s2t_flow is not None: # 4dmatch
            s_pcd_deformed = s_pcd + s2t_flow
            s_pcd_wrapped = (torch.matmul(batched_rot, s_pcd_deformed.transpose(1, 2)) + batched_trn).transpose(1,2)
        else:  # 3dmatch
            s_pcd_wrapped = (torch.matmul(batched_rot, s_pcd.transpose(1, 2)) + batched_trn).transpose(1,2)

        s_pcd_matched = s_pcd_wrapped [match_pred[:,0], match_pred[:,1]]
        t_pcd_matched = t_pcd [match_pred[:,0], match_pred[:,2]]
        inlier = torch.sum( (s_pcd_matched - t_pcd_matched)**2 , dim= 1) <  inlier_thr**2

        bsize = len(s_pcd)
        IR=[]
        for i in range(bsize):
            pair_i = match_pred[:, 0] == i
            n_match = pair_i.sum()
            inlier_i = inlier[pair_i]
            n_inlier = inlier_i.sum().float()
            if n_match <3:
                IR.append( n_match.float()*0)
            else :
                IR.append(n_inlier/n_match)

        return torch.stack(IR, dim=0)



    @staticmethod
    def compute_registration_recall(R_est, t_est, data, thr=0.2):

        bs = len(R_est)
        success = 0.


        if data['gt_cov'] is not None:
            
            if isinstance(data['gt_cov'], torch.Tensor):
                data['gt_cov'] = data['gt_cov'].numpy()

            err2 = thr ** 2

            gt = np.zeros( (bs, 4, 4))
            gt[:, -1,-1] = 1
            gt[:, :3, :3] = data['batched_rot'].cpu().numpy()
            gt[:, :3, 3:] = data['batched_trn'].cpu().numpy()

            pred = np.zeros((bs, 4, 4))
            pred[:, -1, -1] = 1
            pred[:, :3, :3] = R_est.detach().cpu().numpy()
            pred[:, :3, 3:] = t_est.detach().cpu().numpy()

            for i in range(bs):

                p = computeTransformationErr( np.linalg.inv(gt[i]) @ pred[i], data['gt_cov'][i])

                if p <= err2:
                    success += 1

            rr = success / bs
            return rr


        else :

            return 0.


    @staticmethod
    def tensor2numpy(tensor):
        if tensor.requires_grad:
            tensor=tensor.detach()
        return tensor.cpu().numpy()