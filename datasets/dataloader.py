import numpy as np
from functools import partial
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from datasets._3dmatch import _3DMatch
from datasets.utils import blend_scene_flow, multual_nn_correspondence
from pointscope import PointScopeClient as PSC

from torch.utils.data import DataLoader
from datasets.projection import Projection, adjust_intrinsic, unproject
from lib.colormaps import ADE20K_COLORMAP

from PIL import Image
import time
from lib.utils import apply_transform
from sklearn.cluster import KMeans
from einops import rearrange
import random


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)

def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)



def collate_fn_3dmatch(list_data, config, neighborhood_limits ):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []

    correspondences_list = []
    src_pcd_list = []
    tgt_pcd_list = []

    batched_rot = []
    batched_trn = []

    gt_cov_list = []

    src_2d3d_list, tgt_2d3d_list, mutual_mask_labels_list, all_mask_labels_list, tsfms = [], [], [], [], []

    # for ind, ( src_pcd, tgt_pcd, src_feats, \
    #     tgt_feats, correspondences, rot, trn, \
    #         gt_cov, src_2d3d, tgt_2d3d, mutual_mask_labels, \
    #         all_mask_labels, tsfm) in enumerate(list_data):
    for ind, ( src_pcd, tgt_pcd, src_feats, \
        tgt_feats, correspondences, rot, trn, \
            gt_cov, src_2d3d, tgt_2d3d, tsfm) in enumerate(list_data):

        correspondences_list.append(correspondences )
        src_pcd_list.append(torch.from_numpy(src_pcd) )
        tgt_pcd_list.append(torch.from_numpy(tgt_pcd) )

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))

        if rot is not None and trn is not None:
            batched_rot.append( torch.from_numpy(rot).float())
            batched_trn.append( torch.from_numpy(trn).float())
            tsfms.append(torch.from_numpy(tsfm))

        gt_cov_list.append(gt_cov)
        
        src_2d3d_list.append(src_2d3d)
        tgt_2d3d_list.append(tgt_2d3d)
        # mutual_mask_labels_list.append(mutual_mask_labels)
        # all_mask_labels_list.append(all_mask_labels)

    gt_cov_list = None if gt_cov_list[0] is None \
        else np.stack(gt_cov_list, axis=0)

    # if timers: cnter['collate_load_batch'] = time.time() - st

    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    if len(batched_rot):
        batched_rot = torch.stack(batched_rot,dim=0)
        batched_trn = torch.stack(batched_trn,dim=0)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []


    # construt kpfcn inds
    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []





    # coarse infomation
    coarse_level = config.coarse_level
    pts_num_coarse = input_batches_len[coarse_level].view(-1, 2)
    b_size = pts_num_coarse.shape[0]
    src_pts_max, tgt_pts_max = pts_num_coarse.amax(dim=0)
    coarse_pcd = input_points[coarse_level] # .numpy()
    coarse_matches= []
    src_ind_coarse_split= [] # src_feats shape :[b_size * src_pts_max]
    src_ind_coarse = []
    tgt_ind_coarse_split= []
    tgt_ind_coarse = []
    accumu = 0
    src_mask = torch.zeros([b_size, src_pts_max], dtype=torch.bool)
    tgt_mask = torch.zeros([b_size, tgt_pts_max], dtype=torch.bool)

    #grid subsample fine level points for differentiable matching
    fine_pts, fine_length = batch_grid_subsampling_kpconv(input_points[0], input_batches_len[0], sampleDl=dl*0.5*0.85)
    fine_ind = batch_neighbors_kpconv(fine_pts, input_points[0], fine_length, input_batches_len[0], dl*0.5*0.85, 1).squeeze().long()

    topk = 4
    src_semantic_masks = torch.zeros([b_size, topk+1, src_pts_max])
    tgt_semantic_masks = torch.zeros([b_size, topk+1, tgt_pts_max])

    src_semantic_feat = torch.zeros([b_size, src_pts_max, src_2d3d_list[0][1].shape[-1]])
    tgt_semantic_feat = torch.zeros([b_size, tgt_pts_max, tgt_2d3d_list[0][1].shape[-1]])
    
    for entry_id, cnt in enumerate( pts_num_coarse ): #input_batches_len[-1].numpy().reshape(-1,2)) :

        n_s_pts, n_t_pts = cnt

        '''split mask for bottlenect feats'''
        src_mask[entry_id][:n_s_pts] = 1
        tgt_mask[entry_id][:n_t_pts] = 1


        '''split indices of bottleneck feats'''
        src_ind_coarse_split.append( torch.arange( n_s_pts ) + entry_id * src_pts_max )
        tgt_ind_coarse_split.append( torch.arange( n_t_pts ) + entry_id * tgt_pts_max )
        src_ind_coarse.append( torch.arange( n_s_pts ) + accumu )
        tgt_ind_coarse.append( torch.arange( n_t_pts ) + accumu + n_s_pts )


        '''get match at coarse level'''
        c_src_pcd = coarse_pcd[accumu : accumu + n_s_pts]
        c_tgt_pcd = coarse_pcd[accumu + n_s_pts: accumu + n_s_pts + n_t_pts]
        s_pc_wrapped = (torch.matmul( batched_rot[entry_id], c_src_pcd.T ) + batched_trn [entry_id]).T
        coarse_match_gt = torch.from_numpy( multual_nn_correspondence(s_pc_wrapped.numpy(), c_tgt_pcd.numpy(), search_radius=config['coarse_match_radius'])  )# 0.1m scaled
        coarse_matches.append(coarse_match_gt)

        accumu = accumu + n_s_pts + n_t_pts
        
        src_2d3d, tgt_2d3d = src_2d3d_list[entry_id], tgt_2d3d_list[entry_id]
        # src_mapping_pcd, src_pcd_2dfeat, src_pcd_mask = src_2d3d
        # tgt_mapping_pcd, tgt_pcd_2dfeat, tgt_pcd_mask = tgt_2d3d
        src_mapping_pcd, src_pcd_2dfeat = src_2d3d
        tgt_mapping_pcd, tgt_pcd_2dfeat = tgt_2d3d
        
        src_mapping_ind = torch.cdist(c_src_pcd, src_mapping_pcd).min(1).indices
        tgt_mapping_ind = torch.cdist(c_tgt_pcd, tgt_mapping_pcd).min(1).indices
        
        if False:
            PSC().add_pcd(torch.cat((src_2d3d[0], tgt_2d3d[0])))\
                .add_feat(torch.cat((src_2d3d[1], tgt_2d3d[1]))).show()
                
            #########################################################################################################
            patch_crop_thr = 0.4
            feat_sim_thr = 0.6
            src_ = apply_transform(c_src_pcd, tsfms[entry_id].clone())
            tgt_ = c_tgt_pcd+torch.Tensor([3,0,0])
            
            src_2d_feat = src_pcd_2dfeat[src_mapping_ind]
            tgt_2d_feat = tgt_pcd_2dfeat[tgt_mapping_ind]

            src_2d_feat_norm = torch.nn.functional.normalize(src_2d_feat)
            tgt_2d_feat_norm = torch.nn.functional.normalize(tgt_2d_feat)
            
            feat_sim = torch.einsum("nd,md->nm", src_2d_feat_norm, tgt_2d_feat_norm)

            s_points_neiboour = torch.cdist(c_src_pcd, c_src_pcd) < patch_crop_thr
            t_points_neiboour = torch.cdist(c_tgt_pcd, c_tgt_pcd) < patch_crop_thr
            s2t_close_feat = feat_sim > feat_sim_thr
            s2t_subspace = torch.einsum("st,tm->sm", s2t_close_feat.float(), t_points_neiboour.float()) > 0
            t2s_subspace = torch.einsum("st,sm->mt", s2t_close_feat.float(), s_points_neiboour.float()) > 0
            
            
            random_idx = torch.randperm(src_2d_feat.shape[0])[:1]
            # random_idx = torch.tensor([1])
            psc = PSC().vedo(subplot=4).add_pcd(torch.cat([src_, tgt_]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            tgt_correspondence = [torch.arange(tgt_2d_feat.shape[0])[i] for i in feat_sim[random_idx]>feat_sim_thr]
            src_correspondence = [i.repeat(corr.shape[0]) for i, corr in zip(random_idx, tgt_correspondence)]
            tgt_color = torch.zeros_like(tgt_)
            for src_corr, tgt_corr in zip(src_correspondence, tgt_correspondence):
                if src_corr.shape[0] and tgt_corr.shape[0]:
                    psc.add_lines(src_[src_corr], tgt_[tgt_corr], colors=np.array([1,0,0]))
                    for each in (torch.cdist(tgt_[tgt_corr], tgt_) < patch_crop_thr):
                        tgt_color[each] = torch.tensor([1,0,0]).float()
            psc.add_color(torch.cat([torch.zeros_like(src_), tgt_color]))
            psc.draw_at(1).add_pcd(torch.cat([src_, tgt_]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            tgt_correspondence = [torch.arange(tgt_2d_feat.shape[0])[i] for i in s2t_subspace[random_idx]]
            src_correspondence = [i.repeat(corr.shape[0]) for i, corr in zip(random_idx, tgt_correspondence)]
            tgt_color = torch.zeros_like(tgt_)
            for src_corr, tgt_corr in zip(src_correspondence, tgt_correspondence):
                if src_corr.shape[0] and tgt_corr.shape[0]:
                    psc.add_lines(src_[src_corr], tgt_[tgt_corr], colors=np.array([1,0,0]))
                    tgt_color[tgt_corr] = torch.tensor([1,0,0]).float()
            psc.add_color(torch.cat([torch.zeros_like(src_), tgt_color]))
            
            psc.draw_at(2).add_pcd(torch.cat([src_, tgt_]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            src_correspondence = [torch.arange(src_2d_feat.shape[0])[i] for i in feat_sim.T[random_idx]>feat_sim_thr]
            tgt_correspondence = [i.repeat(corr.shape[0]) for i, corr in zip(random_idx, src_correspondence)]
            src_color = torch.zeros_like(src_)
            for src_corr, tgt_corr in zip(src_correspondence, tgt_correspondence):
                if src_corr.shape[0] and tgt_corr.shape[0]:
                    psc.add_lines(src_[src_corr], tgt_[tgt_corr], colors=np.array([1,0,0]))
                    for each in (torch.cdist(src_[src_corr], src_) < patch_crop_thr):
                        src_color[each] = torch.tensor([1,0,0]).float()
            psc.add_color(torch.cat([src_color, torch.zeros_like(tgt_)]))
            psc.draw_at(3).add_pcd(torch.cat([src_, tgt_]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            src_correspondence = [torch.arange(src_2d_feat.shape[0])[i] for i in t2s_subspace.T[random_idx]]
            tgt_correspondence = [i.repeat(corr.shape[0]) for i, corr in zip(random_idx, src_correspondence)]
            src_color = torch.zeros_like(src_)
            for src_corr, tgt_corr in zip(src_correspondence, tgt_correspondence):
                if src_corr.shape[0] and tgt_corr.shape[0]:
                    psc.add_lines(src_[src_corr], tgt_[tgt_corr], colors=np.array([1,0,0]))
                    src_color[src_corr] = torch.tensor([1,0,0]).float()
                    
            psc.add_color(torch.cat([src_color, torch.zeros_like(tgt_)]))
            psc.show()
            #########################################################################################################
            
            src_ = apply_transform(c_src_pcd, tsfms[entry_id].clone())
            tgt_ = c_tgt_pcd+torch.Tensor([3,0,0])
            
            src_2d_feat = src_pcd_2dfeat[src_mapping_ind]
            tgt_2d_feat = tgt_pcd_2dfeat[tgt_mapping_ind]

            src_2d_feat_norm = torch.nn.functional.normalize(src_2d_feat)
            tgt_2d_feat_norm = torch.nn.functional.normalize(tgt_2d_feat)
            
            feat_sim = torch.einsum("nd,md->nm", src_2d_feat_norm, tgt_2d_feat_norm)
            
            psc = PSC().vedo().add_pcd(torch.cat([src_, tgt_]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            src_color, tgt_color = torch.zeros_like(src_), torch.zeros_like(tgt_)

            feat_sim -= feat_sim.min()
            feat_sim /= feat_sim.max()
            idx = 0
            
            src_color[idx] = torch.tensor([1,0,0]).float()
            tgt_color = torch.tensor([1,0,0]).float() * feat_sim[idx, None].T
            psc.add_color(torch.cat([src_color, tgt_color]))
            
            psc.show()
            
            
            #########################################################################################################
            # Guassian
            src_ = apply_transform(c_src_pcd, tsfms[entry_id].clone())
            tgt_ = c_tgt_pcd+torch.Tensor([3,0,0])
            
            src_2d_feat = src_pcd_2dfeat[src_mapping_ind]
            tgt_2d_feat = tgt_pcd_2dfeat[tgt_mapping_ind]

            src_2d_feat_norm = torch.nn.functional.normalize(src_2d_feat)
            tgt_2d_feat_norm = torch.nn.functional.normalize(tgt_2d_feat)
            
            feat_sim_thr = 0.50
            feat_sim = torch.einsum("nd,md->nm", src_2d_feat_norm, tgt_2d_feat_norm)
            s2t_close_feat = feat_sim > feat_sim_thr
            
            
            src_dists = torch.exp(15*-torch.cdist(c_src_pcd, c_src_pcd))
            tgt_dists = torch.exp(15*-torch.cdist(c_tgt_pcd, c_tgt_pcd))
            
            s2t_subspace = torch.einsum("st,tm->sm", s2t_close_feat.float(), tgt_dists)
            s2t_subspace = s2t_subspace / (s2t_subspace.max(1).values[None, ...].T+1e-12)
            
            t2s_subspace = torch.einsum("st,sm->tm", s2t_close_feat.float(), src_dists)
            t2s_subspace = t2s_subspace / (t2s_subspace.max(1).values[None, ...].T+1e-12)
            
            idx = 0
            # idx = torch.randint(src_.shape[0], (1,)).item()
            psc = PSC().vedo(subplot=2).add_pcd(torch.cat([src_, tgt_]))#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            src_color, tgt_color = torch.zeros_like(src_), torch.zeros_like(tgt_)
            src_color[idx] = torch.tensor([1,0,0]).float()
            tgt_color = torch.tensor([1,0,0]).float() * s2t_subspace[idx, None].T
            psc.add_color(torch.cat([src_color, tgt_color]))
            
            src_color, tgt_color = torch.zeros_like(src_), torch.zeros_like(tgt_)
            tgt_color[idx] = torch.tensor([1,0,0]).float()
            src_color = torch.tensor([1,0,0]).float() * t2s_subspace[idx, None].T
            psc.draw_at(1).add_pcd(torch.cat([src_, tgt_]))
            psc.add_color(torch.cat([src_color, tgt_color]))
            
            psc.show()
            
            #########################################################################################################

            #########################################################################################################
            # Guassian (paper vis)
            
            src_dense = apply_transform(src_pcd_list[0].float(), tsfms[entry_id].clone())
            tgt_dense = tgt_pcd_list[0].float() # +torch.Tensor([3,0,0])

            src_ = apply_transform(c_src_pcd, tsfms[entry_id].clone())
            tgt_ = c_tgt_pcd # + torch.Tensor([3,0,0])

            # s_c2d_inds = torch.cdist(src_dense, src_).topk(1, largest=False)
            # s_c2d_inds = s_c2d_inds.indices[s_c2d_inds.values<0.01]
            # t_c2d_inds = torch.cdist(tgt_dense, tgt_).topk(1, largest=False)
            # t_c2d_inds = t_c2d_inds.indices[t_c2d_inds.values<0.01]

            s_c2d_inds = torch.cdist(src_dense, src_).topk(5, largest=False).indices
            t_c2d_inds = torch.cdist(tgt_dense, tgt_).topk(5, largest=False).indices
            
            src_2d_feat = src_pcd_2dfeat[src_mapping_ind]
            tgt_2d_feat = tgt_pcd_2dfeat[tgt_mapping_ind]

            src_2d_feat_norm = torch.nn.functional.normalize(src_2d_feat)
            tgt_2d_feat_norm = torch.nn.functional.normalize(tgt_2d_feat)
            
            feat_sim_thr = 0.5
            feat_sim = torch.einsum("nd,md->nm", src_2d_feat_norm, tgt_2d_feat_norm)
            # s2t_close_feat = feat_sim > feat_sim_thr
            s2t_close_feat = feat_sim
            s2t_close_feat[feat_sim < feat_sim_thr] = 0
            idx = 1
            # idx = torch.randint(src_.shape[0], (1,)).item()
            psc = PSC().vedo(bg_color=[1,1,1], subplot=5)
            psc.draw_at(0).add_pcd(src_dense)#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
            src_color = torch.zeros_like(src_)
            src_color += 0.1
            src_color[idx] = torch.tensor([1, 0, 0]).float()
            src_feats_color_mean = src_color[s_c2d_inds].mean(1)
            src_feats_color_mean /= src_feats_color_mean.max()
            psc.add_color(src_feats_color_mean)
            
            r = 0.1
            def vis(r, draw_at):
                tgt_dists = torch.exp((1/r**2)*-torch.cdist(c_tgt_pcd, c_tgt_pcd))            
                s2t_subspace = torch.einsum("st,tm->sm", s2t_close_feat.float(), tgt_dists)
                # s2t_subspace = s2t_subspace - (s2t_subspace.min(1).values[None, ...].T)
                s2t_subspace = s2t_subspace / (s2t_subspace.max(1).values[None, ...].T+1e-12)
                psc.draw_at(draw_at).add_pcd(tgt_dense)#.add_feat(torch.cat([src_2d_feat, tgt_2d_feat]))
                tgt_color = torch.zeros_like(tgt_)
                tgt_color = torch.tensor([1, 0.0, 0.0]).float() * s2t_subspace[idx, None].T
                tgt_color[:, 1:] = 0.1
                
                tgt_feats_color_mean = tgt_color[t_c2d_inds].mean(1)
                tgt_feats_color_mean /= tgt_feats_color_mean.max()
                psc.add_color(tgt_feats_color_mean)

            vis(0.01, 1)
            vis(0.3, 2)
            vis(1.0, 3)
            vis(1.5, 4)
            psc.show()
            
            #########################################################################################################
            

            num_clusters = topk+1
            pcd_2dfeat = torch.cat([src_pcd_2dfeat, tgt_pcd_2dfeat])
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(pcd_2dfeat.numpy())
            
            src_cluster_ids_x, tgt_cluster_ids_x = kmeans.labels_[:src_pcd_2dfeat.shape[0]], kmeans.labels_[src_pcd_2dfeat.shape[0]:]
            
            src_pcd_color = torch.zeros_like(src_mapping_pcd)
            tgt_pcd_color = torch.zeros_like(tgt_mapping_pcd)
            
            for i in np.unique(src_cluster_ids_x):
                c = torch.rand((3))
                src_pcd_color[src_cluster_ids_x==i] = c
                tgt_pcd_color[tgt_cluster_ids_x==i] = c
                
            PSC().add_pcd(src_mapping_pcd, tsfms[entry_id].clone()).add_color(src_pcd_color)\
                .add_pcd(tgt_mapping_pcd+torch.Tensor([3,0,0])).add_color(tgt_pcd_color).show()
        ################################################################################################################################
        # Project features to 3D space
        c_src_pcd_2dfeat = src_pcd_2dfeat[src_mapping_ind]
        c_tgt_pcd_2dfeat = tgt_pcd_2dfeat[tgt_mapping_ind]

        src_2d_feat_norm = torch.nn.functional.normalize(c_src_pcd_2dfeat)
        tgt_2d_feat_norm = torch.nn.functional.normalize(c_tgt_pcd_2dfeat)
            
        src_semantic_feat[entry_id, :c_src_pcd_2dfeat.shape[0]] = src_2d_feat_norm
        tgt_semantic_feat[entry_id, :c_tgt_pcd_2dfeat.shape[0]] = tgt_2d_feat_norm

        ################################################################################################################################
        # Project masks to 3D space
        # mutual_mask_labels = mutual_mask_labels_list[entry_id]
        # all_mask_labels = all_mask_labels_list[entry_id]
        
        # # non_mutual_mask_labels = src_mask_label[((src_mask_label[:, None] - mutual_mask_labels[None, :])==0).sum(1)==0]
        
        # if mutual_mask_labels.shape[0] > 0:
        #     # ## Project the mask to source point cloud 
        #     # c_src_pcd_mask = project_mask2pcd(src_2d3d, c_src_pcd, mask_len, mutual_mask_labels)
        #     # ## Project the mask to target point cloud 
        #     # c_tgt_pcd_mask = project_mask2pcd(tgt_2d3d, c_tgt_pcd, mask_len, mutual_mask_labels)
        #     ## Project the mask to source point cloud 
        #     c_src_pcd_mask = src_pcd_mask[:, src_mapping_ind]
        #     ## Project the mask to target point cloud 
        #     c_tgt_pcd_mask = tgt_pcd_mask[:, tgt_mapping_ind]

        #     # Select top-k largest region and merge the rest
        #     mutual_mask_labels_num = mutual_mask_labels.shape[0]
            
        #     hard_mask_thr = config.semantic_mask_thr
        #     ## Get source average region
        #     c_src_pcd_mask_hard = c_src_pcd_mask>hard_mask_thr
        #     avg_region_size = c_src_pcd.shape[0] / mutual_mask_labels_num
        #     src_region_size = c_src_pcd_mask_hard.sum(1) / avg_region_size # normalize region size
        #     ## Get target average region
        #     c_tgt_pcd_mask_hard = c_tgt_pcd_mask > hard_mask_thr
        #     avg_region_size = c_tgt_pcd.shape[0] / mutual_mask_labels_num
        #     tgt_region_size = c_tgt_pcd_mask_hard.sum(1) / avg_region_size # normalize region size

        #     ## Get mutual largest region index        
        #     region_size_together = src_region_size + tgt_region_size
        #     region_sizes_sorted_ind = region_size_together.argsort()
            
        #     k = region_sizes_sorted_ind.shape[0]-1 if region_sizes_sorted_ind.shape[0] <= topk else topk
            
        #     largest_region_idx = region_sizes_sorted_ind[-k:]
        #     ## Select source multual largest region
        #     src_pcd_mask_largest_region = c_src_pcd_mask[largest_region_idx] 
        #     ## Select target multual largest region
        #     tgt_pcd_mask_largest_region = c_tgt_pcd_mask[largest_region_idx] 
        #     if k > 0:
        #         # merge the rest
        #         rest_region_idx = region_sizes_sorted_ind[:-k]
        #         src_pcd_mask_rest_region = np.clip(c_src_pcd_mask[rest_region_idx].sum(axis=0, keepdims=True), 0, 1)
        #         tgt_pcd_mask_rest_region = np.clip(c_tgt_pcd_mask[rest_region_idx].sum(axis=0, keepdims=True), 0, 1)
        #     else:
        #         src_pcd_mask_rest_region = np.zeros_like(c_src_pcd_mask[:1])
        #         tgt_pcd_mask_rest_region = np.zeros_like(c_tgt_pcd_mask[:1])
            
        #     src_pcd_mask_merged = np.concatenate([
        #         src_pcd_mask_largest_region, 
        #         src_pcd_mask_rest_region,
        #         # (c_src_pcd_mask_hard.sum(0)==0)[None, ...] # The region outside of mutual region
        #     ])
            
        #     tgt_pcd_mask_merged = np.concatenate([
        #         tgt_pcd_mask_largest_region, 
        #         tgt_pcd_mask_rest_region,
        #         # (c_tgt_pcd_mask_hard.sum(0)==0)[None, ...] # The region outside of mutual region
        #     ])
        #     src_semantic_masks[entry_id, :src_pcd_mask_merged.shape[0], :n_s_pts] = torch.from_numpy(src_pcd_mask_merged)
        #     tgt_semantic_masks[entry_id, :tgt_pcd_mask_merged.shape[0], :n_t_pts] = torch.from_numpy(tgt_pcd_mask_merged)

        # src_semantic_masks_temp = rearrange(src_semantic_masks, 'b s m -> (b s) m')
        # tgt_semantic_masks_temp = rearrange(tgt_semantic_masks, 'b s n -> (b s) n')
        
        # src_semantic_masks_hard = src_semantic_masks_temp > config.semantic_mask_thr
        # tgt_semantic_masks_hard = tgt_semantic_masks_temp > config.semantic_mask_thr

        # semantic_masks_valid = ((src_semantic_masks_hard.sum(dim=-1) > 0) & (tgt_semantic_masks_hard.sum(dim=-1) > 0))
        
        # if not semantic_masks_valid.any():
        #     num_clusters = 4
        #     pcd_2dfeat = torch.cat([c_src_pcd_2dfeat, c_tgt_pcd_2dfeat])
            
        #     kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(pcd_2dfeat.numpy())
        #     src_cluster_ids_x, tgt_cluster_ids_x = kmeans.labels_[:c_src_pcd_2dfeat.shape[0]], kmeans.labels_[c_src_pcd_2dfeat.shape[0]:]

        #     src_pcd_mask_merged = np.stack([src_cluster_ids_x==i for i in range(num_clusters)])
        #     tgt_pcd_mask_merged = np.stack([tgt_cluster_ids_x==i for i in range(num_clusters)])

        #     src_semantic_masks[entry_id, :src_pcd_mask_merged.shape[0], :n_s_pts] = torch.from_numpy(src_pcd_mask_merged)
        #     tgt_semantic_masks[entry_id, :tgt_pcd_mask_merged.shape[0], :n_t_pts] = torch.from_numpy(tgt_pcd_mask_merged)

        # print("elapse: {}".format(time.time() - start_time))
        ################################################################################################################################
        
        VIS_DEPTH_PCD = False
        if VIS_DEPTH_PCD:
            src_tsfm = tsfms[entry_id].clone()
            psc = PSC().add_pcd(c_src_pcd, src_tsfm).add_color(np.zeros_like(c_src_pcd))
            for i in src_2d3d:
                mask_projection = i["mask_projection"]
                depth_mask_image = i["depth_mask_image"]
                world2camera = i["world2camera"]
                depth_pcd = unproject((depth_mask_image[0]*1000.0).detach().cpu().numpy(), 
                        mask_projection.intrinsics, 
                        world2camera.inverse().cpu())
                psc.add_pcd(depth_pcd, src_tsfm)
        
            offset = np.array([0, 0, 0])
            psc.add_pcd(c_tgt_pcd+offset).add_color(np.zeros_like(c_tgt_pcd))
            for i in tgt_2d3d:
                mask_projection = i["mask_projection"]
                depth_mask_image = i["depth_mask_image"]
                world2camera = i["world2camera"]
                depth_pcd = unproject((depth_mask_image[0]*1000.0).detach().cpu().numpy(), 
                        mask_projection.intrinsics, 
                        world2camera.inverse().cpu())
                psc.add_pcd(depth_pcd+offset)
            psc.show()
            
        VIS_DEPTH_FEAT_PCD = False
        if VIS_DEPTH_FEAT_PCD:
            src_tsfm = tsfms[entry_id].clone()
            psc = PSC().add_pcd(c_src_pcd, src_tsfm).add_color(np.zeros_like(c_src_pcd))
            for i in src_2d3d:
                feat_projection = i["feat_projection"]
                depth_feat_image = i["depth_feat_image"]
                world2camera = i["world2camera"]
                depth_pcd = unproject((depth_feat_image[0]*1000.0).detach().cpu().numpy(), 
                        feat_projection.intrinsics, 
                        world2camera.inverse().cpu())
                psc.add_pcd(depth_pcd, src_tsfm)
        
            offset = np.array([0, 0, 0])
            psc.add_pcd(c_tgt_pcd+offset).add_color(np.zeros_like(c_tgt_pcd))
            for i in tgt_2d3d:
                feat_projection = i["feat_projection"]
                depth_feat_image = i["depth_feat_image"]
                world2camera = i["world2camera"]
                depth_pcd = unproject((depth_feat_image[0]*1000.0).detach().cpu().numpy(), 
                        feat_projection.intrinsics, 
                        world2camera.inverse().cpu())
                psc.add_pcd(depth_pcd+offset)
            psc.show()

        VIS_FEAT_IMAGE_PCD = False
        if VIS_FEAT_IMAGE_PCD:
            src_tsfm = tsfms[entry_id].clone()
            c_src_pcd_color = np.zeros_like(c_src_pcd)
            psc = PSC().add_pcd(c_src_pcd, src_tsfm).add_color(c_src_pcd_color.copy())
            for i in src_2d3d:
                projection = i["feat_projection"]
                depth_image = i["depth_feat_image"]
                world2camera = i["world2camera"]
                color_image = np.array(Image.open(i["color_path"]).resize((depth_image.shape[2], depth_image.shape[1])))
                color_image = np.ones_like(color_image)
                
                inds2d, inds3d = projection.projection(
                    c_src_pcd, depth_image, world2camera)
                
                c_src_pcd_color[inds3d] = color_image[inds2d[:, 1], inds2d[:, 0]]
            
            psc.add_color(c_src_pcd_color)
        
            offset = np.array([3, 0, 0])
            c_tgt_pcd_color = np.zeros_like(c_tgt_pcd)
            
            psc.add_pcd(c_tgt_pcd+offset).add_color(c_tgt_pcd_color.copy())
            for i in tgt_2d3d:
                
                projection = i["feat_projection"]
                depth_image = i["depth_feat_image"]
                world2camera = i["world2camera"]
                color_image = np.array(Image.open(i["color_path"]).resize((depth_image.shape[2], depth_image.shape[1])))
                color_image = np.ones_like(color_image)
                
                inds2d, inds3d = projection.projection(
                    c_tgt_pcd, depth_image, world2camera)
                
                c_tgt_pcd_color[inds3d] = color_image[inds2d[:, 1], inds2d[:, 0]]
            
            psc.add_color(c_tgt_pcd_color)
            psc.show()

        VIS_SEM_MASK = False
        if VIS_SEM_MASK:
            psc = PSC().vedo()
            src_tsfm = tsfms[entry_id].clone()
            
            offset = np.array([3, 0, 0])
            src_salient_mask = src_pcd_mask_merged.argmax(0)
            tgt_salient_mask = tgt_pcd_mask_merged.argmax(0)
            src_c = np.zeros_like(c_src_pcd)
            tgt_c = np.zeros_like(c_tgt_pcd)
            for i in range(src_pcd_mask_merged.shape[0]):
                src_c[src_salient_mask==i] = np.array(ADE20K_COLORMAP[mutual_mask_labels[i]+1]) / 255
                tgt_c[tgt_salient_mask==i] = np.array(ADE20K_COLORMAP[mutual_mask_labels[i]+1]) / 255
            psc.add_pcd(c_src_pcd, src_tsfm).add_color(src_c).add_pcd(c_tgt_pcd+offset).add_color(tgt_c)
            
            src_iter_offset = np.array([0, 3, 0])[..., None]
            tgt_iter_offset = np.array([0, 3, 0])
            tgt_offset = np.array([3, 0, 0])
            src_tsfm = tsfms[entry_id].clone()
            
            for ind, (src_mask, tgt_mask) in enumerate(zip(src_pcd_mask_merged, tgt_pcd_mask_merged)):
                src_mask_hard = src_mask>hard_mask_thr
                tgt_mask_hard = tgt_mask>hard_mask_thr
                
                if (tgt_mask_hard).sum(0) and (tgt_mask_hard).sum(0):
                    src_pcd_subspace = c_src_pcd[src_mask_hard]
                    tgt_pcd_subspace = c_tgt_pcd[tgt_mask_hard]
                    
                    color = np.array(ADE20K_COLORMAP[mutual_mask_labels[ind]+1]) / 255
                    src_c = np.zeros_like(src_pcd_subspace) + color
                    src_c *= src_mask[src_mask_hard][..., None]
                    tgt_c = np.zeros_like(tgt_pcd_subspace) + color
                    tgt_c *= tgt_mask[tgt_mask_hard][..., None]
                
                    src_tsfm[:3, -1:] += src_iter_offset
                    tgt_offset += tgt_iter_offset
                    
                    psc.add_pcd(src_pcd_subspace, src_tsfm).add_color(src_c)
                    psc.add_pcd(tgt_pcd_subspace+tgt_offset).add_color(tgt_c)
                
            psc.show()
            

        vis=False # for debug
        if vis :
            viz_coarse_nn_correspondence_mayavi(c_src_pcd, c_tgt_pcd, coarse_match_gt, scale_factor=0.04)




        vis=False # for debug
        if vis :
            pass
            import mayavi.mlab as mlab

            # src_nei_valid = src_nei_mask[coarse_match_gt[0]].view(-1)
            # tgt_nei_valid = tgt_nei_mask[coarse_match_gt[1]].view(-1)
            #
            # f_src_pcd = src_m_nei_pts.view(-1, 3)[src_nei_valid]
            # f_tgt_pcd = tgt_m_nei_pts.view(-1,3)[tgt_nei_valid]
            #
            # mlab.points3d(f_src_pcd[:, 0], f_src_pcd[:, 1], f_src_pcd[:, 2], scale_factor=0.02,color=c_gray1)
            # mlab.points3d(f_tgt_pcd[:, 0], f_tgt_pcd[:, 1], f_tgt_pcd[:, 2], scale_factor=0.02,color=c_gray2)
            #
            # src_m_nn_pts =src_m_nn_pts.view(-1, 3)
            # src_m_nn_pts_wrapped = src_m_nn_pts_wrapped.view(-1,3)
            # tgt_m_nn_pts =  tgt_m_nei_pts [ torch.arange(tgt_m_nei_pts.shape[0]), nni.view(-1), ... ]
            # mlab.points3d(src_m_nn_pts[:, 0], src_m_nn_pts[:, 1], src_m_nn_pts[:, 2], scale_factor=0.04,color=c_red)
            # mlab.points3d(src_m_nn_pts_wrapped[:, 0], src_m_nn_pts_wrapped[:, 1], src_m_nn_pts_wrapped[:, 2], scale_factor=0.04,color=c_red)
            # mlab.points3d(tgt_m_nn_pts[:, 0], tgt_m_nn_pts[:, 1], tgt_m_nn_pts[:, 2], scale_factor=0.04 ,color=c_blue)
            # mlab.show()
            # viz_coarse_nn_correspondence_mayavi(c_src_pcd, c_tgt_pcd, coarse_match_gt,
            #                                     f_src_pcd=src_m_nei_pts.view(-1,3)[src_nei_valid],
            #                                     f_tgt_pcd=tgt_m_nei_pts.view(-1,3)[tgt_nei_valid], scale_factor=0.08)



    src_ind_coarse_split = torch.cat(src_ind_coarse_split)
    tgt_ind_coarse_split = torch.cat(tgt_ind_coarse_split)
    src_ind_coarse = torch.cat(src_ind_coarse)
    tgt_ind_coarse = torch.cat(tgt_ind_coarse)


    dict_inputs = {
        'src_pcd_list': src_pcd_list,
        'tgt_pcd_list': tgt_pcd_list,
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'coarse_matches': coarse_matches,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_ind_coarse_split': src_ind_coarse_split,
        'tgt_ind_coarse_split': tgt_ind_coarse_split,
        'src_ind_coarse': src_ind_coarse,
        'tgt_ind_coarse': tgt_ind_coarse,
        'batched_rot': batched_rot,
        'batched_trn': batched_trn,
        'gt_cov': gt_cov_list,
        #for refine
        'correspondences_list': correspondences_list,
        'fine_ind': fine_ind,
        'fine_pts': fine_pts,
        'fine_length': fine_length,

        # "src_semantic_masks": src_semantic_masks,
        # "tgt_semantic_masks": tgt_semantic_masks,
        "tsfms": tsfms,
        
        "src_2dfeat": src_semantic_feat,
        "tgt_2dfeat": tgt_semantic_feat,
    }

    return dict_inputs

def project_mask2pcd(info_2d3d, pcd, mask_len, selected_mask_labels):
    ## Project the mask to source point cloud 
    pcd_mask = np.zeros((mask_len, pcd.shape[0]))
    for frame in info_2d3d:
        mask = frame["mask"]
        projection = frame["mask_projection"]
        depth_image = frame["depth_mask_image"]
        world2camera = frame["world2camera"]
        
        inds2d_mask, inds3d_mask = projection.projection(
            pcd, depth_image, world2camera)
        
        pcd_mask[:, inds3d_mask] += mask[:, inds2d_mask[:, 1], inds2d_mask[:, 0]]
    
    # Only select the multual masks
    pcd_mask = pcd_mask[selected_mask_labels]
    pcd_mask = pcd_mask.clip(0, 255)
    # Normalize to 0-1
    pcd_mask = pcd_mask - pcd_mask.min() 
    pcd_mask = pcd_mask / pcd_mask.max() 
    return pcd_mask

def project_feat2pcd(info_2d3d, pcd):
    ## Project the feat to source point cloud 
    
    pcd_2d_feat = np.zeros((pcd.shape[0], info_2d3d[0]["feat_local"].shape[-1]))
    
    for frame in info_2d3d:
        feat_local = frame["feat_local"]
        projection = frame["feat_projection"]
        depth_image = frame["depth_feat_image"]
        world2camera = frame["world2camera"]
        
        inds2d, inds3d = projection.projection(
            pcd, depth_image, world2camera)
        
        pcd_2d_feat[inds3d] = feat_local[0, inds2d[:, 0], inds2d[:, 1]]
    
    return pcd_2d_feat


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        # if timer.total_time - last_display > 0.1:
        #     last_display = timer.total_time
        #     print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits




def get_datasets(config):
    if (config.dataset == '3dmatch'):
        train_set = _3DMatch(config, 'train', data_augmentation=True)
        val_set = _3DMatch(config, 'val', data_augmentation=False)
        test_set = _3DMatch(config, 'test', data_augmentation=False)
    else:
        raise NotImplementedError

    return train_set, val_set, test_set



def get_dataloader(dataset, config, shuffle=True, neighborhood_limits=None):

    if config.dataset == '3dmatch':
        collate_fn = collate_fn_3dmatch
    else:
        raise NotImplementedError()

    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config['kpfcn_config'], collate_fn=collate_fn)
    print("neighborhood:", neighborhood_limits)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers'],
        collate_fn=partial(collate_fn, config=config['kpfcn_config'], neighborhood_limits=neighborhood_limits ),
        drop_last=False
    )

    return dataloader, neighborhood_limits




if __name__ == '__main__':


    pass
