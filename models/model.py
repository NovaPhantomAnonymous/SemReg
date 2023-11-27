import torch
from models.loss import MatchMotionLoss as MML
import numpy as np
from models.matching import Matching as CM
import math
from pointscope import PointScopeClient as PSC
from lib.utils import apply_transform
from models.subspace_matching import subspace_matches_remap
import torch.nn.functional as F

import pytorch_lightning as pl
import os.path as osp
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import open3d as o3d
from datasets.dataloader import collate_fn_3dmatch

class Model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = self.cfg.model
        # self.save_hyperparameters()
        
    def configure_optimizers(self):
        return {
            "optimizer": self.cfg.optimizer,
            "lr_scheduler": {
                "scheduler": self.cfg.scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def train_dataloader(self):
        return self.cfg.train_loader

    def val_dataloader(self):
        return self.cfg.val_loader

    def test_dataloader(self):
        return self.cfg.test_loader

    def training_step(self, batch, batch_idx) -> dict:
        outputs = self.model(batch)
        loss = self.cfg.desc_loss(outputs)
        for l in loss:
            if l !="loss" and loss[l].requires_grad:
                loss[l] = loss[l].detach()
        batch_size = len(batch["src_pcd_list"])
        self.log_dict(loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self.model(val_batch)
        loss = self.cfg.desc_loss(outputs)
        batch_size = len(val_batch["src_pcd_list"])
        self.log_dict(loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)

    def on_test_start(self):
        self.save_data = []

    def update_batch(self, batch_data, rot, trn):
        tsfm = torch.eye(4)
        tsfm[:3, :3] = rot
        tsfm[:3, -1:] = trn
        tsfm = tsfm.to(batch_data["points"][0].device)
        stack_lengths = batch_data["stack_lengths"]
        batch_data["points"] = [torch.cat([apply_transform(points[:stack_length[0]], tsfm), points[stack_length[0]:],]) for points, stack_length in zip(batch_data["points"], stack_lengths)]
        batch_data['s_pcd'] = apply_transform(batch_data['s_pcd'], tsfm)
    
    def visualization_step(self, batch, batch_idx):
        return

        data = self.model(batch)
        s_pcd = data["s_pcd"][0].cpu()
        t_pcd = data["t_pcd"][0].cpu()
        
        
        ##################################################################################################################################
        src_2dfeat = batch["src_2dfeat"][0].cpu()
        tgt_2dfeat = batch["tgt_2dfeat"][0].cpu()

        s_pcd_gt = apply_transform(s_pcd, batch["tsfms"][0].cpu())
        
        dense_points = data["points"][0].cpu()
        src_dense_points_len = data["stack_lengths"][0][0]
        src_dense_points, tgt_dense_points = dense_points[:src_dense_points_len], dense_points[src_dense_points_len:]
        
        src_dense_points_gt = apply_transform(src_dense_points, batch["tsfms"][0].cpu())
        
        feat_tsne = TSNE(n_components=3, 
                         learning_rate='auto', 
                         init='random').fit_transform(torch.cat([src_2dfeat, tgt_2dfeat]))
        feat_tsne -= feat_tsne.min()
        feat_tsne /= feat_tsne.max()
        
        src_feats_color = feat_tsne[:src_2dfeat.shape[0]]
        tgt_feats_color = feat_tsne[src_2dfeat.shape[0]:]
        
        s_c2d_inds = torch.cdist(src_dense_points, s_pcd).topk(5, largest=False).indices
        t_c2d_inds = torch.cdist(tgt_dense_points, t_pcd).topk(5, largest=False).indices
        
        src_feats_color_mean = src_feats_color[s_c2d_inds].mean(1)
        tgt_feats_color_mean = tgt_feats_color[t_c2d_inds].mean(1)
        
        PSC().vedo(bg_color=[1,1,1], subplot=2)\
            .add_pcd(src_dense_points_gt)\
            .add_color(src_feats_color_mean)\
            .draw_at(1)\
            .add_pcd(tgt_dense_points+torch.tensor([0, 0, 2]))\
            .add_color(tgt_feats_color_mean)\
            .show()
        
        ##################################################################################################################################
        
        
        
        
        ##################################################################################################################################
        # Visualize SemReg feature with correspondence 
        data = self.model(batch)
        s_pcd = data["s_pcd"][0].cpu()
        t_pcd = data["t_pcd"][0].cpu()
        
        
        s_pcd_gt = apply_transform(s_pcd, batch["tsfms"][0].cpu())
        
        dense_points = data["points"][0].cpu()
        src_dense_points_len = data["stack_lengths"][0][0]
        src_dense_points, tgt_dense_points = dense_points[:src_dense_points_len], dense_points[src_dense_points_len:]
        src_dense_points_gt = apply_transform(src_dense_points, batch["tsfms"][0].cpu())


        src_feats = data["src_feats"][0].cpu()
        tgt_feats = data["tgt_feats"][0].cpu()


        conf_matrix_pred = data['conf_matrix_pred']
        match_pred, _, _ = CM.get_match(conf_matrix_pred, thr=self.cfg.conf_threshold, mutual=False)
        
        src_ind = match_pred[:, 1][:25]
        tgt_ind = match_pred[:, 2][:25]
        
        outlier = (torch.cdist(s_pcd_gt[src_ind], t_pcd[tgt_ind]).diag() > 0.1).nonzero().flatten()
        inlier = (torch.cdist(s_pcd_gt[src_ind], t_pcd[tgt_ind]).diag() < 0.1).nonzero().flatten()
        
        # PSC().add_pcd(s_pcd).add_pcd(t_pcd+torch.tensor([3, 0, 0])).add_lines(s_pcd[src_ind], (t_pcd+torch.tensor([3, 0, 0]))[tgt_ind]).show()
        
        feat_tsne = TSNE(n_components=3, 
                         learning_rate='auto', 
                         init='random').fit_transform(torch.cat([src_feats, tgt_feats]))
        feat_tsne -= feat_tsne.min()
        feat_tsne /= feat_tsne.max()
        
        src_feats_color = feat_tsne[:src_feats.shape[0]]
        tgt_feats_color = feat_tsne[src_feats.shape[0]:]
        
        s_c2d_inds = torch.cdist(src_dense_points_gt, s_pcd_gt).topk(5, largest=False).indices
        t_c2d_inds = torch.cdist(tgt_dense_points, t_pcd).topk(5, largest=False).indices
        
        src_feats_color_mean = src_feats_color[s_c2d_inds].mean(1)
        tgt_feats_color_mean = tgt_feats_color[t_c2d_inds].mean(1)
        
        # offset = torch.tensor([1,-0.5,1])
        offset = torch.tensor([3,0,0])
        psc = PSC().vedo(bg_color=[1,1,1], subplot=2)\
            .add_pcd(torch.cat([src_dense_points_gt, tgt_dense_points+offset]))\
            .add_color(np.concatenate([src_feats_color_mean, tgt_feats_color_mean]))\
            .add_lines(s_pcd_gt[src_ind[outlier]], (t_pcd+offset)[tgt_ind[outlier]], colors=[1,0,0])\
            .add_lines(s_pcd_gt[src_ind[inlier]], (t_pcd+offset)[tgt_ind[inlier]], colors=[0,1,0])

        result_path = "/home/sheldonfung/Proj/semreg_compare/PCR-CG/outputs/3DMatch/1_gpu_2_img_initmode_pri3d_256_gamma0.95_lr0.005_finalfeatsdim_32/test/pth/655.pth"

        data = torch.load(result_path)

        pcd = data['pcd']
        feats = data['feats']
        overlaps = data['overlaps']
        saliency = data['saliency']
        len_src = data['len_src']
        rot = data['rot']
        trans = data['trans']
        tsfm = torch.eye(4)
        tsfm[:3, :3] = rot
        tsfm[:3, -1:] = trans
        
        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_feat, tgt_feat = feats[:len_src], feats[len_src:]
        
        src_pcd_gt = apply_transform(src_pcd, batch["tsfms"][0].cpu())

        src_overlap, src_saliency = overlaps[:len_src], saliency[:len_src]
        tgt_overlap, tgt_saliency = overlaps[len_src:], saliency[len_src:]

        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency

        src_idx = src_scores.topk(5000, dim=0)[1]
        tgt_idx = tgt_scores.topk(6000, dim=0)[1]

        src_feat_ = src_feat[src_idx]
        tgt_feat_ = tgt_feat[tgt_idx]

        feat_dist = torch.cdist(src_feat_, tgt_feat_)
        a = feat_dist.topk(1, largest=False)
        b = a.values.flatten().topk(1000, largest=False)
        src_ind = b.indices[torch.randperm(1000)[:25]]
        tgt_ind = a.indices[src_ind].flatten()
        
        outlier = (torch.cdist(src_pcd_gt[src_idx][src_ind], tgt_pcd[tgt_idx][tgt_ind]).diag() > 0.6).nonzero().flatten()
        inlier = (torch.cdist(src_pcd_gt[src_idx][src_ind], tgt_pcd[tgt_idx][tgt_ind]).diag() < 0.6).nonzero().flatten()

        # PSC().vedo().add_pcd(src_pcd).add_pcd(tgt_pcd+torch.tensor([3, 0, 0]))\
        #     .add_lines(src_pcd[src_idx][src_ind], (tgt_pcd+torch.tensor([3, 0, 0]))[tgt_idx][tgt_ind], colors=[1,0,0])\
        #     .add_lines(src_pcd[src_idx][src_ind][inlier[:, 0]], (tgt_pcd+torch.tensor([3, 0, 0]))[tgt_idx][tgt_ind][inlier[:, 1]], colors=[0,1,0])\
        #     .show()
        
        psc.draw_at(1).add_pcd(torch.cat([src_pcd_gt, tgt_pcd+offset])).add_feat(torch.cat([src_feat, tgt_feat]))\
            .add_lines(src_pcd_gt[src_idx][src_ind[outlier]], (tgt_pcd+offset)[tgt_idx][tgt_ind[outlier]], colors=[1,0,0])\
            .add_lines(src_pcd_gt[src_idx][src_ind[inlier]], (tgt_pcd+offset)[tgt_idx][tgt_ind[inlier]], colors=[0,1,0])\
            .show()

        ##################################################################################################################################


    def do_match(self, batch_data):
        data = self.model(batch_data)  # [N1, C1], [N2, C2]        
        feat_2d_sim = data["feat_2d_sim"]
        s_pcd = data["s_pcd"]
        t_pcd = data["t_pcd"]
        
        s2t_subspace = data["s2t_subspace"]
        t2s_subspace = data["t2s_subspace"]
            
        s2t_subspace = s2t_subspace > 0.4
        t2s_subspace = t2s_subspace.transpose(1,2) > 0.4
        
        conf_matrix_pred = data['conf_matrix_pred'] * ((s2t_subspace+t2s_subspace)>0).float()
        match_pred, _, _ = CM.get_match(conf_matrix_pred, thr=self.cfg.conf_threshold, mutual=False)
        
        rot, trn = MML.ransac_regist_coarse(data['s_pcd'], data['t_pcd'], data['src_mask'], data['tgt_mask'], match_pred)
        return rot, trn, match_pred, conf_matrix_pred

    def test_step(self, batch, batch_idx):

        data = self.model(batch)  # [N1, C1], [N2, C2]        
        feat_2d_sim = data["feat_2d_sim"]
        s_pcd = data["s_pcd"]
        t_pcd = data["t_pcd"]
        
        points = data["points"][0]
        src_points_len = data["stack_lengths"][0][0]
        src_points, tgt_points = points[:src_points_len], points[src_points_len:]
        
        src_mask = data["src_mask"]
        tgt_mask = data["tgt_mask"]

        patch_crop_thr = 0.5
        feat_sim_thr = 0.4
        
        s2t_subspace = data["s2t_subspace"]
        t2s_subspace = data["t2s_subspace"]
            
        # s_points_neiboour = torch.cdist(s_pcd, s_pcd) < patch_crop_thr
        # t_points_neiboour = torch.cdist(t_pcd, t_pcd) < patch_crop_thr
        # s2t_close_feat = feat_2d_sim > feat_sim_thr
        
        # s2t_subspace = torch.einsum("bst,btm->bsm", s2t_close_feat.float(), t_points_neiboour.float()) > 0
        # t2s_subspace = torch.einsum("bst,bsm->bmt", s2t_close_feat.float(), s_points_neiboour.float()) > 0
            
        # data['conf_matrix_pred'] = data['conf_matrix_pred'] * (feat_2d_sim>feat_sim_thr).float()
        # conf_matrix_pred = data['conf_matrix_pred']*s2t_subspace*t2s_subspace.transpose(1,2)
        
        # match_pred, _, _ = CM.get_match(conf_matrix_pred, thr=self.cfg.conf_threshold, mutual=False)
        # ir, match_pred = self.get_matches(data)
        
        s2t_subspace = s2t_subspace > 0.5
        t2s_subspace = t2s_subspace.transpose(1,2) > 0.5
        
        # match_pred_s2t, _, _ = CM.get_match(
        #     conf_matrix_pred * s2t_subspace.float(), thr=self.cfg.conf_threshold, mutual=False)
        # match_pred_t2s, _, _ = CM.get_match(
        #     conf_matrix_pred * t2s_subspace.float(), thr=self.cfg.conf_threshold, mutual=False)
        # match_pred = torch.cat([match_pred_s2t, match_pred_t2s])
        
        conf_matrix_pred = data['conf_matrix_pred'] * ((s2t_subspace+t2s_subspace)>0).float()
        # conf_matrix_pred = data['conf_matrix_pred']
        match_pred, _, _ = CM.get_match(conf_matrix_pred, thr=self.cfg.conf_threshold, mutual=False)
        
        
        # # rot, trn = MML.ransac_regist_coarse(data['s_pcd'], data['t_pcd'], data['src_mask'], data['tgt_mask'], match_pred)

        # rot, trn, match_pred, conf_matrix_pred = self.do_match(batch)
        # # s_pcd, t_pcd, src_mask, tgt_mask = batch['s_pcd'].clone(), batch['t_pcd'].clone(), batch['src_mask'].clone(), batch['tgt_mask'].clone()
        # # self.update_batch(batch, rot, trn)
        # # for _ in range(2):
        # #     rot, trn, ir, match_pred, conf_matrix_pred = self.do_match(batch)
        # #     self.update_batch(batch, rot, trn)
        
        rot, trn = MML.ransac_regist_coarse(s_pcd, t_pcd, src_mask, tgt_mask, match_pred)
        est_tsfm = torch.eye(4)
        est_tsfm[:3, :3] = rot
        est_tsfm[:3, -1:] = trn
        
        
        
        ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=0.1).mean()
        rr1 = MML.compute_registration_recall(rot, trn, batch, thr=0.2) # 0.2m

        self.log_dict({
            "RR": rr1,
            "IR": ir,
            "FMR": (ir>0.05).float(),
        }, prog_bar=True, logger=True, batch_size=len(rot))

        # self.save_data.append({
        #     "s_pcd": data["s_pcd"].cpu().numpy(),
        #     "t_pcd": data["t_pcd"].cpu().numpy(),
        #     "src_mask": data["src_mask"].cpu().numpy(),
        #     "tgt_mask": data["tgt_mask"].cpu().numpy(),
        #     "conf_matrix_pred": data["conf_matrix_pred"].cpu().numpy(),
        #     "subspace_conf_matrix_pred": data["subspace_conf_matrix_pred"].cpu().numpy() if "subspace_conf_matrix_pred" in data else None,
        #     "src_sub_ind_padded": data["src_sub_ind_padded"].cpu().numpy() if data["src_sub_ind_padded"] is not None else None,
        #     "tgt_sub_ind_padded": data["tgt_sub_ind_padded"].cpu().numpy() if data["tgt_sub_ind_padded"] is not None else None,
        #     "semantic_masks_valid": data["semantic_masks_valid"].cpu().numpy(),
        #     "src_semantic_masks": data["src_semantic_masks"].cpu().numpy(),
        #     "tgt_semantic_masks": data["tgt_semantic_masks"].cpu().numpy(),
        #     "batched_rot": data["batched_rot"].cpu().numpy(),
        #     "batched_trn": data["batched_trn"].cpu().numpy(),
        #     "gt_cov": data["gt_cov"],
        # })
        
        # self.save_data.append({
        #     "s_pcd": data["s_pcd"].cpu().numpy(),
        #     "t_pcd": data["t_pcd"].cpu().numpy(),
        #     "src_mask": data["src_mask"].cpu().numpy(),
        #     "tgt_mask": data["tgt_mask"].cpu().numpy(),
        #     "conf_matrix_pred": conf_matrix_pred.cpu().numpy(),
        #     # "s2t_subspace": s2t_subspace.cpu().numpy(),
        #     # "t2s_subspace": t2s_subspace.cpu().numpy(),
        #     "batched_rot": data["batched_rot"].cpu().numpy(),
        #     "batched_trn": data["batched_trn"].cpu().numpy(),
        #     "gt_cov": data["gt_cov"],
        #     "tsfms": data["tsfms"][0].cpu().numpy(),
        # })
        
        self.save_data.append({
            "s_pcd": s_pcd.cpu().numpy(),
            "t_pcd": t_pcd.cpu().numpy(),
            "src_feats": data["src_feats"].cpu().numpy(),
            "tgt_feats": data["tgt_feats"].cpu().numpy(),
            "src_points": src_points.cpu().numpy(), 
            "tgt_points": tgt_points.cpu().numpy(),
            "src_mask": src_mask.cpu().numpy(),
            "tgt_mask": tgt_mask.cpu().numpy(),
            "conf_matrix_pred": conf_matrix_pred.cpu().numpy(),
            "batched_rot": batch["batched_rot"].cpu().numpy(),
            "batched_trn": batch["batched_trn"].cpu().numpy(),
            "gt_cov": batch["gt_cov"],
            "tsfms": batch["tsfms"][0].cpu().numpy(),
            "est_tsfm": est_tsfm,
            "RR": rr1,
            "IR": ir,
            "FMR": (ir>0.05).float(),
        })
        
    def on_test_end(self):
        print("Saving validation data...")
        for ind, d in enumerate(tqdm(self.save_data)):
            path = osp.join(self.cfg.val_data_dir, f"{ind}.npz")
            np.savez_compressed(path, data=d)

    @torch.no_grad()
    def demo(self):
        
        neighborhood_limits = np.array([39, 36, 36, 38])

        src_pcd = torch.load(self.cfg.demo.src_prefix+".pth")
        src_feats = np.ones_like(src_pcd[:, :1])
        src_pcd_2d_feat = torch.load(self.cfg.demo.src_prefix+".semantic_feat.pth")

        tgt_pcd = torch.load(self.cfg.demo.tgt_prefix+".pth")
        tgt_feats = np.ones_like(tgt_pcd[:, :1])
        tgt_pcd_2d_feat = torch.load(self.cfg.demo.tgt_prefix+".semantic_feat.pth")
        
        correspondences = None
        rot = np.eye(3)
        trans = np.zeros([3, 1])

        gt_cov = None
        src_2d3d = (src_pcd_2d_feat["pcd"], src_pcd_2d_feat["pcd_2dfeat"])
        tgt_2d3d = (tgt_pcd_2d_feat["pcd"], tgt_pcd_2d_feat["pcd_2dfeat"])
        tsfm = np.eye(4)
        
        data_batch = collate_fn_3dmatch(
            [[src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, gt_cov, src_2d3d, tgt_2d3d, tsfm]],
            self.cfg.kpfcn_config,
            neighborhood_limits
        )
        
        data = self.model(data_batch)  # [N1, C1], [N2, C2]        
        feat_2d_sim = data["feat_2d_sim"]
        s_pcd = data["s_pcd"]
        t_pcd = data["t_pcd"]
        
        points = data["points"][0]
        src_points_len = data["stack_lengths"][0][0]
        src_points, tgt_points = points[:src_points_len], points[src_points_len:]
        
        src_mask = data["src_mask"]
        tgt_mask = data["tgt_mask"]

        s2t_subspace = data["s2t_subspace"]
        t2s_subspace = data["t2s_subspace"]
            
        s2t_subspace = s2t_subspace > 0.3
        t2s_subspace = t2s_subspace.transpose(1,2) > 0.3
        
        conf_matrix_pred = data['conf_matrix_pred'] * ((s2t_subspace+t2s_subspace)>0).float()
        match_pred, _, _ = CM.get_match(conf_matrix_pred, thr=self.cfg.conf_threshold, mutual=False)
        rot, trn = MML.ransac_regist_coarse(s_pcd, t_pcd, src_mask, tgt_mask, match_pred)
        est_tsfm = torch.eye(4)
        est_tsfm[:3, :3] = rot
        est_tsfm[:3, -1:] = trn
        return src_pcd, tgt_pcd, est_tsfm
        

    def test_file(self, val_dir):
        success1 = 0.
        success2 = 0.
        IR = 0.
        FMR = 0.
        rot_deg1 = 0.
        rot_deg2 = 0.
        trans1 = 0.
        trans2 = 0.
        print(f"Confident threshold: {self.cfg.conf_threshold}")
        val_dir_len = len(os.listdir(val_dir))
        corr_num = []
        for val_path in tqdm(os.listdir(val_dir)):
            data_l = np.load(osp.join(val_dir, val_path), allow_pickle=True)["data"].tolist()
            
            data = {}
            for key in data_l:
                if type(data_l[key]) == np.ndarray:
                    data[key] = torch.from_numpy(data_l[key])
            # ir, match_pred = self.get_matches(data)
            
            # s2t_subspace = data["s2t_subspace"]
            # t2s_subspace = data["t2s_subspace"]
            tsfms = data["tsfms"]
            
            # conf_threshold = 0.0005
            conf_matrix_pred = data['conf_matrix_pred']
            # match_pred_s2t, _, _ = CM.get_match(conf_matrix_pred * (s2t_subspace).float(), thr=conf_threshold, mutual=False)
            # match_pred_t2s, _, _ = CM.get_match(conf_matrix_pred * (t2s_subspace).float(), thr=conf_threshold, mutual=False)
            # match_pred = torch.cat([match_pred_s2t, match_pred_t2s])
            
            # match_pred, _, _ = CM.get_match(data['conf_matrix_pred'] * s2t_subspace.float(), thr=conf_threshold, mutual=False)
            
            # match_pred, _, _ = CM.get_match(data['conf_matrix_pred']*((s2t_subspace+t2s_subspace)>0).float(), thr=self.cfg.conf_threshold, mutual=False)
            match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=self.cfg.conf_threshold, mutual=False)
            
            corr_num.append(match_pred.shape[0])
            
            ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=0.1).mean()
            
            IR += ir
            FMR += (ir>0.05).float()

            if False:
                match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=0.05, mutual=False)
                s_pcd = data["s_pcd"][0]
                t_pcd = data["t_pcd"][0]
                s_pcd_gt = apply_transform(s_pcd, tsfms)
                
                offset = np.array([0, 0, 0])
                src_corr = s_pcd_gt[match_pred[:, 1]]
                tgt_corr = t_pcd[match_pred[:, 2]]
                
                t_pcd = t_pcd+offset
                
                PSC().add_pcd(s_pcd_gt).add_pcd(t_pcd).add_lines(src_corr, tgt_corr).show()

            if False:
                s_pcd = data["s_pcd"][0]
                t_pcd = data["t_pcd"][0]+torch.Tensor([1,0,0])
                s_pcd_gt = apply_transform(s_pcd, tsfms)
                src_color = torch.zeros_like(s_pcd)
                tgt_color = torch.zeros_like(t_pcd)
                ind = torch.randperm(s_pcd.shape[0])[0]
                src_color[ind] = torch.tensor([1,0,0]).float()
                tgt_color[s2t_subspace[0][ind]] = torch.tensor([1,0,0]).float()
                PSC().add_pcd(s_pcd_gt).add_color(src_color).add_pcd(t_pcd).add_color(tgt_color).show()

            if not self.cfg.evaluate_corr_only:
                rot1, trn1 = MML.ransac_regist_coarse(data['s_pcd'], data['t_pcd'], data['src_mask'], data['tgt_mask'], match_pred)
                rr1 = MML.compute_registration_recall(rot1, trn1, data, thr=0.2) # 0.2m

                tsfm1 = torch.eye(4)
                tsfm1[:3, :3] = rot1
                tsfm1[:3, -1:] = trn1
                if rr1 == 1.0:
                    err1 = se3_compare(tsfm1, tsfms)
                    trans1 += err1["rot_deg"]
                    rot_deg1 += err1["trans"]
                
                tsfm2 = registration_icp(data['src_points'].numpy(), data['tgt_points'].numpy(), tsfm1)
                rot2 = torch.from_numpy(tsfm2[:3, :3]).float()[None, ...]
                trn2 = torch.from_numpy(tsfm2[:3, -1:]).float()[None, ...]
                rr2 = MML.compute_registration_recall(rot2, trn2, data, thr=0.2) # 0.2m
                if rr2 == 1.0:
                    err2 = se3_compare(torch.from_numpy(tsfm2).float(), tsfms)
                    trans2 += err2["rot_deg"]
                    rot_deg2 += err2["trans"]
                    
                bs = len(rot1)
                assert  bs==1
                success1 += rr1
                success2 += rr2

        corr_num = torch.tensor(corr_num)
        print("corr_num.min:", corr_num.min().item())
        print("corr_num.max:", corr_num.max().item())
        print("corr_num.mean:", corr_num.float().mean().item())
        IRate, FMR = IR / val_dir_len, FMR / val_dir_len
        if not self.cfg.evaluate_corr_only:
            recall1 = success1 / val_dir_len
            recall2 = success2 / val_dir_len
            
            rte1 = trans1 / success1
            rre1 = rot_deg1 / success1
            
            rte2 = trans2 / success2
            rre2 = rot_deg2 / success2
            
            print(f"IR: {IRate} FMR: {FMR}")
            print(f"RR: {recall1} RRE: {rre1} RTE: {rte1}")
            print(f"RR(ICP): {recall2} RRE: {rre2} RTE: {rte2}")
        else:
            print(f"IR: {IRate} FMR: {FMR}")
            
                
    def get_matches(self, data):
        

        
        match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=self.cfg.conf_threshold, mutual=False)
        
        # subspace_conf_matrix_pred = data['subspace_conf_matrix_pred']
        
        
        # if data["semantic_masks_valid"].any():
        #     src_semantic_masks_hard = data["src_semantic_masks"] > 0.3
        #     tgt_semantic_masks_hard = data["tgt_semantic_masks"] > 0.3
            
        #     subspace_conf_mat = [subspace_conf_matrix_pred[:, i][:, :, j] for i,j in zip(src_semantic_masks_hard, tgt_semantic_masks_hard)]
        #     subspace_match_preds = [CM.get_match(i, thr=self.cfg.subspace_conf_threshold, mutual=False)[0] for i in subspace_conf_mat]
            
        #     src_ss_ind_map = data["src_sub_ind_padded"]
        #     tgt_ss_ind_map = data["tgt_sub_ind_padded"]
            
        #     subspace_matches = []
        #     for i in range(len(subspace_match_preds)):
        #         subspace_matches.append(
        #             torch.stack([src_ss_ind_map[i][subspace_match_preds[i][:, 1]], 
        #                         tgt_ss_ind_map[i][subspace_match_preds[i][:, 2]]]).T
        #         ) 
        #     subspace_matches = torch.cat(subspace_matches)
        #     subspace_matches = torch.cat([torch.zeros_like(subspace_matches)[:, :1], subspace_matches], dim=1)


        #     match_pred = torch.cat([
        #         match_pred, subspace_matches
        #     ])
        # return MML.compute_inlier_ratio(match_pred, data, inlier_thr=0.1).mean(), match_pred

        #############################################################################################################################
        
        # src_2dfeat = data["src_2dfeat"]
        # tgt_2dfeat = data["tgt_2dfeat"]
        
        # num_clusters = 5
        # kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(src_2dfeat[0].cpu().numpy())
        # src_cluster_ids_x = kmeans.labels_
        # tgt_cluster_ids_x = kmeans.predict(tgt_2dfeat[0].cpu().numpy())

        # src_ind_map = torch.arange(src_cluster_ids_x.shape[0])
        # src_cluster_ind_map = [src_ind_map[src_cluster_ids_x==i] for i in range(num_clusters)]
        # tgt_ind_map = torch.arange(tgt_cluster_ids_x.shape[0])
        # tgt_cluster_ind_map = [tgt_ind_map[tgt_cluster_ids_x==i] for i in range(num_clusters)]
        
        # subspace_conf_mat = [conf_matrix_pred[:, src_cluster_ids_x==i][:, :, tgt_cluster_ids_x==i] for i in range(num_clusters)]
        
        # subspace_match_preds = [CM.get_match(i, thr=self.cfg.conf_threshold, mutual=False)[0] for i in subspace_conf_mat]
        
        # subspace_matches = []
        # for i in range(num_clusters):
        #     subspace_matches.append(
        #         torch.stack([src_cluster_ind_map[i][subspace_match_preds[i][:, 1]], 
        #                     tgt_cluster_ind_map[i][subspace_match_preds[i][:, 2]]]).T
        #     ) 
        # subspace_matches = torch.cat(subspace_matches)
        # subspace_matches = torch.cat([torch.zeros_like(subspace_matches)[:, :1], subspace_matches], dim=1)
        # return MML.compute_inlier_ratio(subspace_matches, data, inlier_thr=0.1).mean(), subspace_matches

        #############################################################################################################################




        # semantic_masks_valid = data["semantic_masks_valid"]
        # if semantic_masks_valid.any():
        #     src_sub_len = (~data["src_sub_ind_padded_mask"]).sum(1)
        #     tgt_sub_len = (~data["tgt_sub_ind_padded_mask"]).sum(1)
            
        #     sub_len = torch.stack([src_sub_len, tgt_sub_len]).T
            
        #     subspace_conf_matrix_pred = data['subspace_conf_matrix_pred']
        #     subspace_conf_matrix_pred = [m[None, :i, :j] for m, (i, j) in zip(subspace_conf_matrix_pred, sub_len)]
            
        #     subspace_matches_list = []
        #     for i, conf_matrix in enumerate(subspace_conf_matrix_pred):
        #         subspace_match_pred, _, _ = CM.get_match(
        #             conf_matrix, 
        #             thr=self.cfg.subspace_conf_threshold, mutual=False
        #         )
        #         subspace_match_pred[:, 0] = i
        #         subspace_matches_list.append(subspace_match_pred)
        #     subspace_match_pred = torch.cat(subspace_matches_list)
            
        #     subspace_matches = subspace_matches_remap(
        #         subspace_match_pred,
        #         data["src_sub_ind_padded"][semantic_masks_valid],
        #         data["tgt_sub_ind_padded"][semantic_masks_valid],
        #     )


                
        # semantic_masks_valid = data["semantic_masks_valid"]
        # if semantic_masks_valid.any():
        #     subspace_match_pred, _, _ = CM.get_match(
        #         data['subspace_conf_matrix_pred'], 
        #         thr=self.cfg.subspace_conf_threshold, mutual=False
        #     )
        #     subspace_matches = subspace_matches_remap(
        #         subspace_match_pred,
        #         data["src_sub_ind_padded"][semantic_masks_valid],
        #         data["tgt_sub_ind_padded"][semantic_masks_valid],
        #     )

        #     match_pred = torch.cat([
        #         match_pred, subspace_matches
        #     ])

        ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=0.1).mean()
        
        
        ##################################### Visualization #####################################
        VIS_SUBSPACE_CORR = False
        if VIS_SUBSPACE_CORR:
            # offset = np.array([3, 0, 0])
            offset = np.array([0, 0, 0])
            s_pcd = data['s_pcd'][0].clone().cpu()
            t_pcd = data['t_pcd'][0].clone().cpu()
            tsfm = data["tsfms"][0].clone().cpu()
            s_pcd_gt = apply_transform(s_pcd, tsfm)
            
            src_sub_ind_padded = data['src_sub_ind_padded'].clone().cpu()
            tgt_sub_ind_padded = data['tgt_sub_ind_padded'].clone().cpu()
            src_corr = src_sub_ind_padded[subspace_match_pred[:, 0], subspace_match_pred[:, 1]]
            tgt_corr = tgt_sub_ind_padded[subspace_match_pred[:, 0], subspace_match_pred[:, 2]]
            
            t_pcd = t_pcd+offset
            
            PSC().add_pcd(s_pcd_gt).add_pcd(t_pcd).add_lines(s_pcd_gt[src_corr], t_pcd[tgt_corr]).show()
            
        VIS_CORR = False
        if VIS_CORR:
            # offset = np.array([3, 0, 0])
            s_pcd = data['s_pcd'][0].clone().cpu()
            t_pcd = data['t_pcd'][0].clone().cpu()
            tsfm = data["tsfms"][0].clone().cpu()
            s_pcd_gt = apply_transform(s_pcd, tsfm)
            
            offset = np.array([0, 0, 0])
            src_corr = s_pcd_gt[match_pred[:, 1]]
            tgt_corr = t_pcd[match_pred[:, 2]]
            
            t_pcd = t_pcd+offset
            
            PSC().add_pcd(s_pcd_gt).add_pcd(t_pcd).add_lines(src_corr, tgt_corr).show()

        ############################################################################################
        return ir, match_pred
    
    
def make_open3d_point_cloud(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def registration_icp(
    src_points, 
    ref_points, 
    init_tf, 
    distance_threshold=0.05):

    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    result = o3d.registration.registration_icp(
        src_pcd, ref_pcd, distance_threshold, init_tf,
        o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(
            relative_fitness = 0.000001, 
            relative_rmse = 0.000001, 
            max_iteration = 400)
    )
    return np.array(result.transformation)

def se3_cat(a, b):
    """Concatenates two SE3 transforms"""
    rot_a, trans_a = a[..., :3, :3], a[..., :3, 3:4]
    rot_b, trans_b = b[..., :3, :3], b[..., :3, 3:4]

    rot = rot_a @ rot_b
    trans = rot_a @ trans_b + trans_a
    dst = se3_init(rot, trans)
    return dst

def se3_inv(pose):
    """Inverts the SE3 transform"""
    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    irot = rot.transpose(-1, -2)
    itrans = -irot @ trans
    return se3_init(irot, itrans)

def se3_init(rot=None, trans=None):

    assert rot is not None or trans is not None

    if rot is not None and trans is not None:
        pose = torch.cat([rot, trans], dim=-1)
    elif rot is None:  # rotation not provided: will set to identity
        pose = F.pad(trans, (3, 0))
        pose[..., 0, 0] = pose[..., 1, 1] = pose[..., 2, 2] = 1.0
    elif trans is None:  # translation not provided: will set to zero
        pose = F.pad(rot, (0, 1))

    return pose

def se3_compare(a, b):
    combined = se3_cat(a, se3_inv(b))

    trace = combined[..., 0, 0] + combined[..., 1, 1] + combined[..., 2, 2]
    rot_err_deg = torch.acos(torch.clamp(0.5 * (trace - 1), -1., 1.)) \
                  * 180 / math.pi
    trans_err = torch.norm(combined[..., :, 3], dim=-1)

    err = {
        'rot_deg': rot_err_deg,
        'trans': trans_err
    }
    return err