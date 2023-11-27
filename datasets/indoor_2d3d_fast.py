from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import random
from io import StringIO
from lib.utils import load_obj
from datasets.projection import Projection, adjust_intrinsic, unproject
import logging
from pointscope import PointScopeClient as PSC
from lib.colormaps import ADE20K_COLORMAP
from lib.utils import resize

from datasets.same_semantic_mapping import MaskLabelMapping
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logging.getLogger('PIL').setLevel(logging.WARNING)


class IndoorDataset2D3D(Dataset):

    def __init__(self, infos, base_dir, rgbd_dir, data_augmentation=True, use_precompute_2dfeat=True):
        super().__init__()

        self.infos = infos

        self.base_dir = base_dir
        self.rgbd_dir = rgbd_dir
        self.feat_dir = os.path.join(base_dir, "pcd_2dinfo_pack")
        
        self.data_augmentation = data_augmentation
        self.use_precompute_2dfeat = use_precompute_2dfeat

        self.infos_2d = load_obj(os.path.join(self.base_dir, 'info_2d_pack.pkl'))
        
    def __len__(self):
        return len(self.infos)

    def get_2d_info_path(self, info_path):
        return os.path.join(self.rgbd_dir, info_path)
    
    def get_pcd_semantic_info(self, pcd, info_2d, rot=None):
        pcd_2dinfo = torch.load(os.path.join(self.feat_dir, info_2d["semantic_info"]))
        pcd = pcd_2dinfo["pcd"]
        pcd_2dfeat = pcd_2dinfo["pcd_2dfeat"]
        pcd_mask = pcd_2dinfo["pcd_mask"]
        pcd_mask /= pcd_mask.max()
        return pcd, pcd_2dfeat, pcd_mask

    def _get_image_pair(self, item, src_pcd, tgt_pcd, aug_src=None, rot_ab=None):

        src_path = self.infos["src"][item]
        tgt_path = self.infos["tgt"][item]
        
        src_infos_2d = self.infos_2d[src_path]
        tgt_infos_2d = self.infos_2d[tgt_path]

        src_mapping_pcd, src_pcd_2dfeat, src_pcd_mask = self.get_pcd_semantic_info(
            src_pcd, src_infos_2d)
        tgt_mapping_pcd, tgt_pcd_2dfeat, tgt_pcd_mask = self.get_pcd_semantic_info(
            tgt_pcd, tgt_infos_2d)

        if self.data_augmentation:
            if aug_src:
                src_mapping_pcd = np.matmul(rot_ab, src_mapping_pcd.T).T
            else:
                tgt_mapping_pcd = np.matmul(rot_ab, tgt_mapping_pcd.T).T

        # src_mask_label = src_pcd_mask.sum(1).nonzero()[0]
        # tgt_mask_label = tgt_pcd_mask.sum(1).nonzero()[0]
        
        # mutual_mask_labels = src_mask_label[((src_mask_label[:, None] - tgt_mask_label[None, :])==0).nonzero()[0]]
        # all_mask_labels = np.unique(np.concatenate([src_mask_label, tgt_mask_label]))
        
        mutual_mask_labels, all_mask_labels = None, None
        
        return (src_mapping_pcd.float(), src_pcd_2dfeat, src_pcd_mask), \
            (tgt_mapping_pcd.float(), tgt_pcd_2dfeat, tgt_pcd_mask), \
            mutual_mask_labels, all_mask_labels


    def get_pcd_semantic_info(self, pcd, info_2d, rot=None):
        pcd_2dinfo = torch.load(os.path.join(self.feat_dir, info_2d["semantic_info"]))
        pcd = pcd_2dinfo["pcd"]
        pcd_2dfeat = pcd_2dinfo["pcd_2dfeat"]
        return pcd, pcd_2dfeat
    
    def _get_image_pair(self, item, src_pcd, tgt_pcd, aug_src=None, rot_ab=None):

        src_path = self.infos["src"][item]
        tgt_path = self.infos["tgt"][item]
        
        src_infos_2d = self.infos_2d[src_path]
        tgt_infos_2d = self.infos_2d[tgt_path]

        src_mapping_pcd, src_pcd_2dfeat = self.get_pcd_semantic_info(
            src_pcd, src_infos_2d)
        tgt_mapping_pcd, tgt_pcd_2dfeat = self.get_pcd_semantic_info(
            tgt_pcd, tgt_infos_2d)

        if self.data_augmentation:
            if aug_src:
                src_mapping_pcd = np.matmul(rot_ab, src_mapping_pcd.T).T
            else:
                tgt_mapping_pcd = np.matmul(rot_ab, tgt_mapping_pcd.T).T

        return (src_mapping_pcd.float(), src_pcd_2dfeat), (tgt_mapping_pcd.float(), tgt_pcd_2dfeat)

