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


logging.getLogger('PIL').setLevel(logging.WARNING)


class IndoorDataset2D3D(Dataset):

    def __init__(self, infos, base_dir, rgbd_dir, data_augmentation=True, use_precompute_2dfeat=True):
        super().__init__()

        self.infos = infos

        self.base_dir = base_dir
        self.rgbd_dir = rgbd_dir
        # self.feat_dir = os.path.join(base_dir, "dinov2_features")
        # self.feat_dir = os.path.join(base_dir, "dinov2_features")
        self.feat_dir = os.path.join(base_dir, "semantic_masks_merge")
        self.mask_dir = os.path.join(base_dir, "semantic_masks")
        
        self.data_augmentation = data_augmentation
        self.use_precompute_2dfeat = use_precompute_2dfeat

        self.depth_image_transforms_feat = transforms.Compose([
            transforms.Resize((36, 50), Image.NEAREST),
            transforms.ToTensor(),
        ])
        
        self.depth_image_transforms_mask = transforms.Compose([
            transforms.Resize((128, 176), Image.NEAREST),
            transforms.ToTensor(),
        ])
        
        
        self.depth_image_fs_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # self.infos_2d = load_obj(os.path.join(self.base_dir, 'info_2d.pkl'))
        # self.infos_2d = load_obj(os.path.join(self.base_dir, 'info_2d_with_mask.pkl'))
        self.infos_2d = load_obj(os.path.join(self.base_dir, 'info_2d_with_mask_pack.pkl'))
        
        self.scene_poses = self.gether_scene_pose()
        
        self.mlm = MaskLabelMapping()

    def __len__(self):
        return len(self.infos)

    def gether_scene_pose(self):
        
        scene_poses = {}
        pcd_list = list(set(self.infos["src"]+self.infos["tgt"]))
        
        for item in pcd_list:
            
            pcd_path = os.path.join(self.base_dir, item)
            pcd_info_path = pcd_path.replace(".pth", ".info.txt")
            with open(pcd_info_path, 'r') as f:
                _ = f.readline()
                scene_pose = np.loadtxt(StringIO(f.read()))
        
            scene_poses[item] = scene_pose
        return scene_poses
    
    def get_2d_info_path(self, info_path):
        return os.path.join(self.rgbd_dir, info_path)

    def _get_images(self, infos_2d, pcd, scene_pose, world2camera):
        intrinsics = np.loadtxt(self.get_2d_info_path(infos_2d["intrinsics"]))
        fullsize_projection = Projection(intrinsics)
        
        big_size, image_size = [640, 480], [50, 36]
        feat_intrinsics = adjust_intrinsic(intrinsics, big_size, image_size)
        feat_projection = Projection(feat_intrinsics, thresh=2.5)

        big_size, image_size = [640, 480], [176, 128]
        mask_intrinsics = adjust_intrinsic(intrinsics, big_size, image_size)
        mask_projection = Projection(mask_intrinsics)

        
        data_2d3d = []
        # masks_vis = []
        frame = np.load(os.path.join(self.feat_dir, infos_2d["dino_info"]))
        frame_feat = {i: frame[i] for i in frame}
        
        for idx, image_info in enumerate(infos_2d["image_info"]):
        
            pose_path = self.get_2d_info_path(image_info["pose_path"])
            color_path = self.get_2d_info_path(image_info["color_path"])
            depth_path = self.get_2d_info_path(image_info["depth_path"])
            data = dict(color_path=color_path)
            
            pose = np.loadtxt(pose_path)
            pose_rev = np.linalg.inv(pose)
            world2camera_current = torch.mm(torch.from_numpy(pose_rev).float(), 
                                    torch.mm(torch.from_numpy(scene_pose).float(), world2camera))

            
            # init_pcd = torch.from_numpy(pcd).float()
            
            depth_image = Image.open(depth_path)  # 640 480
            depth_mask_image = self.depth_image_transforms_mask(depth_image) / 1000.0
            
            # mask_2d = np.load(os.path.join(self.mask_dir, image_info["mask_path"]))
            data["color_path"] = color_path


            # frame_feat = np.load(os.path.join(self.feat_dir, image_info["mask_path"]))
            # mask, mask_label, feat_local, feat_global  = (frame_feat[i] for i in frame_feat)
            
            mask, mask_label, feat_local, feat_global = \
                frame_feat[f'mask_{idx}'], frame_feat[f'mask_label_{idx}'], \
                frame_feat[f'feat_local_{idx}'], frame_feat[f'feat_global_{idx}']
            
            data["feat_local"] = feat_local
            data["feat_global"] = feat_global
            # mask, mask_label = mask_2d["mask"], mask_2d["label"]
            # mask = resize(input=mask, size=depth_fs_image.shape[1:], mode="bilinear", align_corners=False)
            
            # data["mask_2d"] = torch.from_numpy(mask_2d).long()
            # inds2d_fs, inds3d_fs = fullsize_projection.projection(
            #     init_pcd, depth_fs_image, world2camera_current)
            

            mask_label_map = self.mlm(mask_label)
            n, h, w = mask.shape
            masks = np.zeros((len(self.mlm), h, w))
            mask_label_unique = np.unique(mask_label_map)
            masks[mask_label_unique] = np.stack([mask[mask_label_map==i].sum(0).clip(0, 255).astype(np.uint8) for i in mask_label_unique])
            
            data["mask"] = masks
            data["mask_label"] = mask_label_map
            
            data["mask_projection"] = mask_projection
            data["depth_mask_image"] = depth_mask_image
            data["world2camera"] = world2camera_current
            
            depth_feat_image = self.depth_image_transforms_feat(depth_image) / 1000.0
            data["feat_projection"] = feat_projection
            data["depth_feat_image"] = depth_feat_image

            
            # mask_arg_max = mask.argmax(0)
            # mask_vis = np.zeros((*mask_arg_max.shape, 3), dtype=np.uint8)
            # for ind, label in enumerate(mask_label):
            #     mask_vis[mask_arg_max==ind] = np.array(ADE20K_COLORMAP[label+1])
            # masks_vis.append(mask_vis)
            
            # data["inds2d_fullsize"] = inds2d_fs.long()
            # data["inds3d_fullsize"] = inds3d_fs.long()
            
            # inds2d, inds3d = projection.projection(
            #     init_pcd, depth_image, world2camera_current)
            # data["inds2d"] = inds2d.long()
            # data["inds3d"] = inds3d.long()
            
            # if self.use_precompute_2dfeat:
                # color_feat_path = os.path.join(self.feat_dir, image_info["color_feat_path"])
            #     data["feature_2d"] = torch.load(color_feat_path)
            # else:
            #     data["color_image"] = Image.open(color_path)
                
            # if DEBUG_DEPTHMAP:
            #     data["depth_pcd"] = unproject(
            #         (depth_fs_image[0]*1000.0).detach().cpu().numpy(), 
            #         fullsize_projection.intrinsics, 
            #         world2camera_current.inverse())
                
            data_2d3d.append(data)

        # img = Image.fromarray(np.concatenate(masks_vis, axis=1))
        # return data_2d3d, img
        return data_2d3d
            
        
    def _get_image_pair(self, item, src_pcd, tgt_pcd, aug_src=None, rot_ab=None):

        # src_path = self.infos[item]['pcd0']
        # tgt_path = self.infos[item]['pcd1']
        
        src_path = self.infos["src"][item]
        tgt_path = self.infos["tgt"][item]
        
        src_infos_2d = self.infos_2d[src_path]
        tgt_infos_2d = self.infos_2d[tgt_path]
        src_scene_pose = self.scene_poses[src_path]
        tgt_scene_pose = self.scene_poses[tgt_path]
        
        if self.data_augmentation:

            if aug_src:
                # print("augment rotation to source point cloud")
                src_world2camera = np.eye(4)
                src_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                src_world2camera = torch.from_numpy(
                    src_world2camera).float()

                tgt_world2camera = torch.eye(4).float()
            else:
                # print("augment rotation to target point cloud")
                tgt_world2camera = np.eye(4)
                tgt_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                tgt_world2camera = torch.from_numpy(
                    tgt_world2camera).float()
                src_world2camera = torch.eye(4).float()
        else:
            src_world2camera = torch.eye(4).float()
            tgt_world2camera = torch.eye(4).float()

        # start_time = time.time()
        # src_2d3d, src_img = self._get_images(src_infos_2d, src_pcd, src_scene_pose, src_world2camera)
        # tgt_2d3d, tgt_img = self._get_images(tgt_infos_2d, tgt_pcd, tgt_scene_pose, tgt_world2camera)
        
        # src_img.save("a.png")
        # tgt_img.save("b.png")

        src_2d3d = self._get_images(src_infos_2d, src_pcd, src_scene_pose, src_world2camera)
        tgt_2d3d = self._get_images(tgt_infos_2d, tgt_pcd, tgt_scene_pose, tgt_world2camera)
        
        src_mask_label = np.unique(np.concatenate([i["mask_label"] for i in src_2d3d]))
        tgt_mask_label = np.unique(np.concatenate([i["mask_label"] for i in tgt_2d3d]))
        
        mutual_mask_labels = src_mask_label[((src_mask_label[:, None] - tgt_mask_label[None, :])==0).nonzero()[0]]
        
        all_mask_labels = np.unique(np.concatenate([src_mask_label, tgt_mask_label]))
        # print("elapse: {}".format(time.time() - start_time))
        
        return src_2d3d, tgt_2d3d, mutual_mask_labels, all_mask_labels


