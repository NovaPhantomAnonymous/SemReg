import os
import torch
import numpy as np


def unproject(depth_img, depth_intrinsic, pose):
    " batchsize 240 320-->240 320"
    if isinstance(depth_img, torch.Tensor):
        depth_img = depth_img.squeeze(dim=0).numpy()
    if isinstance(depth_intrinsic, torch.Tensor):
        depth_intrinsic = depth_intrinsic.numpy()

    if depth_intrinsic.shape[0] == 3:
        res = np.eye(4)
        res[:3, :3] = depth_intrinsic
        depth_intrinsic = res
            
    depth_shift = 1000.0
    x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
                       np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x
    uv_depth[:, :, 1] = y
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]
    point_list = []
    n = uv_depth.shape[0]
    points = np.ones((n, 4))
    X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 0] = X
    points[:, 1] = Y
    points[:, 2] = uv_depth[:, 2]
    inds = points[:, 2] > 0
    points = points[inds, :]
    points_world = np.dot(points, np.transpose(pose))
    return points_world[:, :3]


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):

    if intrinsic_image_dim == image_dim:
        return intrinsic

    intrinsic_return = np.copy(intrinsic)

    height_after = image_dim[1]
    height_before = intrinsic_image_dim[1]
    height_ratio = height_after / height_before

    width_after = image_dim[0]
    width_before = intrinsic_image_dim[0]
    width_ratio = width_after / width_before

    if width_ratio >= height_ratio:
        resize_height = height_after
        resize_width = height_ratio * width_before

    else:
        resize_width = width_after
        resize_height = width_ratio * height_before

    intrinsic_return[0,0] *= float(resize_width)/float(width_before)
    intrinsic_return[1,1] *= float(resize_height)/float(height_before)
    # account for cropping/padding here
    intrinsic_return[0,2] *= float(resize_width-1)/float(width_before-1)
    intrinsic_return[1,2] *= float(resize_height-1)/float(height_before-1)

    return intrinsic_return


class Projection(object):
    def __init__(self, intrinsic_matrix=0, thresh=0.1):
        """
        intrinsic_matrix is a 4x4 matrix, torch.FloatTensor

        """

        if isinstance(intrinsic_matrix, np.ndarray):
            self.intrinsics = torch.from_numpy(intrinsic_matrix).float()
        else:
            self.intrinsics = intrinsic_matrix
            
        if self.intrinsics.shape[0] == 3:
            res = torch.eye(4)
            res[:3, :3] = self.intrinsics
            self.intrinsics = res
        
        self.thresh = thresh

    @staticmethod
    def matrix_multiplication(matrix, points):
        """
        matrix: 4x4, torch.FloatTensor
        points: nx3, torch.FloatTensor
        reutrn: nx3, torch.FloatTensor
        """
        device = torch.device("cuda" if matrix.get_device() != -1 else "cpu")
        points = torch.cat([points.t(), torch.ones((1, points.shape[0]), device=device)])
        if matrix.shape[0] ==3:
            mat=torch.eye(4).to(device)
            mat[:3,:3]=matrix
            matrix=mat

        return torch.mm(matrix, points).t()[:, :3]



    def projection(self, points, depth_map, world2camera):
        """
        points: nx3 point cloud xyz in world space, torch.FloatTensor
        depth_map: height x width, torch.FloatTensor
        world2camera: 4x4 matrix, torch.FloatTensor

        return: mapping of 2d pixel coordinates to 3d point cloud indices
            inds2d: n x 2 array, torch.LongTensor (notice the order xy == width, height)
            inds3d: n x 1 array, index of point cloud
        """
        depth_map=depth_map.squeeze(0)
        height = depth_map.size(0)
        width = depth_map.size(1)

        xyz_in_camera_space = Projection.matrix_multiplication(world2camera, points)
        xyz_in_image_space = Projection.matrix_multiplication(self.intrinsics, xyz_in_camera_space)

        projected_depth = xyz_in_image_space[:,2]
        xy_in_image_space = (xyz_in_image_space[:,:2] / projected_depth.repeat(2,1).T[:,:]).long()

        mask_height = (xy_in_image_space[:,1] >= 0) & (xy_in_image_space[:,1] < height)
        mask_width = (xy_in_image_space[:,0] >= 0) & (xy_in_image_space[:,0] < width)
        mask_spatial = mask_height & mask_width
        depth = depth_map[xy_in_image_space[mask_spatial,1], 
                          xy_in_image_space[mask_spatial,0]]
        mask_depth = torch.abs(projected_depth[mask_spatial] - depth) < self.thresh

        inds2d = xy_in_image_space[mask_spatial][mask_depth]
        inds3d = torch.arange(points.size(0))[mask_spatial][mask_depth]

        return inds2d, inds3d
    def get_mask(self, img_fov_points, img_foc_depth_pcd,depth_map):
        """
        points: nx3 point cloud xyz in world space, torch.FloatTensor
        depth_map: height x width, torch.FloatTensor
        world2camera: 4x4 matrix, torch.FloatTensor

        return: mapping of 2d pixel coordinates to 3d point cloud indices
            inds2d: n x 2 array, torch.LongTensor (notice the order xy == width, height)
            inds3d: n x 1 array, index of point cloud
        """
        depth_map=depth_map.squeeze(0)
        height = depth_map.size(0)
        width = depth_map.size(1)

        xyz_in_camera_space = img_fov_points
        xyz_in_image_space = Projection.matrix_multiplication(self.intrinsics, xyz_in_camera_space)

        projected_depth = xyz_in_image_space[:,2]
        xy_in_image_space = (xyz_in_image_space[:,:2] / projected_depth.repeat(2,1).T[:,:]).long()

        mask_height = (xy_in_image_space[:,1] >= 0) & (xy_in_image_space[:,1] < height)
        mask_width = (xy_in_image_space[:,0] >= 0) & (xy_in_image_space[:,0] < width)
        mask_spatial = mask_height & mask_width
        depth = depth_map[xy_in_image_space[mask_spatial,1],
                          xy_in_image_space[mask_spatial,0]]
        mask_depth = torch.abs(projected_depth[mask_spatial] - depth) < self.thresh

        inds2d = xy_in_image_space[mask_spatial][mask_depth]
        inds3d = torch.arange(img_fov_points.size(0))[mask_spatial][mask_depth]

        return inds2d, inds3d

if __name__ == "__main__":
    intrinsic_matrix = torch.zeros((4,4))
    depth_map = torch.zeros((240, 320))
    world2camera = torch.zeros((4,4))

    projection = Projection(intrinsic_matrix)
    inds2d, inds3d = projection.projection(points, depth_map, world2camera)






