"""
soure code: https://github.com/poodarchu/Det3D/blob/master/det3d/ops/point_cloud/point_cloud_ops.py
"""

import numpy as np
import torch
import numba

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


class PointPillarsFeatures(object):

    def __init__(
            self,
            max_points=35,
            pc_range=(-1, -1, -1, 1, 1, 1),
            max_voxels=20000,
            voxel_size=(0.05, 0.05, 2.0),
            with_distance=False,
            zero_pad=True
    ):
        """
            :param voxel_size: List(<float>: 3). Size of voxels, only utilize x and y size.
        """
        self.max_points = max_points
        self.pc_range = pc_range
        #x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        #voxel_dim = (((x_max - x_min) * (y_max - y_min) * (z_max - z_min)) / max_voxels) ** (1 / 3)
        #voxel_size = (voxel_dim, voxel_dim, voxel_dim)
        self.voxel_size = voxel_size
        self.max_voxels = max_voxels
        self.max_points = max_points

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.with_distance = with_distance
        self.zero_pad = zero_pad


    @staticmethod
    def points_to_voxel(points,
                        voxel_size,
                        coors_range,
                        max_points=35,
                        reverse_index=True,
                        max_voxels=20000):
        """convert kitti points(N, >=3) to voxels. This version calculate
        everything in one loop. now it takes only 4.2ms(complete point cloud)
        with jit and 3.2ghz cpu.(don't calculate other features)
        Note: this function in ubuntu seems faster than windows 10.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points and
                points[:, 3:] contain other information such as reflectivity.
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
            coors_range: [6] list/tuple or array, float. indicate voxel range.
                format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel.
            reverse_index: boolean. indicate whether return reversed coordinates.
                if points has xyz format and reverse_index is True, output
                coordinates will be zyx format, but points in features always
                xyz format.
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. you should shuffle points
                before call this function because max_voxels may drop some points.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
        """
        if not isinstance(voxel_size, np.ndarray):
            voxel_size = np.array(voxel_size, dtype=points.dtype)
        if not isinstance(coors_range, np.ndarray):
            coors_range = np.array(coors_range, dtype=points.dtype)
        voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
        # don't create large array in jit(nopython=True) code.
        num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
        coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
        voxels = np.zeros(
            shape=(max_voxels, max_points, points.shape[-1]), dtype=np.float32)
        coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
        voxel_num = _points_to_voxel_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels
        )

        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]
        return voxels, coors, num_points_per_voxel

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        """Create boolean mask by actually number of a padded tensor.
        Args:
            actual_num ([type]): [description]
            max_num ([type]): [description]
        Returns:
            [type]: [description]
        """

        actual_num = torch.unsqueeze(actual_num, axis + 1)
        # tiled_actual_num: [N, M, 1]
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(
            max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]
        return paddings_indicator


    def __call__(self, x, *args, **kwargs):

        if isinstance(x, torch.Tensor):
            x = x.data.numpy()

        features, coors, num_voxels = self.points_to_voxel(
            points=x,
            voxel_size=np.array(list(self.voxel_size)),
            coors_range=np.array(list(self.pc_range)),
            max_points=self.max_points,
            max_voxels=int(self.max_voxels)
        )

        # Find distance of x, y, and z from cluster center

        features = torch.from_numpy(features)
        num_voxels = torch.from_numpy(num_voxels)
        coors = torch.from_numpy(coors)

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 0].float().unsqueeze(1) * self.vx + self.x_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        if self.zero_pad:
            features = self.zero_pad_voxels(features)
            coors = self.zero_pad_voxels(coors)
            num_voxels = self.zero_pad_voxels(num_voxels)

        # logging metrics for voxels
        """sums = features.sum(-1)

        n_dense = len(torch.where(sums != 0.0)[0])
        point_density = n_dense / (sums.shape[0]*sums.shape[1])
        print(f"point-wise density: {point_density}")"""

        return features, coors, num_voxels

    def zero_pad_voxels(self, tensor):
        """
        Zero pad feature tensor to have max_voxels
        @return: torch.Tensor (self.max_voxels, N, D)
        """

        if tensor.ndim == 3:
            P, N, D = tensor.shape
            padded = torch.zeros(self.max_voxels, N, D)
            padded[:P, :, :] = tensor
        elif tensor.ndim == 2:
            P, D = tensor.shape
            padded = torch.zeros(self.max_voxels, D)
            padded[:P, :] = tensor
        elif tensor.ndim == 1:
            P = tensor.shape[0]
            padded = torch.zeros(self.max_voxels)
            padded[:P] = tensor
        else:
            raise NotImplementedError(f"ndim not supported: {tensor.ndim}")

        #print(f"voxel density: {P / self.max_voxels}" )

        return padded

    @classmethod
    def build(cls, cfg, zero_pad=True):
        pc_features = PointPillarsFeatures(
            max_voxels=cfg["max_voxels"],
            voxel_size=cfg["voxel_size"],
            max_points=cfg["max_points"],
            with_distance=cfg["with_distance"],
            zero_pad=zero_pad
        )
        return pc_features