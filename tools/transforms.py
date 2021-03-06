import pdb

import torch
from pytorch3d.transforms import RotateAxisAngle

class MeanShift(object):
    def __call__(self, pc, *args, **kwargs):
        mean = torch.mean(pc, dim=0)
        output = pc - mean
        return output


class NormalizeShift(object):
    def __init__(self):
        self.means = None
        self.max_dist = None
        # print("set means: ", self.means)

    def fit_transform(self, pc):

        self.means = torch.mean(pc, dim=0)
        translated = pc - self.means
        

        origin = torch.zeros(3)
        dist = torch.norm(translated - origin, p=2, dim=1)
        self.max_dist = torch.max(dist)
        output = translated / self.max_dist
        return output

    def __call__(self, pc):
        # print("set means: ", self.means)
        translated = pc - self.means
        output = translated / self.max_dist
        return output

    def scale(self, pc):
        output = pc / self.max_dist
        return output


class YForward2NegZForward(object):
    """ Transform from (Y forward, Z up) to (Z forward, -Y up)
        See visualization in docs:
            https://delicious-ai.atlassian.net/wiki/spaces/ML/pages/1241055233/Coordinate+systems
    """

    def __call__(self, verts, invert=True, *args, **kwargs):
        """

        @param verts: (torch.Tensor) point cloud (n x 3)
        @param invert: (bool) invert last dimensions (multiply by negative one)
        @param args:
        @param kwargs:
        @return:
        """
        x, y, z = verts.unbind(-1)
        if invert:
            swapped = [x.unsqueeze(-1), z.unsqueeze(-1), -y.unsqueeze(-1)]
        else:
            swapped = [x.unsqueeze(-1), z.unsqueeze(-1), y.unsqueeze(-1)]
        return torch.cat(swapped, -1)


class NegZForward2YForward(object):
    """Inverse of YForward2NegZForward"""

    def __call__(self, verts, invert=True, *args, **kwargs):
        x, y, z = verts.unbind(-1)
        if invert:
            swapped = [x.unsqueeze(-1), -z.unsqueeze(-1), y.unsqueeze(-1)]
        else:
            swapped = [x.unsqueeze(-1), z.unsqueeze(-1), y.unsqueeze(-1)]

        return torch.cat(swapped, -1)

class NegZForward2XForward(object):
    """Inverse of YForward2NegZForward"""

    def __call__(self, verts, invert=True, *args, **kwargs):
        x, y, z = verts.unbind(-1)
        if invert:
            swapped = [x.unsqueeze(-1), -z.unsqueeze(-1), y.unsqueeze(-1)]
        else:
            swapped = [x.unsqueeze(-1), z.unsqueeze(-1), y.unsqueeze(-1)]

        return torch.cat(swapped, -1)

class MaxNormalizer(object):
    def __init__(self, max):
        self.max = max

    def __call__(self, x, *args, **kwargs):
        return x / self.max

class MaxDeNormalizer(object):
    def __init__(self, max):
        self.max = max
    def __call__(self, x, *args, **kwargs):
        return x * self.max


class RandomRotationZ(object):
    """ Rotate points around the Z axis """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.angle = None


    def fit_transform(self, pc):
        self.angle = float(torch.normal(self.mean, self.std, (1, )))
        rot = RotateAxisAngle(angle=self.angle,
                              axis="Y",  # Swap axes operator swaps Z & Y axes in simulated data and
                                         # for real data Z & Y axes are already swapped
                              degrees=True)
        return rot.transform_points(pc)


    def __call__(self, pc):
        rot = RotateAxisAngle(angle=self.angle,
                              axis="Y",  # Swap axes operator swaps Z & Y axes in simulated data and
                                         # for real data Z & Y axes are already swapped
                              degrees=True)
        return rot.transform_points(pc)


class AddNoise(object):
    """ Adds Noise to the point cloud """

    def __init__(self, mean, std, noise_percentage):
        self.mean = mean
        self.std = std
        self.noise_percentage = noise_percentage


    def __call__(self, pc):
        noise = torch.normal(self.mean, self.std, size=pc.shape)
        mask = torch.FloatTensor(pc.shape).uniform_() < self.noise_percentage
        return pc + (noise*mask)


class MirrorX(object):
    """ Mirrors pointclouds about the X axis """

    def __init__(self, probability):
        self.mirror = None
        self.probability = probability

    def fit_transform(self, pc):
        self.mirror = int(float(torch.rand(1)) < self.probability)
        if self.mirror:
            pc[:, 0] = -pc[:, 0]
        return pc


    def __call__(self, pc, *args, **kwargs):
        if self.mirror:
            pc[:, 0] = -pc[:, 0]
        return pc
