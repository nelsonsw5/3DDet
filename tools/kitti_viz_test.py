import pickle
import json
from vizualize_results import viz, viz_multiview
from wandb_utils.wandb import WandB
from pytorch3d.io import load_obj
import pdb
import torch
from transforms import NormalizeShift, YForward2NegZForward, NegZForward2YForward
from bound_box import ThreeDimBoundBox
import numpy as np




run = WandB(
    project='3DDet_Viz',
    enabled=True,
    log_objects=True,
    name='GT'
)
with open('/multiview/Kitti/training/label_2/000000.txt') as k:
    label = k.readlines()
loc = []
dims = []
labels = []

# bbobj = ThreeDimBoundBox(label=label[0])
# boxes = bbobj.kittiBox3D()

for i in label:
    centroid = []
    dimensions = []
    # print(i.split())
    dimensions.append(float(i.split()[8]))
    dimensions.append(float(i.split()[9]))
    dimensions.append(float(i.split()[10]))
    dims.append(dimensions)
    centroid.append(float(i.split()[11]))
    centroid.append(float(i.split()[12]))
    centroid.append(float(i.split()[13]))
    loc.append(centroid)
dims = np.array(dims)
loc = np.array(loc)

lidar_file = "/multiview/Kitti/training/velodyne/000000.bin"
pc_hat = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
# pc = np.empty((pc_hat.shape))
# pc[:,0] = (pc_hat[:,1] * -1)
# pc[:,1] = (pc_hat[:,2] * -1)
# pc[:,2] = pc_hat[:,0]
# pc[:,3] = pc_hat[:,3]
pc = pc_hat
# boxes = run.get_bb_dict(loc, dims, labels=labels)
print(boxes)
log_dict = run.get_point_cloud_log(pc, boxes=boxes)
run.log(log_dict)
run.finish()