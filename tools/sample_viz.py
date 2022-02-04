import json
from wandb_utils.wandb import WandB
from pytorch3d.io import load_obj
import torch
import numpy as np
from transforms import NormalizeShift


centroids = []
dims = []
id = '000000'
run = WandB(
    project='3DDet_Viz',
    enabled=True,
    log_objects=True,
    name='Pred Test Viz - DL'
)
## x = front/back
## y = left/right
## z = up/down
centroids = [[4, 1, 0]]
dims = [[.14,.14,.37], [.17, .17, .56], [.57, .50, .28], [.21, .2, .54], [.27, .35, .42]]
label = ["16ozcan", "2Oozbottle", "twelvepack", "twoliter", "sixpack"]
centroids = torch.tensor(centroids)
dims = torch.tensor(dims)
json_mapping = '/multiview/3d-count/obj_detection/json_kitti_mapping.json'
f = open(json_mapping)
json_map = json.load(f)
f.close()
obj_path = json_map[str(int(id))]['53k_point_cloud_path']
pc, _, _ = load_obj(obj_path)
print(min(pc[:,0]))
print(max(pc[:,0]))
print(min(pc[:,1]))
print(max(pc[:,1]))
print(min(pc[:,2]))
print(max(pc[:,2]))

# norm = NormalizeShift()
# pc = norm.fit_transform(pc)
# centroids = norm(centroids)
# dims = norm.scale(dims)
boxes = run.get_bb_dict(centroids, dims)
log_dict = run.get_point_cloud_log(pc, boxes=boxes)
run.log(log_dict)
#print("id: ", id)
#print("centroids: ", centroids)
run.finish()
