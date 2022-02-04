import pickle
import json
from vizualize_results import viz, viz_multiview
from wandb_utils.wandb import WandB
from pytorch3d.io import load_obj
import pdb
import torch
from transforms import NormalizeShift


with open('/home/stephen/Documents/3DDet/output/kitti_models/pointpillar/default/eval/eval_with_train/epoch_2/val/result.pkl', 'rb') as j:
    x = pickle.load(j)
labels = []
centroids = []
dims = []
# id = '000008'
count_limit = 50
count = 0
# pdb.set_trace()
run = WandB(
    project='3DDet_Viz',
    enabled=True,
    log_objects=True,
    name='Pred Test Viz - DL'
)
for scene in x:
    if count < count_limit:
        centroids = scene['location']
        dims = scene['dimensions']
        labels = scene['name']
        id = scene['frame_id']
        #pdb.set_trace()
        for row in centroids:
            row = list(row)
        for row1 in dims:
            row1 = list(row1)
        for row2 in labels:
            row2 = list(row2)
        # print("centroids: ", centroids)
        labels = list(labels)
        # pdb.set_trace()
        centroids = torch.tensor(centroids)
        dims = torch.tensor(dims)
        
        json_mapping = '/multiview/3d-count/obj_detection/json_kitti_mapping.json'
        f = open(json_mapping)
        json_map = json.load(f)
        f.close()
        obj_path = json_map[str(int(id))]['53k_point_cloud_path']
        pc, _, _ = load_obj(obj_path)
        print("pc: ",obj_path)
        print("pc min 0: ", min(pc[:,0]))
        print("pc min 1: ", min(pc[:,1]))
        print("pc min 2: ", min(pc[:,2]))

        print("pc max 0: ", max(pc[:,0]))
        print("pc max 1: ", max(pc[:,1]))
        print("pc max 2: ", max(pc[:,2]))
        print()
        # pdb.set_trace()
        norm = NormalizeShift()
        pc = norm.fit_transform(pc)
        centroids = norm(centroids)
        dims = norm.scale(dims)

        boxes = run.get_bb_dict(centroids, dims, labels=labels)
        log_dict = run.get_point_cloud_log(pc, boxes=boxes)
        run.log(log_dict)
        #print("id: ", id)
        #print("centroids: ", centroids)
        count = count + 1
run.finish()

