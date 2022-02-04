import pdb
import torch
from wandb_utils.wandb import WandB
import json
from pytorch3d.io import load_obj
import numpy as np


def viz(centroids, dims, labels, id):
    # pdb.set_trace()
    centroids = torch.tensor(centroids)
    dims = torch.tensor(dims)
    json_mapping = '/multiview/3d-count/obj_detection/json_kitti_mapping.json'
    f = open(json_mapping)
    json_map = json.load(f)
    f.close()
    obj_path = json_map[str(int(id))]['53k_point_cloud_path']
    #pdb.set_trace()
    pc, _, _ = load_obj(obj_path)

    run = WandB(
        project='3DDet_Viz',
        enabled=True,
        log_objects=True,
        name='Pred Test Viz - DL: ' + id
    )

    # pdb.set_trace()
    # norm = NormalizeShift()
    # pc = norm.fit_transform(pc)
    # centroids = norm(centroids)
    # dims = norm.scale(dims)
    boxes = run.get_bb_dict(centroids, dims, labels=labels)
    log_dict = run.get_point_cloud_log(pc, boxes=boxes)
    run.log(log_dict)
    run.finish()

def viz_multiview(id, run1):
    import json
    from pytorch3d.io import load_obj
    from transforms import NormalizeShift
    json_mapping = '/multiview/3d-count/obj_detection/json_kitti_mapping.json'
    f = open(json_mapping)
    json_map = json.load(f)
    f.close()
    #pdb.set_trace()
    label_path = json_map[str(int(id))]['53k_label_path']

    k = open(label_path)
    data = json.load(k)
    # Closing file
    k.close()
    box_list = []
    labels = []
    for key in data['bounding_boxes'].keys():
        # for row in range(len(data['bounding_boxes'][key])):
        #     data['bounding_boxes'][key][row] = tuple(data['bounding_boxes'][key][row])
        box_list.append(data['bounding_boxes'][key])
        labels.append(key)
    #print("box_list in MV: ", box_list)
    boxes = run1.get_bb_dict_from_bb(corners=box_list, labels=labels)
    obj_path = json_map[str(int(id))]['53k_point_cloud_path']
    pc, _, _ = load_obj(obj_path)
    # norm = NormalizeShift()
    # pc = norm.fit_transform(pc)
    # pc = pc.data.numpy()
    #print("Raw boxes: ", boxes)
    pc_dict = run1.get_point_cloud_log(pc, boxes=boxes)
    #pdb.set_trace()
    run1.log(pc_dict)
    return
