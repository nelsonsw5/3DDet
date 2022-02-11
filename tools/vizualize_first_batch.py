import pdb


def viz(train_set, id, run):
    import torch
    from transforms import NormalizeShift
    from wandb_utils.wandb import WandB
    import json
    centroids = []
    dims = []
    labels = []
    import pdb
    box_list = []
    for label in train_set.get_label(id):
        labels.append(label.to_kitti_format().split()[0])
        label_str = label.to_kitti_format()
        parsed_label = label_str.split()
        dims.append([float(parsed_label[8]), float(parsed_label[9]), float(parsed_label[10])])
        centroids.append([float(parsed_label[11]), float(parsed_label[12]), float(parsed_label[13])])
    centroids = torch.tensor(centroids)
    dims = torch.tensor(dims)
    pc = torch.Tensor(train_set.get_lidar(id)[:, 0:3])
    boxes = run.get_bb_dict(centroids, dims, labels=labels)
    log_dict = run.get_point_cloud_log(pc, boxes=boxes)
    run.log(log_dict)

def viz_multiview(id, run1):
    import json
    from pytorch3d.io import load_obj
    from transforms import NormalizeShift
    json_mapping = '/multiview/3d-count/obj_detection/scripts/json_kitti_mapping.json'
    f = open(json_mapping)
    json_map = json.load(f)
    f.close()

    label_path = json_map[str(int(id))]["24k_frustum_label"]
    

    k = open(label_path)
    data = json.load(k)
    # Closing file
    k.close()
    box_list = []
    labels = []
    for key in data['bounding_boxes'].keys():
        box_list.append(data['bounding_boxes'][key])
        labels.append(key)
    boxes = run1.get_bb_dict_from_bb(corners=box_list, labels=labels)
    # print(boxes)
    obj_path = json_map[str(int(id))]['53k_point_cloud_path']
    pc, _, _ = load_obj(obj_path)
    # shift = YForward2NegZForward()
    # pc = shift(pc)
    pc_dict = run1.get_point_cloud_log(pc, boxes=boxes)
    #pdb.set_trace()
    run1.log(pc_dict)
    return