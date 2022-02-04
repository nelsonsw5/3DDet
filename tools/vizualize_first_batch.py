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
    #pdb.set_trace()
    box_list = []
    for label in train_set.get_label(id):
        # corners = list(label.generate_corners3d())
        # box_list.append(corners)
        labels.append(label.to_kitti_format().split()[0])
        #pdb.set_trace()
        label_str = label.to_kitti_format()
        parsed_label = label_str.split()
        # labels.append(str(parsed_label[0]))
        dims.append([float(parsed_label[8]), float(parsed_label[9]), float(parsed_label[10])])
        centroids.append([float(parsed_label[11]), float(parsed_label[12]), float(parsed_label[13])])
    # print("box_list in DL: ", box_list)
    # boxes = run.get_bb_dict_from_bb(corners=box_list, labels=labels)
    # pdb.set_trace()
    centroids = torch.tensor(centroids)
    dims = torch.tensor(dims)
    # pdb.set_trace()
    pc = torch.Tensor(train_set.get_lidar(id)[:, 0:3])
    # pdb.set_trace()
    # norm = NormalizeShift()
    # pc = norm.fit_transform(pc)
    # centroids = norm(centroids)
    # dims = norm.scale(dims)
    # pdb.set_trace()
    boxes = run.get_bb_dict(centroids, dims, labels=labels)
    #print("boxes: ", boxes[0])
    #print("Dataloader boxes: ", boxes)
    log_dict = run.get_point_cloud_log(pc, boxes=boxes)
    run.log(log_dict)

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