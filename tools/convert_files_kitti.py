from transforms import NormalizeShift
import json
import pdb
from os import listdir
from os.path import isfile, join
import numpy as np
import pytorch3d
from pytorch3d.io import load_obj
import torch
from tqdm import tqdm


def convert_cloud(file, i):
    try:
        bin_file_name = f'{i:06}.bin'
        point_cloud = pytorch3d.io.load_obj(file)
        np_point_cloud = point_cloud[0]
        xarr = np_point_cloud[:,1]
        yarr = -1 * np_point_cloud[:,0]
        zarr = np_point_cloud[:,2]
        ## Get data from pcd (x, y, z, intensity, ring, time)
        np_x = (np.array(xarr, dtype=np.float32)).astype(np.float32)
        np_y = (np.array(yarr, dtype=np.float32)).astype(np.float32)
        np_z = (np.array(zarr, dtype=np.float32)).astype(np.float32)
        np_i = (np.ones(len(zarr), dtype=np.float32)).astype(np.float32)
        transformed_cloud = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
        transformed_cloud.tofile('/multiview/3d-count/obj_detection/normalized_data/testing/velodyne/' + bin_file_name)
        i += 1
    except:
        print("something went wrong")
    return

def convert_label(file, i, norm_shift=None):
    try:
        txt_file_name = f'{i:06}.txt'
        with open('/multiview/3d-count/obj_detection/normalized_data/testing/label_2/' + txt_file_name, "w") as text_file:
            label = json.load(open(file))
            bbs = label['bounding_boxes']
            for cls, points in bbs.items():
                class_name = cls.split("_")[0]
                occluded = 0
                truncated = 0
                alpha = 0
                points = np.array(points)
                x = points[:,1]
                y = -1 * points[:,0]
                z = points[:,2]
                width = max(x) - min(x)
                height = max(y) - min(y)
                length = max(z) - min(z)
                location_x = np.mean(x)
                location_y = np.mean(y)
                location_z = np.mean(z)
                out_arr = list([class_name, str(float(truncated)), str(int(occluded)), str(float(alpha)), str(0.0), str(0.0), str(0.0), str(0.0), str(height), str(width), str(length), str(location_x), str(location_y), str(location_z)])
                out_str = " ".join(out_arr) + '\n'
                text_file.write(out_str)
            i += 1
    except:
        print("something went wrong")
    
metadata_file = '/multiview/3d-count/obj_detection/normalized_data/test-metadata.json'
metadata = json.load(open(metadata_file))
counter = 0
scene_list = []
for scenes in metadata.values():
    scene_list.append(scenes)
pbar = tqdm(range(len(scene_list)), total=len(scene_list))
for i in pbar:
    pc_file = scene_list[i]['point_cloud']
    label = scene_list[i]['label']
    convert_cloud(pc_file,counter)
    convert_label(label, counter)
    counter = counter + 1