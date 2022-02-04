import argparse
import json
import pdb
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
import pytorch3d
from pytorch3d.io import load_obj
from transforms import NormalizeShift


def get_obj(o):
    # pdb.set_trace()
    return o["point_cloud"]


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    metadata_file = args.metadata_file

    if metadata_file:
        metadata = json.load(open(metadata_file))
        objects = metadata.values()
        obj_files = map(get_obj, objects)
    else:
        obj_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.split('.')[-1] == "obj"]
        obj_files.sort()

    i = 0

    global_max = [0,0,0]
    global_min = [0,0,0]

    for file in tqdm(obj_files):
        try:
            # print(file)
            # pdb.set_trace()
            # file = input_dir + file
            bin_file_name = f'{i:06}.bin'\

            point_cloud = pytorch3d.io.load_obj(file)
            np_point_cloud = point_cloud[0].data.numpy()
            
            xarr = np_point_cloud[:,0]
            yarr = np_point_cloud[:,1]
            zarr = np_point_cloud[:,2]
            ## Get data from pcd (x, y, z, intensity, ring, time)
            np_x = (np.array(xarr, dtype=np.float32)).astype(np.float32)
            np_y = (np.array(yarr, dtype=np.float32)).astype(np.float32)
            np_z = (np.array(zarr, dtype=np.float32)).astype(np.float32)

            max_val = [np.max(np_x), np.max(np_y), np.max(np_z)]
            min_val = [np.min(np_x), np.min(np_y), np.min(np_z)]

            global_max = np.maximum(max_val, global_max)
            global_min = np.minimum(min_val, global_min)

            np_i = (np.ones(len(zarr), dtype=np.float32)).astype(np.float32)
            points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
            points_32.tofile(output_dir + '/' + bin_file_name)
            i += 1
        
        # pdb.set_trace()
    print(global_max)
    print(global_min)
    print(i)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=None, help='The directory with the obj files')
    parser.add_argument('--output-dir', type=str, default=None, help='The directory where you would like to write the output bin files')
    parser.add_argument('--metadata-file', type=str, default=None, help='The directory where you would like to write the output label files')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)