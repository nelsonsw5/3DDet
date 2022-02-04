import argparse
import json
import pdb
from os import listdir
from os.path import isfile, join
import numpy as np


def get_label(o):
    return o["label"]

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    metadata_file = args.metadata_file

    if metadata_file:
        metadata = json.load(open(metadata_file))
        objects = metadata.values()
        json_files = map(get_label, objects)
    else:
        json_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.split('.')[-1] == "json"]
        json_files.sort()
    i = 0
    for file in json_files:
        txt_file_name = f'{i:06}.txt'
        try:
            with open(output_dir + '/' + txt_file_name, "w") as text_file:
                label = json.load(open(file))
                bbs = label['bounding_boxes']
                # pdb.set_trace()
                for cls, points in bbs.items():
                    class_name = cls.split("_")[0]
                    occluded = 0
                    truncated = 0
                    # pdb.set_trace()
                    # if cls in label['centroids']:
                    #     occluded = 2
                    # else:
                    #     occluded = 0
                    alpha = 0

                    points = np.array(points)
                    x = points[:,0]
                    y = points[:,1]
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
            print("file does not exist")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=None, help='The directory with the json label files')
    parser.add_argument('--output-dir', type=str, default=None, help='The directory where you would like to write the output label files')
    parser.add_argument('--metadata-file', type=str, default=None, help='The directory where you would like to write the output label files')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)