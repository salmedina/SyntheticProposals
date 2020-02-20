import argparse
import json
import pickle
import os.path as osp
import numpy as np
from glob import glob
from collections import defaultdict
from easydict import EasyDict as edict

''' 
This program goes through the MEVA anntotations looking for the highest bounding box of each tracklet
The output gives per camera view a list of <centroid>, <height> to be used to calculate the height of the persons
in the synthetic proposals
'''

def get_camera_id_from_path(input_path):
    # e.g. 2018-03-15.14-55-00.15-00-00.school.G339.json
    tokens = osp.basename(input_path).split('.')
    return tokens[4]


def calc_height(bbox):
    _, y1, _, y2 = bbox
    return y2 - y1


def calc_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)//2, (y1+y2)//2


def main(input_dir, event_types, output_path):

    camera_heights_map = defaultdict(list)
    for json_path in glob(osp.join(input_dir, '*.json')):
        print(json_path)
        camera_id = get_camera_id_from_path(json_path)
        anno_data = json.load(open(json_path))
        for action_id, anno in anno_data.items():
            anno = edict(anno)
            if anno.event_type not in event_types:
                continue

            bbox_centroids = list(map(calc_centroid, anno.trajectory.values()))
            bbox_heights = list(map(calc_height, anno.trajectory.values()))
            max_idx = np.argmax(bbox_heights)
            camera_heights_map[camera_id].append((bbox_centroids[max_idx], bbox_heights[max_idx]))

    pickle.dump(camera_heights_map, open(output_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory with annotation json files')
    parser.add_argument('--event_types', type=str, help='Type of events to check to sample sizes')
    parser.add_argument('--output_path', type=str, help='Path of output npy file')
    args = parser.parse_args()

    event_types = args.event_types.split(',')

    main(args.input_dir, event_types, args.output_path)