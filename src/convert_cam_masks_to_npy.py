import argparse
import pickle
import numpy as np
import os.path as osp
from glob import glob
from PIL import Image


def main(input_dir, output_path):
    masks_map = dict()
    for mask_path in glob(osp.join(input_dir, '*.jpg')):
        print(f'Processing {mask_path}')
        camera_id = osp.splitext(osp.basename(mask_path))[0]
        mask_arr = np.array(Image.open(mask_path))
        height, width, _ = mask_arr.shape
        pos_list = list()
        for y in range(height):
            for x in range(width):
                if not np.array_equal(mask_arr[y, x], np.array([255, 255, 255])):
                    pos_list.append((x, y))

        masks_map[camera_id] = pos_list

    pickle.dump(masks_map, open(output_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks_dir', type=str, help='Directory with the camera affordance masks')
    parser.add_argument('--output_path', type=str, help='Path of npy file that stores pixel pos')
    args = parser.parse_args()

    main(args.masks_dir, args.output_path)