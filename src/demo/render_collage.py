import argparse
import os
import numpy as np
import os.path as osp
from glob import glob
from PIL import Image


def main(args):
    input_height, input_width = 360, 640
    output_height, output_width = 1080, 1920
    dir_list = [osp.join(args.input_dir, ind_dir) for ind_dir in os.listdir(args.input_dir)]
    ind_frame_paths = [sorted(glob(os.path.join(ind_list, '*.png'))) for ind_list in dir_list]
    num_ind_frames = [len(frame_paths) for frame_paths in ind_frame_paths]
    min_num_frames = min(num_ind_frames)
    total_frames = min_num_frames * 3

    for frame_idx in range(total_frames):
        output_array = np.zeros((output_height, output_width, 4), dtype=np.uint8)
        for x_pos in range(3):
            for y_pos in range(3):
                global_pos = x_pos*3 + y_pos
                subframe_array = np.array(Image.open(ind_frame_paths[global_pos][frame_idx%num_ind_frames[global_pos]]))
                output_array[y_pos*input_height:(y_pos+1)*input_height, x_pos*input_width:(x_pos+1)*input_width, :] = subframe_array
        save_path = osp.join(args.output_dir, f'{frame_idx:05}.png')
        Image.fromarray(output_array).save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory that has rendered frames of actions')
    parser.add_argument('--output_dir', type=str, help='Directory where the collage frames will be saved')
    parser.add_argument('--cycles', type=int, default=3, help='Number of times which the minimum video will be repeated')
    parser.add_argument('--in_height', type=int, default=360, help='Individual frames height')
    parser.add_argument('--in_width', type=int, default=640, help='Individual frames width')
    args = parser.parse_args()

    main(args)