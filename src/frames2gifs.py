import argparse
import imageio
import numpy as np
import os.path as osp
from glob import glob
from PIL import Image
from tqdm import tqdm


def create_gif(frames_dir, output_path, fps=30):
    gif_writer = imageio.get_writer(output_path, fps=fps)
    for frame_path in glob(osp.join(frames_dir, '*.png')):
        frame_array = np.array(Image.open(frame_path))
        gif_writer.append_data(frame_array)
    gif_writer.close()


def main(all_frames_dir, output_dir):

    frames_dir_list = list(glob(osp.join(all_frames_dir, '*/')))
    for frames_dir in tqdm(frames_dir_list, total=len(frames_dir_list), unit='gif'):
        video_name = osp.basename(osp.normpath(frames_dir))
        gif_path = osp.join(output_dir, f'{video_name}.gif')
        create_gif(frames_dir, gif_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, help='Directory with all the proposals frames')
    parser.add_argument('--output_dir', type=str, help='Directory where the gifs will be saved')
    args = parser.parse_args()

    main(args.frames_dir, args.output_dir)