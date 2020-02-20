import argparse
import pickle
import trimesh
import os
import torch
import os.path as osp
import numpy as np

from glob import glob
from PIL import Image
from collections import namedtuple
from tqdm import tqdm

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.mesh import MeshViewer

CameraParams = namedtuple('CameraParams', ['x', 'y', 'z', 'pitch', 'yaw', 'roll'])
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(12345)

def get_camera_id_from_path(input_path, separator='_'):
    # e.g. 2018-03-15.14-55-00.15-00-00.school.G339.ext
    tokens = osp.splitext(osp.basename(input_path))[0].split(separator)
    return tokens[4]


def load_random_actions(actions_csv_path, num_samples):
    '''
    actions csv fields: npz_path, start_frame, end_frame
    returns: list of (npz,start,end) of size num_actions
    '''
    actions_anno_list = [l.strip().split(',') for l in open(actions_csv_path).readlines()]
    actions_anno_list = [(npz_path, int(start_frame), int(end_frame)) for npz_path, start_frame, end_frame in actions_anno_list]
    num_npz = len(actions_anno_list)
    random_npz_idxs = np.random.choice(num_npz, num_samples, replace=True)

    return [actions_anno_list[idx] for idx in random_npz_idxs]


def load_random_scenes(bg_dir, num_samples):
    '''
    background images are in png format
    return list of (bg_img_path, camera_id)
    '''
    all_bg_list = list(glob(osp.join(bg_dir, '*.png')))
    num_bgs = len(all_bg_list)
    random_idxs = np.random.choice(num_bgs, num_samples, replace=True)

    return [(all_bg_list[idx], get_camera_id_from_path(all_bg_list[idx])) for idx in random_idxs]


def load_random_positions(scene_list, masks_pkl_path):
    '''
    scene_list is list of (bg_img_path, camera_id)
    masks_npy has k:v camera_id:list((x, y))
    '''
    masks_map = pickle.load(open(masks_pkl_path, 'rb'))

    positions_list = list()
    for _, camera_id in scene_list:
        random_idx = np.random.randint(len(masks_map[camera_id]))
        positions_list.append(masks_map[camera_id][random_idx])

    return positions_list


def get_nn_height(pos, pos_heights_list):
    pos_x, pos_y = pos
    pos_height_array = np.array([[x, y, h] for (x, y), h in pos_heights_list])

    nearest_idx = (np.abs(pos_height_array[:, 1]-pos_y)).argmin()
    return pos_height_array[nearest_idx][2]


def load_sample_heights(scene_list, pos_list, heights_pkl_path):
    '''
    scene_list is list of (bg_img_path, camera_id)
    '''
    cam_height_map = pickle.load(open(heights_pkl_path, 'rb'))

    heights_list = list()
    for (_, camera_id), pos in zip(scene_list, pos_list):
        heights_list.append(get_nn_height(pos, cam_height_map[camera_id]))

    return heights_list


def load_random_angles(num_samples):
    mean = [-45., 45.]
    cov = [[180., 0], [0., 180.]]
    x = np.random.multivariate_normal(mean, cov, num_samples//2)
    return np.concatenate((x[:, 0], x[:, 1]))


def load_bg_patch_bboxes(scene_list, pos_list, height_list):
    bbox_list = list()

    for (bg_path, _), (x, y), height in zip(scene_list, pos_list, height_list):
        img = Image.open(bg_path)
        x1 = max(0, x-(height//2))
        y1 = max(0, y-height)
        x2 = min(img.width, x1+height)
        y2 = min(img.height, y1+height)

        bbox_list.append([x1, y1, x2, y2])

    return bbox_list


def save_image_patch(img_path, bbox, output_size, save_path):
    Image.open(img_path).crop(bbox).resize((output_size, output_size)).save(save_path)


def main(num_samples, bg_dir, masks_pkl_path, heights_pkl_path, output_size, output_dir):
    # 1. Gather all the information required
    scenes_list = load_random_scenes(bg_dir, num_samples)
    pos_list = load_random_positions(scenes_list, masks_pkl_path)
    height_list = load_sample_heights(scenes_list, pos_list, heights_pkl_path)
    bg_bbox_list = load_bg_patch_bboxes(scenes_list, pos_list, height_list)

    # 2. Render all the actions based on the information gathered
    for idx in tqdm(range(num_samples), total=num_samples, unit='video'):
        bg_path = scenes_list[idx][0]
        bg_bbox = bg_bbox_list[idx]
        bg_patch_path = osp.join(output_dir, f'{str(idx)}.png')
        save_image_patch(bg_path, bg_bbox, output_size, bg_patch_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, help='Number of synthetic proposal videos to be generated')
    parser.add_argument('--bg_dir', type=str, help='Directory which has the bg images')
    parser.add_argument('--masks_pkl', type=str, help='npy file of positions for camera view dict camera_id:[position]')
    parser.add_argument('--heights_pkl', type=str, help='npy file with the heights at a certain position per camera id')
    parser.add_argument('--output_size', type=int, default=224, help='Size of squared size of output bg patches')
    parser.add_argument('--output_dir', type=str, help='Directory where the samples will be generated')
    args = parser.parse_args()

    '''
    INPUTS
    1
    /home/zal/Devel/SyntheticProposals/annotated_sitting_npz.csv
    /mnt/Alfheim/Data/MEVA/umd_cmu_merge_v3/sampled_frames/
    /mnt/Alfheim/Data/MEVA/umd_cmu_merge_v3/camera_maps/sitdown/camera_masks.pkl
    /home/zal/Devel/SyntheticProposals/cam_heights_sitstand.pkl
    /mnt/Alfheim/Data/MEVA/synthetic_proposals/sitdown
    '''

    main(args.num_samples, args.bg_dir, args.masks_pkl, args.heights_pkl, args.output_size, args.output_dir)
