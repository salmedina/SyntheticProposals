import argparse
import os
import os.path as osp
import trimesh
import torch
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres

from PIL import Image

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_frame(mv, bdata, bm, faces, frame_num, x, y, z, pitch, yaw, roll):
    fId = frame_num  # frame id of the mocap sequence

    root_orient = torch.Tensor(bdata['poses'][fId:fId + 1, :3]).to(comp_device)  # controls the global root orientation
    pose_body = torch.Tensor(bdata['poses'][fId:fId + 1, 3:66]).to(comp_device)  # controls the body
    pose_hand = torch.Tensor(bdata['poses'][fId:fId + 1, 66:]).to(comp_device)  # controls the finger articulation
    betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)  # controls the body shape
    dmpls = torch.Tensor(bdata['dmpls'][fId:fId + 1]).to(comp_device)  # controls soft tissue dynamics
    rotation = torch.tensor([[np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)]]).to(comp_device)

    tx, ty, tz = x, y, z
    camera_pose = np.array([[1., 0., 0, tx],
                            [0., 1., 0, ty],
                            [0., 0., 1., tz],
                            [0., 0., 0., 1.]])
    mv.update_camera_pose(camera_pose)
    mv.set_background_color([1., 1., 1.])

    body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, dmpls=dmpls, root_orient=rotation)
    body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['pink'], (6890, 1)))

    mv.set_static_meshes([body_mesh])

    return mv.render(render_wireframe=False)

def main(args):
    bm_path = osp.join(args.dataset_dir, 'body_models/smplh/male/model.npz')
    dmpl_path = osp.join(args.dataset_dir, 'body_models/dmpls/male/model.npz')
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    bm = BodyModel(bm_path=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, path_dmpl=dmpl_path).to(comp_device)
    faces = c2c(bm.f)
    mv = MeshViewer(width=args.output_width, height=args.output_height, use_offscreen=True)

    csv_lines = [l.strip().split(',') for l in open(args.csv_path, 'r').readlines()]

    for npz_path, start_frame, end_frame, total_frames, mocap_fr, duration in csv_lines[1:]:
        print(npz_path, start_frame, end_frame, total_frames, mocap_fr, duration)
        npz_path, start_frame, end_frame, total_frames, mocap_fr, duration = npz_path, int(start_frame), int(end_frame), int(total_frames), float(mocap_fr), float(duration)
        npz_bdata_path = osp.join(args.dataset_dir, npz_path)
        bdata = np.load(npz_bdata_path)
        sampling_step = bdata['mocap_framerate'] / 30.
        sampled_frame_indices = np.round(np.arange(start_frame, end_frame, sampling_step)).astype(int)
        if sampled_frame_indices[-1] == total_frames:
            sampled_frame_indices[-1] -= 1

        sample_dir = osp.join(args.output_dir, osp.splitext(osp.basename(npz_path))[0])
        os.makedirs(sample_dir, exist_ok=True)
        print(sample_dir)

        for idx, frame_idx in enumerate(sampled_frame_indices):
            frame_path = osp.join(sample_dir, f'{idx:05}.png')
            Image.fromarray(render_frame(mv, bdata, bm, faces, frame_idx,
                                         args.cam_x, args.cam_y, args.cam_z,
                                         args.pitch, args.yaw, args.roll)).save(frame_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to the root of the AMASS dataset')
    parser.add_argument('--csv_path', type=str, help='Path to the csv file with the annotations')
    parser.add_argument('--output_dir', type=str, help='Output path for the video frames')
    parser.add_argument('--output_fr', type=float, default=30., help='Framerate at which the npz will be sampled')
    parser.add_argument('--output_height', type=int, default=360, help='Output frames width')
    parser.add_argument('--output_width', type=int, default=640, help='Output frames height')
    parser.add_argument('--cam_x', type=float, default=0.0, help='Camera X translation')
    parser.add_argument('--cam_y', type=float, default=-0.25, help='Camera Y translation')
    parser.add_argument('--cam_z', type=float, default=1.9, help='Camera Z translation')
    parser.add_argument('--pitch', type=float, default=0.0, help='Camera Pitch')
    parser.add_argument('--yaw', type=float, default=0.0, help='Camera Yaw')
    parser.add_argument('--roll', type=float, default=0.0, help='Camera Roll')

    args = parser.parse_args()
    main(args)