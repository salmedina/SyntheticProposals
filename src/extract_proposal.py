import argparse
import json
import imageio
import os.path as osp
from PIL import Image
from easydict import EasyDict as edict

def main(anno_path, anno_id, video_path, output_dir):
    annotations = json.load(open(anno_path, 'r'))
    anno = edict(annotations[anno_id])

    print(f'Crop:  {anno.start_frame} - {anno.end_frame}')
    video_reader = imageio.get_reader(video_path)
    for frame_idx, frame_arr in enumerate(video_reader):
        if frame_idx > anno.end_frame:
            break
        if (frame_idx >= anno.start_frame) and (frame_idx <= anno.end_frame):
            print(f'Frame: {frame_idx}')
            bbox = anno.trajectory[str(frame_idx)]
            Image.fromarray(frame_arr).crop(bbox).save(osp.join(output_dir, f'{frame_idx}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, help='Path to json annotation')
    parser.add_argument('--id', type=str, help='Annotation id to be extracted')
    parser.add_argument('--video_path', type=str, help='Path to the video from which the proposal will be extracted')
    parser.add_argument('--output_dir', type=str, help='Output directory of cropped frames')
    args = parser.parse_args()

    main(args.anno_json, args.id, args.video_path, args.output_dir)
