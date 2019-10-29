import json
import os.path as osp
from easydict import EasyDict
from collections import defaultdict

virat_dataset_dir = '/home/zal/Data/VIRAT/clean_clips'
virat_anno_path = '../../annotations/ssp1.json'

virat_anno = EasyDict(json.load(open(virat_anno_path, 'r')))

count = defaultdict(int)
missing_count = defaultdict(int)
for name, value in virat_anno.database.items():
    if 'phone' in value.annotations.label:
        print(f'{name},{value.subset},{value.annotations.label}')
        count[value.subset] += 1

        if not osp.exists(osp.join(virat_dataset_dir, f'{name}.mp4')):
            missing_count[value.subset] += 1

print('Annotation count')
for k, v in count.items():
    print(f'{k}:    {v}')

print('Missing count')
for k, v in missing_count.items():
    print(f'{k}:    {v}')
