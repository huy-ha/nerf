from pathlib import Path
from os.path import abspath, basename, splitext, exists
import json
import numpy as np
from os import mkdir, symlink, rmdir
from random import shuffle
import shutil

def mkdir_new(path):
    if exists(path):
        shutil.rmtree(path)
    mkdir(path)

if __name__ == '__main__':
    for dir_path in list(str(path) for path in Path('./.').glob("*")):
        print(str(dir_path))
        png_paths = [abspath(str(png_path)) for png_path in Path(dir_path).rglob(('*.png'))
                     if 'depth' not in str(png_path) and
                     'normal' not in str(png_path)]

        transforms = json.load(open(f'{dir_path}/transforms.json', 'r'))
        n = len(transforms['frames'])
        print(f'found {n} images')
        shuffle(transforms['frames'])
        train_frames, val_frames = np.split(
            transforms['frames'], [int(.8 * n)])
        test_frames = val_frames
        val_transforms = {
            'camera_angle_x': transforms['camera_angle_x'],
            'frames': list(val_frames)
        }
        #
        test_transforms = {
            'camera_angle_x': transforms['camera_angle_x'],
            'frames': list(test_frames)
        }
        #
        train_transforms = {
            'camera_angle_x': transforms['camera_angle_x'],
            'frames': list(train_frames)
        }

        mkdir_new(f'{dir_path}/test')
        for frame in test_transforms['frames']:
            output = f'{dir_path}/test/' + basename(frame['file_path']) + '.png'
            symlink(frame['file_path']+'.png', output)

        mkdir_new(f'{dir_path}/train')
        for frame in train_transforms['frames']:
            output = f'{dir_path}/train/' + basename(frame['file_path']) + '.png'
            symlink(frame['file_path']+'.png', output)

        mkdir_new(f'{dir_path}/val')
        for frame in val_transforms['frames']:
            output = f'{dir_path}/val/' + basename(frame['file_path']) + '.png'
            symlink(frame['file_path']+'.png', output)


        json.dump(train_transforms, open(f'{dir_path}/transforms_train.json', 'w'))
        json.dump(test_transforms, open(f'{dir_path}/transforms_test.json', 'w'))
        json.dump(val_transforms, open(f'{dir_path}/transforms_val.json', 'w'))
