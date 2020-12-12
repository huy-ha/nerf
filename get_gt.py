import json
import imageio
import numpy as np
import tensorflow as tf
import os

tf.compat.v1.enable_eager_execution()


def load_gt_data(basedir, timestep, testskip=1):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    all_imgs = []
    all_poses = []
    all_timesteps = []
    max_time = float('-inf')
    min_time = float('inf')
    counts = [0]
    imgs = []
    poses = []
    timesteps = []

    for frame in meta['frames']:
        if frame['timestep'] == timestep:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            timestep = frame['timestep']
            timesteps.append(timestep)
            max_time = max(max_time, timestep)
            min_time = min(min_time, timestep)
    # keep all 4 channels (RGBA)
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)
    all_timesteps.append(timesteps)


    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    timesteps = np.concatenate(all_timesteps, 0) # 826
    timesteps = [(t - min_time)/max_time - 0.5 for t in timesteps]

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # render_poses = tf.stack([pose_spherical(angle, -30.0, 6.0)
    #                          for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    return imgs, poses, [H, W, focal], i_split, timesteps


unseen_timesteps = [1, 35]
for t in unseen_timesteps:
    # Load data
    images, poses, hwf, i_split, timesteps = \
        load_gt_data(scene_dir_path='debug/277',
                     testskip=1)
    # Cast intrinsics to right types
    # H, W, focal = hwf
    # H, W = int(H), int(W)
    # unseen_t = np.array([t] * (H * W))
    moviebase = os.path.join(
        '/home/sujipark/nerf', 'gt_{}'.format(t))
    imageio.mimwrite(moviebase + 'rgb.mp4',
                     images, fps=30, quality=8)