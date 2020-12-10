import os
import tensorflow as tf
import numpy as np
import imageio
import json


def trans_t(t): return tf.convert_to_tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1],
], dtype=tf.float32)


def rot_phi(phi): return tf.convert_to_tensor([
    [1, 0, 0, 0],
    [0, tf.cos(phi), -tf.sin(phi), 0],
    [0, tf.sin(phi), tf.cos(phi), 0],
    [0, 0, 0, 1],
], dtype=tf.float32)


def rot_theta(th): return tf.convert_to_tensor([
    [tf.cos(th), 0, -tf.sin(th), 0],
    [0, 1, 0, 0],
    [tf.sin(th), 0, tf.cos(th), 0],
    [0, 0, 0, 1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],
                    [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, num_training=None):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        if s == 'train' and num_training:
            s += "_{}".format(num_training)
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_timesteps = []
    max_time = float('-inf')
    min_time = float('inf')
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        timesteps = []

        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
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

    render_poses = tf.stack([pose_spherical(angle, -30.0, 6.0)
                             for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        imgs = tf.compat.v1.image.resize_area(imgs, [H, W]).numpy()
        focal = focal/2.

    return imgs, poses, render_poses, [H, W, focal], i_split, timesteps
