from .reptile import Reptile
import tensorflow as tf
from .variables import (
    interpolate_vars,
    average_vars,
    subtract_vars,
    add_vars,
    scale_vars,
    VariableState)
import random
import numpy as np
from run_nerf import render, render_path
from run_nerf_helpers import (
    img2mse, mse2psnr, get_rays_np,
    to8b
)
from load_blender import load_blender_data
import os
import imageio


class NerfReptile:
    def __init__(self):
        pass

    # pylint: disable=R0913,R0914

    def train_nerf_step(self,
                        dataset,
                        N_rand,
                        chunk,
                        grad_vars,
                        optimizer,
                        half_res,
                        testskip,
                        white_bkgd,
                        inner_iters,
                        meta_step_size,
                        meta_batch_size,
                        render_kwargs_train):
        old_vars = self._model_state.export_variables()
        new_vars = []
        for (images, poses, render_poses, hwf, i_split) in _sample_scene(
                dataset,  half_res, testskip,
                white_bkgd, meta_batch_size):
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            i_train, i_val, i_test = i_split
            i_batch = 0
            rays_rgb = _create_ray_batches(
                H, W, focal, poses, images, i_train)
            for batch in range(inner_iters):
                batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
                batch = tf.transpose(batch, [1, 0, 2])
                batch_rays, target_s = batch[:2], batch[2]
                if i_batch >= rays_rgb.shape[0]:
                    np.random.shuffle(rays_rgb)
                    i_batch = 0
                i_batch += N_rand
                self.train_innerstep(batch_rays,
                                     target_s,
                                     chunk,
                                     H, W, focal,
                                     grad_vars,
                                     optimizer,
                                     render_kwargs_train)
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(
            interpolate_vars(old_vars, new_vars, meta_step_size))

    def train_innerstep(self,
                        batch_rays,
                        target_s,
                        chunk,
                        H, W, focal,
                        grad_vars,
                        optimizer,
                        render_kwargs_train):
        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal,
                chunk=chunk,
                rays=batch_rays,
                verbose=True,
                retraw=True,
                **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)
            psnr0 = None

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))
        return loss, psnr, psnr0, trans

    def evaluate(self,
                 metalearning_iter,
                 test_scenes,
                 N_importance,
                 log_fn,
                 save_dir,
                 N_rand,
                 inner_iters,
                 chunk, use_viewdirs,
                 grad_vars, optimizer,
                 render_kwargs_train,
                 render_kwargs_test):
        old_vars = self._full_state.export_variables()
        losses = []
        psnrs = []
        psnr0s = []
        transs = []
        for test_scene in test_scenes:
            scene_id, images, poses, render_poses, hwf, i_split = load_data(
                test_scene)
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            i_train, i_val, i_test = i_split
            # Train
            i_batch = 0
            rays_rgb = _create_ray_batches(
                H, W, focal, poses, images, i_train)
            for batch in range(inner_iters):
                batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
                batch = tf.transpose(batch, [1, 0, 2])
                batch_rays, target_s = batch[:2], batch[2]
                if i_batch >= rays_rgb.shape[0]:
                    np.random.shuffle(rays_rgb)
                    i_batch = 0
                i_batch += N_rand
                loss, psnr, psnr0, trans = self.train_innerstep(
                    batch_rays, target_s, chunk, H, W, focal,
                    grad_vars, optimizer, render_kwargs_train)
            losses.append(loss)
            psnrs.append(psnr)
            psnr0s.append(psnr0)
            transs.append(trans)

            testsavedir = os.path.join(
                save_dir, 'testset_{:06d}'.format(metalearning_iter))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwf, chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

            # Save videos
            rgbs, disps = render_path(
                render_poses, hwf, chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                save_dir, '{}_spiral_{:06d}_'.format(
                    scene_id,
                    metalearning_iter))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

            if use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                rgbs_still, _ = render_path(
                    render_poses, hwf, chunk,
                    render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=30, quality=8)

            # Log
            log_fn(
                f'[{scene_id}] ({metalearning_iter} | ' +
                f'loss: {loss:.5f} | PSNR: {psnr:.2f} |')

            tf.contrib.summary.scalar('loss', loss)
            tf.contrib.summary.scalar('psnr', psnr)
            tf.contrib.summary.histogram('tran', trans)
            if N_importance > 0:
                tf.contrib.summary.scalar('psnr0', psnr0)

            # Log a rendered validation view to Tensorboard
            img_i = np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            rgb, disp, acc, extras = render(
                H, W, focal, chunk=chunk, c2w=pose, **render_kwargs_test)

            psnr = mse2psnr(img2mse(rgb, target))

            # Save out the validation image for Tensorboard-free monitoring
            testimgdir = os.path.join(
                save_dir, 'tboard_val_imgs')
            if not os.path.exists(testimgdir):
                os.makedirs(testimgdir, exist_ok=True)
            imageio.imwrite(os.path.join(
                testimgdir, '{:06d}.png'.format(metalearning_iter)), to8b(rgb))

            tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
            tf.contrib.summary.image(
                'disp', disp[tf.newaxis, ..., tf.newaxis])
            tf.contrib.summary.image(
                'acc', acc[tf.newaxis, ..., tf.newaxis])

            tf.contrib.summary.scalar('psnr_holdout', psnr)
            tf.contrib.summary.image(
                'rgb_holdout', target[tf.newaxis])

            if N_importance > 0:
                tf.contrib.summary.image(
                    'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                tf.contrib.summary.image(
                    'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                tf.contrib.summary.image(
                    'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])
            self._full_state.import_variables(old_vars)
        return np.mean(losses), np.mean(psnrs), np.mean(psnr0s), np.mean(transs)


def _create_ray_batches(H, W, focal, poses, images, i_train):
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = np.stack([rays_rgb[i]
                         for i in i_train], axis=0)
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    return rays_rgb


def load_data(scene_dir_path, white_bkgd, half_res, testskip):
    images, poses, render_poses, hwf, i_split = load_blender_data(
        scene_dir_path, half_res, testskip)
    print('Loaded blender', images.shape,
          render_poses.shape, hwf, scene_dir_path)

    if white_bkgd:
        images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
    else:
        images = images[..., :3]
    return os.path.basename(scene_dir_path), images, poses, render_poses, hwf, i_split


def _sample_scene(dataset,
                  half_res,
                  testskip,
                  white_bkgd,
                  meta_batch_size):
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for scene_path in enumerate(shuffled[:meta_batch_size]):
        yield load_data(scene_path,
                        white_bkgd=white_bkgd,
                        half_res=half_res,
                        testskip=testskip)
