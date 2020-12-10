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
from run_nerf_helpers import (
    img2mse, mse2psnr, get_rays_np,
    to8b, render, render_path, render_timesteps,
    create_ray_batches, load_data)
from load_blender import load_blender_data
import os
import imageio
from copy import deepcopy
from tqdm import tqdm


def set_variables(models, variables):
    for key, model in models.items():
        model.set_weights(variables[key])


def meta_step(models,
              dataset,
              N_rand,
              chunk,
              grad_vars,
              create_optimizer,
              half_res,
              testskip,
              white_bkgd,
              inner_iters,
              meta_step_size,
              meta_batch_size,
              render_kwargs_train,
              metalearning_iter,
              N_importance,
              use_viewdirs,
              writer,
              save_dir,
              log_qualitative=False):
    old_vars = {k: deepcopy(v.get_weights())
                for k, v in models.items()}
    new_vars = []
    losses = []
    psnrs = []
    psnr0s = []
    transs = []
    for scene_id, (scene_path, images, poses, render_poses, hwf, i_split, timesteps) in enumerate(_sample_scene(
            dataset,  half_res, testskip,
            white_bkgd, meta_batch_size)):
        optimizer = create_optimizer()
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        i_train, i_val, i_test = i_split
        i_batch = 0
        rays_rgb = create_ray_batches(
            H, W, focal, poses, images, i_train)
        train_timesteps = []
        for i in i_train:
            train_timesteps.extend([timesteps[i]] * H * W)
        train_timesteps = np.asarray(train_timesteps)

        idxs = np.random.permutation(rays_rgb.shape[0])
        rays_rgb = rays_rgb[idxs]
        train_timesteps = train_timesteps[idxs]
        losses.append([])
        psnrs.append([])
        psnr0s.append([])
        transs.append([])
        for batch_idx, batch in enumerate(range(inner_iters)):
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch_timestep = train_timesteps[i_batch: i_batch + N_rand]
            batch = tf.transpose(batch, [1, 0, 2])
            batch_rays, target_s = batch[:2], batch[2]
            if i_batch >= rays_rgb.shape[0]:
                # np.random.shuffle(rays_rgb)
                idxs = np.random.permutation(rays_rgb.shape[0])
                rays_rgb = rays_rgb[idxs]
                train_timesteps = train_timesteps[idxs]
                i_batch = 0
            i_batch += N_rand
            loss, psnr, psnr0, trans = \
                train_innerstep(batch_rays,
                                target_s,
                                chunk,
                                H, W, focal,
                                grad_vars,
                                optimizer,
                                batch_timestep,
                                render_kwargs_train)
            step = metalearning_iter * meta_batch_size * inner_iters +\
                inner_iters * scene_id + batch_idx
            writer.add_scalar('metatrain_loss', float(loss), step)
            writer.add_scalar('metatrain_psnr', float(psnr), step)
            losses[-1].append(loss)
            psnrs[-1].append(psnr)
            psnr0s[-1].append(psnr0)
            transs[-1].append(trans)
        new_vars.append({k: deepcopy(v.get_weights())
                         for k, v in models.items()})
        if log_qualitative:
            log_qualitative_results(
                writer,
                metalearning_iter,
                f'{scene_id:04d}',
                save_dir,
                render_poses,
                poses,
                i_test,
                hwf, chunk,
                render_kwargs_train,
                images, N_importance,
                timesteps, use_viewdirs,
                render_test_set=False)
        set_variables(models, old_vars)
    new_vars = {k: average_vars([
        variables[k]
        for variables in new_vars])
        for k in old_vars.keys()}
    set_variables(models, {
        k: interpolate_vars(
            old_vars[k], new_vars[k], meta_step_size)
        for k in new_vars.keys()})
    losses = np.array(losses)
    psnrs = np.array(psnrs)
    psnr0s = np.array(psnr0s)
    transs = np.array(transs)
    return {
        'loss/init': np.mean(losses[:, 0]),
        'psnr/init': np.mean(psnrs[:, 0]),
        'psnr0/init': np.mean(psnr0s[:, 0]),
        'trans/init': np.mean(transs[:, 0]),
        'loss/mean': np.mean(losses),
        'psnr/mean': np.mean(psnrs),
        'psnr0/mean': np.mean(psnr0s),
        'trans/mean': np.mean(transs),
        'loss/final': np.mean(losses[:, -1]),
        'psnr/final': np.mean(psnrs[:, -1]),
        'psnr0/final': np.mean(psnr0s[:, -1]),
        'trans/final': np.mean(transs[:, -1]),
        'loss/improvement': np.mean(losses[:, -1] - losses[:, 0]),
        'psnr/improvement': np.mean(psnrs[:, -1] - psnrs[:, 0]),
        'psnr0/improvement': np.mean(psnr0s[:, -1] - psnr0s[:, 0]),
        'trans/improvement': np.mean(transs[:, -1] - transs[:, 0]),
    }


def get_losses(batch_rays,
               target_s,
               chunk,
               H, W, focal,
               batch_timestep,
               render_kwargs_train):
    # Make predictions for color, disparity, accumulated opacity.
    rgb, disp, acc, extras = render(
        H, W, focal,
        batch_timestep,
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
    psnr0 = 0.0

    # Add MSE loss for coarse-grained model
    if 'rgb0' in extras:
        img_loss0 = img2mse(extras['rgb0'], target_s)
        loss += img_loss0
        psnr0 = mse2psnr(img_loss0)

    return loss, psnr, psnr0, trans


def train_innerstep(batch_rays,
                    target_s,
                    chunk,
                    H, W, focal,
                    grad_vars,
                    optimizer,
                    batch_timestep,
                    render_kwargs_train):
    with tf.GradientTape() as tape:
        loss, psnr, psnr0, trans = \
            get_losses(batch_rays, target_s, chunk, H,
                       W, focal, batch_timestep, render_kwargs_train)

    gradients = tape.gradient(loss, grad_vars)
    optimizer.apply_gradients(zip(gradients, grad_vars))
    return loss, psnr, psnr0, trans


def meta_evaluate(models,
                  metalearning_iter,
                  test_scenes,
                  N_importance,
                  half_res,
                  testskip,
                  white_bkgd,
                  log_fn,
                  save_dir,
                  N_rand,
                  inner_iters,
                  chunk, use_viewdirs,
                  grad_vars, create_optimizer,
                  render_kwargs_train,
                  render_kwargs_test,
                  render_test_set=False,
                  writer=None,
                  max_eval_scenes=None):
    old_vars = {k: deepcopy(v.get_weights())
                for k, v in models.items()}
    train_losses = []
    train_psnrs = []
    train_psnr0s = []
    train_transs = []
    train_final_losses = []
    train_final_psnrs = []
    train_final_psnr0s = []
    train_final_transs = []

    test_losses = []
    test_psnrs = []
    test_psnr0s = []
    test_transs = []
    test_final_losses = []
    test_final_psnrs = []
    test_final_psnr0s = []
    test_final_transs = []
    for test_scene_i, test_scene_path in enumerate(test_scenes):
        if max_eval_scenes is not None and \
                test_scene_i > max_eval_scenes-1:
            break
        set_variables(models, old_vars)
        optimizer = create_optimizer()
        scene_id, images, poses, render_poses, hwf, i_split, timesteps = load_data(
            test_scene_path,
            white_bkgd=white_bkgd,
            half_res=half_res,
            testskip=testskip)
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        i_train, i_val, i_test = i_split
        # Train on meta test scene
        i_batch = 0
        rays_rgb = create_ray_batches(
            H, W, focal, poses, images, i_train)
        scene_losses = []
        scene_psnrs = []
        scene_psnr0s = []
        scene_transs = []
        for batch_inner_iter in range(inner_iters):
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])
            batch_rays, target_s = batch[:2], batch[2]
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0
            i_batch += N_rand
            loss, psnr, psnr0, trans = train_innerstep(
                batch_rays, target_s, chunk, H, W, focal,
                grad_vars, optimizer, batch_timestep, render_kwargs_train)
            if psnr0 is None:
                psnr0 = 0.0
            scene_losses.append(loss)
            scene_psnrs.append(psnr)
            scene_psnr0s.append(psnr0)
            scene_transs.append(trans)
            if batch_inner_iter == inner_iters - 1:
                # Add final
                train_final_losses.append(loss)
                train_final_psnrs.append(psnr)
                train_final_psnr0s.append(psnr0)
                train_final_transs.append(trans)

        scene_losses = np.mean(scene_losses)
        scene_psnrs = np.mean(scene_psnrs)
        scene_psnr0s = np.mean(scene_psnr0s)
        scene_transs = np.mean(scene_transs)
        # Add average
        train_losses.append(scene_losses)
        train_psnrs.append(scene_psnrs)
        train_psnr0s.append(scene_psnr0s)
        train_transs.append(scene_transs)

        # Test on meta test scene
        i_batch = 0
        rays_rgb = create_ray_batches(
            H, W, focal, poses, images, i_val)

        val_timesteps = []
        for i in i_val:
            val_timesteps.extend([timesteps[i]] * H * W)
        val_timesteps = np.asarray(val_timesteps)

        idxs = np.random.permutation(rays_rgb.shape[0])
        rays_rgb = rays_rgb[idxs]
        val_timesteps = val_timesteps[idxs]

        scene_losses = []
        scene_psnrs = []
        scene_psnr0s = []
        scene_transs = []
        for batch_inner_iter in range(inner_iters):
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])
            batch_timestep = val_timesteps[i_batch: i_batch + N_rand]
            batch_rays, target_s = batch[:2], batch[2]
            if i_batch >= rays_rgb.shape[0]:
                idxs = np.random.permutation(rays_rgb.shape[0])
                rays_rgb = rays_rgb[idxs]
                train_timesteps = train_timesteps[idxs]
                # np.random.shuffle(rays_rgb)
                i_batch = 0
            i_batch += N_rand
            loss, psnr, psnr0, trans = \
                get_losses(batch_rays, target_s, chunk, H,
                           W, focal, batch_timestep, render_kwargs_train)
            if psnr0 is None:
                psnr0 = 0.0
            scene_losses.append(loss)
            scene_psnrs.append(psnr)
            scene_psnr0s.append(psnr0)
            scene_transs.append(trans)
            if batch_inner_iter == inner_iters - 1:
                # Add final
                test_final_losses.append(loss)
                test_final_psnrs.append(psnr)
                test_final_psnr0s.append(psnr0)
                test_final_transs.append(trans)

        scene_losses = np.mean(scene_losses)
        scene_psnrs = np.mean(scene_psnrs)
        scene_psnr0s = np.mean(scene_psnr0s)
        scene_transs = np.mean(scene_transs)
        # Add average
        test_losses.append(scene_losses)
        test_psnrs.append(scene_psnrs)
        test_psnr0s.append(scene_psnr0s)
        test_transs.append(scene_transs)

        # Log
        log_fn(f'[Eval Scene #{scene_id}] ({metalearning_iter} | ' +
               f'loss: {scene_losses:.5f} | PSNR: {scene_psnrs:.2f} )')
        log_qualitative_results(writer, metalearning_iter, scene_id, save_dir,
                                render_poses, poses, i_test, hwf, chunk,
                                render_kwargs_test, images, N_importance,
                                timesteps, use_viewdirs, render_test_set)

    set_variables(models, old_vars)
    return {
        #  Meta test train
        'loss/mean/train': np.mean(train_losses),
        'psnr/mean/train': np.mean(train_psnrs),
        'psnr0/mean/train': np.mean(train_psnr0s),
        'trans/mean/train': np.mean(train_transs),
        'loss/final/train': np.mean(train_final_losses),
        'psnr/final/train': np.mean(train_final_psnrs),
        'psnr0/final/train': np.mean(train_final_psnr0s),
        'trans/final/train': np.mean(train_final_transs),
        #  Meta test test
        'loss/mean/test': np.mean(test_losses),
        'psnr/mean/test': np.mean(test_psnrs),
        'psnr0/mean/test': np.mean(test_psnr0s),
        'trans/mean/test': np.mean(test_transs),
        'loss/final/test': np.mean(test_final_losses),
        'psnr/final/test': np.mean(test_final_psnrs),
        'psnr0/final/test': np.mean(test_final_psnr0s),
        'trans/final/test': np.mean(test_final_transs),
    }


def log_qualitative_results(writer,
                            metalearning_iter,
                            scene_id,
                            save_dir,
                            render_poses,
                            poses,
                            i_split,
                            hwf,
                            chunk,
                            render_kwargs_test,
                            images,
                            N_importance,
                            timesteps,
                            use_viewdirs=True,
                            render_test_set=False):
    H, W, focal = hwf
    testsavedir = os.path.join(
        save_dir, 'testset_iter/{:06d}/scene{}'.format(
            metalearning_iter, scene_id))
    os.makedirs(testsavedir, exist_ok=True)
    split_timesteps = []
    for i in i_split:
        split_timesteps.extend([timesteps[i]] * H * W)
    split_timesteps = np.asarray(split_timesteps)

    render_path(poses[i_split], hwf, split_timesteps, chunk, render_kwargs_test,
                gt_imgs=images[i_split], savedir=testsavedir)
    # Log a rendered validation view to Tensorboard
    img_i = np.random.choice(i_split)
    target = images[img_i]
    pose = poses[img_i, :3, :4]
    split_timestep = [timesteps[img_i]] * (H*W)

    rgb, disp, acc, extras = render(
        H, W, focal, split_timestep, chunk=chunk, c2w=pose, **render_kwargs_test)

    psnr = mse2psnr(img2mse(rgb, target))

    # Save out the validation image for Tensorboard-free monitoring
    testimgdir = os.path.join(
        save_dir, 'tboard_val_imgs')
    if not os.path.exists(testimgdir):
        os.makedirs(testimgdir, exist_ok=True)
    imageio.imwrite(os.path.join(
        testimgdir, '{:06d}_{}.png'.format(
            metalearning_iter,
            scene_id)), to8b(rgb))

    writer.add_image(
        f'rgb/{scene_id}',
        np.squeeze(to8b(rgb)[tf.newaxis], axis=0),
        metalearning_iter,
        dataformats='HWC')
    writer.add_image(
        f'disp/{scene_id}',
        np.squeeze(disp[tf.newaxis, ..., tf.newaxis], axis=0),
        metalearning_iter,
        dataformats='HWC')
    writer.add_image(
        f'acc/{scene_id}',
        np.squeeze(acc[tf.newaxis, ..., tf.newaxis], axis=0),
        metalearning_iter,
        dataformats='HWC')
    writer.add_image(
        f'rgb_holdout/{scene_id}',
        np.squeeze(target[tf.newaxis], axis=0),
        metalearning_iter,
        dataformats='HWC')
    if N_importance > 0:
        writer.add_image(
            f'rgb0/{scene_id}',
            np.squeeze(to8b(extras['rgb0'])[tf.newaxis], axis=0),
            metalearning_iter,
            dataformats='HWC')
        writer.add_image(
            f'disp0/{scene_id}',
            np.squeeze(extras['disp0']
                       [tf.newaxis, ..., tf.newaxis], axis=0),
            metalearning_iter,
            dataformats='HWC')
        writer.add_image(
            f'z_std/{scene_id}',
            np.squeeze(extras['z_std']
                       [tf.newaxis, ..., tf.newaxis], axis=0),
            metalearning_iter,
            dataformats='HWC')

    # Save videos
    if render_test_set:
        sorted_timesteps = sorted(list(set(timesteps)))
        rgbs, disps = render_timesteps(
            poses[i_split[0]], hwf, sorted_timesteps, chunk, render_kwargs_test, savedir=testsavedir)
        moviebase = os.path.join(
            save_dir, '{}_temporal_{:06d}_'.format(
                scene_id,
                metalearning_iter))
        imageio.mimwrite(moviebase + 'rgb.mp4',
                         to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4',
                         to8b(disps / np.max(disps)), fps=30, quality=8)

        if use_viewdirs:
            render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
            rgbs_still, _ = render_path(
                render_poses, hwf, sorted_timesteps, chunk,
                render_kwargs_test)
            render_kwargs_test['c2w_staticcam'] = None
            imageio.mimwrite(moviebase + 'rgb_still.mp4',
                             to8b(rgbs_still), fps=30, quality=8)


def _sample_scene(dataset,
                  half_res,
                  testskip,
                  white_bkgd,
                  meta_batch_size):
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for scene_path in shuffled[:meta_batch_size]:
        yield load_data(scene_path,
                        white_bkgd=white_bkgd,
                        half_res=half_res,
                        testskip=testskip)
