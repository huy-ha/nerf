from load_blender import load_blender_data
from load_deepvoxels import load_dv_data
from load_llff import load_llff_data
from run_nerf_helpers import *
import time
import random
import json
import imageio
import numpy as np
import tensorflow as tf
import sys
import os
from tqdm import tqdm
from reptile.nerf_reptile import train_innerstep

tf.compat.v1.enable_eager_execution()

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = \
        create_nerf(args)

    # Load data
    _, images, poses, render_poses, hwf, i_split, timesteps = \
        load_data(scene_dir_path=args.datadir,
                  white_bkgd=args.white_bkgd,
                  half_res=args.half_res,
                  testskip=args.testskip,
                  num_training=args.num_training)
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    i_train, i_val, i_test = i_split

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(args.lrate)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    args.no_batching = False
    assert not args.no_batching
    rays_rgb = create_ray_batches(H, W, focal, poses, images, i_train)
    train_timesteps = []
    for i in i_train:
        train_timesteps.extend([timesteps[i]] * H * W)
    train_timesteps = np.asarray(train_timesteps)
    idxs = np.random.permutation(rays_rgb.shape[0])
    rays_rgb = rays_rgb[idxs]
    train_timesteps = train_timesteps[idxs]
    i_batch = 0
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    with tqdm(range(start, 1000000), dynamic_ncols=True, smoothing=0.1) as pbar:
        for i in pbar:
            time0 = time.time()
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch_timestep = train_timesteps[i_batch: i_batch + N_rand]
            batch = tf.transpose(batch, [1, 0, 2])
            batch_rays, target_s = batch[:2], batch[2]
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

            #####  Core optimization loop  #####
            loss, psnr, psnr0, trans = train_innerstep(
                batch_rays=batch_rays,
                target_s=target_s,
                chunk=args.chunk,
                H=H, W=W, focal=focal,
                grad_vars=grad_vars,
                optimizer=optimizer,
                batch_timestep=batch_timestep,
                render_kwargs_train=render_kwargs_train)
            dt = time.time()-time0

            #####           end            #####

            # Rest is logging

            def save_weights(net, prefix, i):
                path = os.path.join(
                    basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
                np.save(path, net.get_weights())
                print('saved weights at', path)

            if i % args.i_weights == 0:
                for k in models:
                    save_weights(models[k], k, i)

            if i % args.i_video == 0 and i > 0:
                sorted_timesteps = sorted(list(set(timesteps)))
                set_pose = poses[i_test[0]]
                rgbs, disps = render_timesteps(
                    set_pose, hwf, sorted_timesteps, args.chunk, render_kwargs_test)
                moviebase = os.path.join(
                    basedir, expname,  '{}_temporal_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4',
                                 to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4',
                                 to8b(disps / np.max(disps)), fps=30, quality=8)

                if args.use_viewdirs:
                    render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                    rgbs_still, _ = render_timesteps(
                        set_pose, hwf, sorted_timesteps, args.chunk,
                        render_kwargs_test)
                    render_kwargs_test['c2w_staticcam'] = None
                    imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                     to8b(rgbs_still), fps=30, quality=8)

            if i % args.i_testset == 0 and i > 0:
                testsavedir = os.path.join(
                    basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                split_timesteps = []
                for i in i_test:
                    split_timesteps.extend([timesteps[i]] * H * W)
                split_timesteps = np.asarray(split_timesteps)
                print('test poses shape', poses[i_test].shape)
                render_path(poses[i_test], hwf, split_timesteps, args.chunk,
                            render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                print('Saved test set')

            if i % args.i_print == 0 or i < 10:
                pbar.set_description(
                    f'[{expname}] ({i} | steps:{global_step.numpy()}) | ' +
                    f'loss: {loss.numpy():.5f} | PSNR: {psnr.numpy():.2f} |')
                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                    tf.contrib.summary.scalar('loss', loss)
                    tf.contrib.summary.scalar('psnr', psnr)
                    tf.contrib.summary.histogram('tran', trans)
                    if args.N_importance > 0:
                        tf.contrib.summary.scalar('psnr0', psnr0)

                if i % args.i_img == 0:

                    # Log a rendered validation view to Tensorboard
                    img_i = np.random.choice(i_val)
                    target = images[img_i]
                    pose = poses[img_i, :3, :4]
                    split_timestep = [timesteps[img_i]] * (H*W)
                    rgb, disp, acc, extras = render(H, W, focal, split_timestep, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)

                    psnr = mse2psnr(img2mse(rgb, target))

                    # Save out the validation image for Tensorboard-free monitoring
                    testimgdir = os.path.join(
                        basedir, expname, 'tboard_val_imgs')
                    if i == 0:
                        os.makedirs(testimgdir, exist_ok=True)
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                        tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp', disp[tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'acc', acc[tf.newaxis, ..., tf.newaxis])

                        tf.contrib.summary.scalar('psnr_holdout', psnr)
                        tf.contrib.summary.image(
                            'rgb_holdout', target[tf.newaxis])

                    if args.N_importance > 0:

                        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                            tf.contrib.summary.image(
                                'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                            tf.contrib.summary.image(
                                'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                            tf.contrib.summary.image(
                                'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

            global_step.assign_add(1)
