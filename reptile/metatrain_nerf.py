"""
Training helpers for supervised meta-learning.
"""

import os
import time


import tensorflow as tf

from .nerf_reptile import NerfReptile
from .variables import weight_decay
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
# pylint: disable=R0913,R0914


def train(models, grad_vars,
          train_set,
          test_set,
          render_kwargs_train,
          render_kwargs_test,
          save_dir,
          chunk, N_rand,
          N_importance, use_viewdirs,
          half_res,
          testskip,
          white_bkgd,
          inner_iters=10,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          inner_learning_rate=1e-3,
          eval_interval=10,
          time_deadline=None,
          reptile_fn=NerfReptile,
          log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # TODO weight_decay
    reptile = reptile_fn()
    optimizer = tf.keras.optimizers.Adam(
        inner_learning_rate, beta_1=0)
    train_writer = SummaryWriter(save_dir + '/train')
    test_writer = SummaryWriter(save_dir + '/test')
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(0)
    for i in tqdm(range(meta_iters),
                  desc='Reptile on Nerf',
                  dynamic_ncols=True,
                  smoothing=0.05):
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + \
            (1 - frac_done) * meta_step_size
        loss, psnr, psnr0, trans = \
            reptile.meta_step(models=models,
                              dataset=train_set,
                              N_rand=N_rand,
                              chunk=chunk,
                              grad_vars=grad_vars,
                              optimizer=optimizer,
                              half_res=half_res,
                              testskip=testskip,
                              white_bkgd=white_bkgd,
                              inner_iters=inner_iters,
                              meta_step_size=cur_meta_step_size,
                              meta_batch_size=meta_batch_size,
                              render_kwargs_train=render_kwargs_train)
        train_writer.add_scalar('loss', loss, i)
        train_writer.add_scalar('psnr', psnr, i)
        train_writer.add_histogram('tran', trans, i)
        if N_importance > 0:
            train_writer.add_scalar('psnr0', psnr0, i)
        if i % eval_interval == 0 and i > 0:
            print('#'*10 + ' EVAL ' + '#'*10)
            loss, psnr, psnr0, trans = reptile.meta_evaluate(
                models,
                i, test_set, N_importance,
                half_res,
                testskip,
                white_bkgd, log_fn, save_dir, N_rand,
                inner_iters, chunk, use_viewdirs,
                grad_vars, optimizer, render_kwargs_train, render_kwargs_test,
                render_test_set=i % 100 == 0,
                writer=test_writer)
            log_fn(f'[Average Eval Scenes] ({i} | ' +
                   f'loss: {loss:.5f} | PSNR: {psnr:.2f} |')

            test_writer.add_scalar('loss', loss, i)
            test_writer.add_scalar('psnr', psnr, i)
            test_writer.add_histogram('tran', trans, i)
            if N_importance > 0:
                test_writer.add_scalar('psnr0', psnr0, i)

        if i % 10 == 0 or i == meta_iters-1:
            for key in models:
                path = os.path.join(
                    save_dir, '{}_{:06d}.npy'.format(key, i))
                np.save(path, models[key].get_weights())
                print('Saved weights at', path)
        if time_deadline is not None and time.time() > time_deadline:
            break
        train_writer.flush()
        test_writer.flush()
        global_step.assign_add(1)
