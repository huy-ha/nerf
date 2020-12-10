"""
Training helpers for supervised meta-learning.
"""

import os
import time
import tensorflow as tf
from .nerf_reptile import meta_step, meta_evaluate
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
          log_qualitative_train=5,
          time_deadline=None,
          log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    def create_optimizer(): return tf.keras.optimizers.Adam(
        inner_learning_rate)
    train_writer = SummaryWriter(save_dir + '/train')
    test_writer = SummaryWriter(save_dir + '/test')
    pbar = tqdm(range(meta_iters),
                dynamic_ncols=True,
                smoothing=0.05)
    for i in pbar:
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + \
            (1 - frac_done) * meta_step_size
        loss, psnr, psnr0, trans = \
            meta_step(models=models,
                      dataset=train_set,
                      N_rand=N_rand,
                      chunk=chunk,
                      grad_vars=grad_vars,
                      create_optimizer=create_optimizer,
                      half_res=half_res,
                      testskip=testskip,
                      white_bkgd=white_bkgd,
                      inner_iters=inner_iters,
                      meta_step_size=cur_meta_step_size,
                      meta_batch_size=meta_batch_size,
                      render_kwargs_train=render_kwargs_train,
                      metalearning_iter=i,
                      N_importance=N_importance,
                      use_viewdirs=use_viewdirs,
                      writer=train_writer,
                      save_dir=save_dir,
                      log_qualitative=i % log_qualitative_train == 0)
        pbar.set_description(f'Outer Loss: {float(loss):.02e}')
        train_writer.add_scalar('loss', loss, i)
        train_writer.add_scalar('psnr', psnr, i)
        train_writer.add_histogram('tran', trans, i)
        if N_importance > 0:
            train_writer.add_scalar('psnr0', psnr0, i)
        if i % eval_interval == 0 and i > 0:
            print('#'*10 + ' EVAL ' + '#'*10)
            loss_dict =\
                meta_evaluate(
                    models,
                    i, test_set, N_importance,
                    half_res,
                    testskip,
                    white_bkgd, log_fn, save_dir, N_rand,
                    inner_iters, chunk, use_viewdirs,
                    grad_vars, create_optimizer, render_kwargs_train, render_kwargs_test,
                    render_test_set=i % 100 == 0,
                    writer=test_writer)
            for key, value in loss_dict.items():
                if 'tran' in key:
                    test_writer.add_histogram(key, value, i)
                elif 'psnr0' not in key or N_importance > 0:
                    test_writer.add_scalar(key, value, i)
            output = f'[Average Eval Scenes] ({i} | '
            output += f'Average Loss: {loss_dict["loss/mean/test"]:.5f} | '
            output += f'Average PSNR: {loss_dict["psnr/mean/test"]:.2f} | '
            output += f'Final Loss: {loss_dict["loss/final/test"]:.5f} | '
            output += f'Final PSNR: {loss_dict["psnr/final/test"]:.2f} )'
            log_fn(output)

        if i % 2 == 0 or i == meta_iters-1:
            for key in models:
                path = os.path.join(
                    save_dir, '{}_{:06d}.npy'.format(key, i))
                models[key].save(path)
                print('Saved weights at', path)
        if time_deadline is not None and time.time() > time_deadline:
            break
        train_writer.flush()
        test_writer.flush()
