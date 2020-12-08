import tensorflow as tf
import random
from reptile.metatrain_nerf import train
from reptile.args import train_kwargs, evaluate_kwargs
from reptile.eval import evaluate
from data_utils import read_dataset
from run_nerf import create_nerf
from run_nerf import config_parser
import numpy as np
import os
from pprint import pprint


def parse_args():
    parser = config_parser()
    parser.add_argument('--metadatadir', type=str,
                        default='data/nerf_synthetic/metacubes/',
                        help="path of directory containing all scene " +
                        "directories, each containing posed RGB images ")
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    # parser.add_argument(
    #     '--checkpoint', help='checkpoint directory', default='model_checkpoint')
    parser.add_argument(
        '--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument(
        '--train-shots', help='shots in a training batch', default=0, type=int)
    parser.add_argument(
        '--inner-batch', help='inner batch size', default=5, type=int)
    parser.add_argument(
        '--inner-iters', help='inner iterations', default=50, type=int)
    parser.add_argument(
        '--replacement', help='sample with replacement', action='store_true')
    parser.add_argument('--learning-rate',
                        help='Inner loop step size', default=1e-3, type=float)
    parser.add_argument(
        '--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-step-final', help='meta-training step size by the end',
                        default=0.1, type=float)
    parser.add_argument(
        '--meta-batch', help='meta-training batch size', default=8, type=int)
    parser.add_argument(
        '--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-interval',
                        help='train steps per eval', default=10, type=int)
    parser.add_argument(
        '--weight-decay', help='weight decay rate', default=1, type=float)
    parser.add_argument(
        '--transductive', help='evaluate all samples at once', action='store_true')
    parser.add_argument(
        '--foml', help='use FOML instead of Reptile', action='store_true')
    parser.add_argument('--foml-tail', help='number of shots for the final mini-batch in FOML',
                        default=None, type=int)
    parser.add_argument(
        '--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pprint(sorted(vars(args).items(), key=lambda x: x[0]))
    ckpt_path = os.path.join(args.basedir, args.expname)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)
    random.seed(args.seed)
    train_set, test_set = read_dataset(args.metadatadir)
    print(f'TRAIN: {len(train_set)} | TEST: {len(test_set)}')

    render_kwargs_train, render_kwargs_test, start, grad_vars, models =\
        create_nerf(args)

    train(models, grad_vars,
          train_set,
          test_set,
          render_kwargs_train,
          render_kwargs_test,
          ckpt_path,
          args.chunk, args.N_rand,
          args.N_importance, args.use_viewdirs,
          args.half_res, args.testskip, args.white_bkgd,
          **train_kwargs(args),
          inner_learning_rate=args.learning_rate)
