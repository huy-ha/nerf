import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
from reptile.nerf_reptile import meta_evaluate
from data_utils import read_dataset
from run_nerf import create_nerf
from run_nerf import config_parser
import numpy as np
from tensorboardX import SummaryWriter
from pprint import pprint



def parse_args():
    parser = config_parser()
    parser.add_argument('--metadatadir', type=str,
                        default='data/nerf_synthetic/metacubes/',
                        help="path of directory containing all scene " +
                        "directories, each containing posed RGB images ")
    parser.add_argument('--seed', help='random seed', default=0, type=int)
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
                        help='Adam step size', default=1e-3, type=float)
    parser.add_argument(
        '--meta-step', help='meta-training step size', default=0.05, type=float)
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
    ckpt_path = os.path.join(args.basedir, args.expname)
    assert os.path.exists(ckpt_path)
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)
        random.seed(args.seed)
    # TODO seed everything
    train_set, test_set = read_dataset(args.metadatadir)

    render_kwargs_train, render_kwargs_test, start, grad_vars, models =\
        create_nerf(args)

    #
    optimizer = tf.keras.optimizers.Adam(
        args.learning_rate, beta_1=0)
    test_writer = SummaryWriter(ckpt_path + '/test')
    loss_dict = meta_evaluate(
        models,
        metalearning_iter=start,
        test_scenes=test_set,
        N_importance=args.N_importance,
        half_res=args.half_res,
        testskip=args.testskip,
        white_bkgd=args.white_bkgd,
        log_fn=print,
        save_dir=ckpt_path+ '/test',
        N_rand=args.N_rand,
        inner_iters=args.inner_iters,
        chunk=args.chunk,
        use_viewdirs=args.use_viewdirs,
        grad_vars=grad_vars,
        optimizer=optimizer,
        render_kwargs_train=render_kwargs_train,
        render_kwargs_test=render_kwargs_test,
        render_test_set=False,
        writer=test_writer)
    pprint(loss_dict)
