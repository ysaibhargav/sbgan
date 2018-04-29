# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug


import tensorflow as tf 
import numpy as np
import os
import pdb
import argparse
from skimage.io import imsave
from collections import namedtuple
from collections import OrderedDict, defaultdict
from tensorflow.examples.tutorials.mnist import input_data

from sbgan import SBGAN
from utils import AttributeDict, read_from_yaml, setup_output_dir, Data
from dcgan_ops import *

fc = tf.contrib.layers.fully_connected
c2d = tf.layers.conv2d
c2d_t = tf.layers.conv2d_transpose
pooling = tf.layers.average_pooling2d
Hook = namedtuple("Hook", ["frequency", "is_joint", "function"])


class Config(object):
    def __init__(self, file, loglevel, args):
        config = read_from_yaml(file)
        for arg in args.__dict__.keys():
            value = getattr(args, arg)
            if value is not None:
                if arg not in config:
                    config[arg] = None
                config[arg] = value
                
        output_dir, config = setup_output_dir(config['output_dir'], config, loglevel)
        for k in config:
            setattr(self, k, config[k])


def hook_arg_filter(*_args):
    def hook_decorator(f):
        def func_wrapper(*args, **kwargs):
            return f(*[kwargs[arg] for arg in _args])
        return func_wrapper
    return hook_decorator


@hook_arg_filter("g_z", "epoch")
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    global out_path 

    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists(os.path.join("out", "b-mnist")):
        os.mkdir(os.path.join("out", "b-mnist"))

    img_height = img_width = 28
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], \
            img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255.
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
        folder_path = os.path.join(config.save_dir, 'b-mnist')
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "%s.png"%str(fname))
    imsave(file_path, img_grid)

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def generator(x, scope="generator", isTrain=True): 
    with tf.variable_scope(scope):
        fc1 = fc(x, 6*6*128, activation_fn=tf.tanh, reuse=tf.AUTO_REUSE, scope='fc1')
        x = tf.reshape(fc1, shape=[-1, 6, 6, 128])
        c1 = c2d_t(x, 64, 4, strides=2, reuse=tf.AUTO_REUSE, name='c1')
        c2 = c2d_t(c1, 1, 2, strides=2, reuse=tf.AUTO_REUSE, name='c2')
        o = tf.tanh(c2, name='output')

    return o

def discriminator(x, scope="discriminator", isTrain=True): 
    with tf.variable_scope(scope):
        c1 = c2d(x, 64, 5, reuse=tf.AUTO_REUSE, name='c1') 
        c1_a = tf.tanh(c1, name='c1_a')
        c1_p = pooling(c1_a, 2, 2, name='c1_p')
        c2 = c2d(x, 128, 5, reuse=tf.AUTO_REUSE, name='c2') 
        c2_a = tf.tanh(c2, name='c2_a')
        c2_p = pooling(c2_a, 2, 2, name='c2_p')
        flat = tf.contrib.layers.flatten(c2_p)
        fc1 = fc(flat, 1024, activation_fn=tf.tanh, reuse=tf.AUTO_REUSE, scope='fc1')
        logits = fc(fc1, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope='fc2')

    return logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='output_dir',
                        type=str, default='out',
                        help="Path to store the results.")
    parser.add_argument('-l', '--log', action="store", dest="loglevel", type = str, 
            default="DEBUG", help = "Logging Level")
    parser.add_argument('--z-dims', dest='z_dims', type=int, 
            help="Dimensionality of latent space.")
    parser.add_argument('--ng', dest='n_g', type=int, 
            help="Number of generator particles to use.")
    parser.add_argument('--nd', dest='n_d', type=int, 
            help="Number of discriminator particles to use.")
    parser.add_argument('-cf', '--config_file',dest='config_file', type=str)
    parser.set_defaults(render=False)

    return parser.parse_args()


gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

args = parse_args()
config = Config(args.config_file, args.loglevel, args)
out_path = args.output_dir

if not os.path.exists(out_path):
    os.makedirs(out_path)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
data = {'train': {'x': mnist.train.images, 'y': mnist.train.labels},
        'test': {'x': mnist.test.images, 'y': mnist.test.labels}}
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
train_x_op = tf.image.resize_images(mnist.train.images, [28, 28])
train_x = sess.run(train_x_op)
data = {'train': {'x': train_x}} 
"""
data = Data(data, num_classes=10)
data.build_graph(config, shape=[28, 28, 1])

hook1 = Hook(1, False, show_result)

m = SBGAN(generator, discriminator, n_g=config.n_g)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, config, data, summary=False, hooks = [hook1])
