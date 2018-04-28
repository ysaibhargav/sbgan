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

from sbgan import SBGAN
from utils import AttributeDict, read_from_yaml, setup_output_dir, Data
from dcgan_ops import *

fc = tf.contrib.layers.fully_connected
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

def DCGANdiscriminator(z, scope='discriminator', train=True):

    K = 1
    # z: [?, 28, 28, 1]
    #pdb.set_trace()
    disc_strides = [2, 2, 2, 2]
    disc_kernel_sizes = [5, 3, 3, 3, 3]
    df_dim = 96
    output_kernels = [96, 192, 384, 512]
    z = tf.reshape(z, [-1, 28, 28, 1])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        d_batch_norm = AttributeDict([("d_bn%i" % dbn_i, batch_norm(name='d_bn%i' % dbn_i)) for dbn_i in range(len(disc_strides))])
        h = z
        for layer in range(len(disc_strides)):
            if layer == 0:
                h = lrelu(conv2d(h, output_kernels[layer], name='d_h%i_conv' % layer, \
                            k_h=disc_kernel_sizes[layer], k_w=disc_kernel_sizes[layer], \
                            d_h=disc_strides[layer], d_w=disc_strides[layer],
                            ))
            else:
                h = lrelu(d_batch_norm["d_bn%i" % layer](conv2d(h, output_kernels[layer], \
                                                            name='d_h%i_conv' % layer, k_h=disc_kernel_sizes[layer], k_w=disc_kernel_sizes[layer], \
                                                            d_h=disc_strides[layer], d_w=disc_strides[layer]), train=train))

        #h_end = lrelu(linear(tf.reshape(h, [batch_size, -1]), df_dim*4, "d_h_end_lin")) # for feature norm
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        h_end = lrelu(linear(tf.contrib.layers.flatten(h), df_dim*4, "d_h_end_lin"))    
        h_out = linear(h_end, K, 'd_h_out_lin')
    return h_out


def DCGANgenerator(z, scope='generator'):
    #z: [?, 100]
    gen_strides = [2, 2, 2, 2]
    g_kernel_dim = [5, 3, 3, 3, 3]
    g_w_dim = OrderedDict([('g_h4_W', (5, 5, 1, 96)), ('g_h3_W', (3, 3, 96, 192)), ('g_h2_W', (3, 3, 192, 384)), \
                                    ('g_h1_W', (3, 3, 384, 512)), ('g_h0_lin_W', (100, 2048))])
    batch_size = config.z_batch_size
    g_out_dim = OrderedDict([('g_h4', (28, 28)), ('g_h3', (14, 14)), ('g_h2', (7, 7)), ('g_h1', (4, 4)), ('g_h0', (2, 2))])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        g_batch_norm = AttributeDict([("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) for gbn_i in range(len(gen_strides))])
        h = linear(z, g_w_dim["g_h0_lin_W"][-1], 'g_h0_lin')
        h = tf.nn.relu(g_batch_norm.g_bn0(h))
        h = tf.reshape(h, [batch_size, g_out_dim["g_h0"][0], g_out_dim["g_h0"][1], -1])

        for layer in range(1, len(gen_strides)+1):
            out_shape = [batch_size, g_out_dim["g_h%i" % layer][0],
                         g_out_dim["g_h%i" % layer][1], g_w_dim["g_h%i_W" % layer][-2]]

            h = deconv2d(h,
                         out_shape,
                         k_h=g_kernel_dim[layer-1], k_w=g_kernel_dim[layer-1],
                         d_h=gen_strides[layer-1], d_w=gen_strides[layer-1],
                         name='g_h%i' % layer)
            if layer < len(gen_strides):
                h = tf.nn.relu(g_batch_norm["g_bn%i" % layer](h))
    return tf.nn.tanh(h) 



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

data = Data(data, num_classes=10)
data.build_graph(config)

hook1 = Hook(1, False, show_result)

m = SBGAN(DCGANgenerator, DCGANdiscriminator, n_g=config.n_g)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, config, data, summary=False, hooks = [hook1])
