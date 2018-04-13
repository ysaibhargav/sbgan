# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug

import argparse
import tensorflow as tf 
import numpy as np
import os
import pdb
from skimage.io import imsave
from collections import namedtuple
from collections import OrderedDict, defaultdict
from dcgan_ops import *
from utils import AttributeDict, read_from_yaml, setup_output_dir
from sbgan import SBGAN
import pdb
fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["frequency", "is_joint", "function"])

config = None

def parse_arguments():
    parser = argparse.ArgumentParser(description='SBGAN Argument Parser')
    parser.add_argument('-cf', '--config_file',dest='config_file', type=str)
    parser.add_argument('-l', '--log', action="store", dest="loglevel", type = str, default="DEBUG", help = "Logging Level")
    return parser.parse_args()

class Config(object):
    def __init__(self, file, loglevel):
        config = read_from_yaml(file)
        output_dir, config = setup_output_dir(config['output_dir'], config, loglevel)   
        for k in config:
            setattr(self, k, config[k])
        setattr(self, 'output_dir', output_dir) 

class Data(object):
    def __init__(self):
        self._x_train = None
        self._xs_train = None
        self._ys_train = None
        self._xs_test = None
        self._ys_test = None

    def build_graph(self, config):
        '''
        Modify this function according to the dataset.
        Builds the computation graph for the data
        '''
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        _x_train = mnist.train.images[:1000]
        idx = np.random.choice(_x_train.shape[0], size=config.n_supervised, replace=False)

        _xs_train = _x_train[idx]
        _ys_train = mnist.train.labels[idx]

        #TODO: remove overlap between supervised and semi-supervised
        dataset = tf.data.Dataset.from_tensor_slices(_x_train)
        dataset = dataset.shuffle(buffer_size=55000).batch(config.x_batch_size)
        self.unsupervised_iterator = dataset.make_initializable_iterator()
        self.x = [self.unsupervised_iterator.get_next() for _ in range(config.n_d)]
        self.z = tf.random_normal([2, config.n_g, config.z_batch_size, config.z_dims], stddev = config.z_std)
        if config.exp == 'semisupervised':
            self.n_classes = 10
            dataset = tf.data.Dataset.from_tensor_slices((_xs_train, _ys_train))
            dataset = dataset.batch(config.n_supervised)
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(indices = y, depth = 10)))
            self.supervised_iterator = dataset.make_initializable_iterator()
            self.xs, self.ys = self.supervised_iterator.get_next()
            
            if hasattr(self, 'test_num'):
                test_len = mnist.test.images.size[0]
                idx = np.random.choice(a=range(test_len), size=self.test_num)
                mnist.test.images = mnist.test.images[idx]
                mnist.test.labels = mnist.test.labels[idx]

            dataset = tf.data.Dataset.from_tensor_slices((mnist.test.images, mnist.test.labels))
            dataset = dataset.batch(config.test_batch_size)
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(indices = y, depth = 10)))
            self.test_iterator = dataset.make_initializable_iterator()
            self.x_test, self.y_test = self.test_iterator.get_next()

def hook_arg_filter(*_args):
    def hook_decorator(f):
        def func_wrapper(*args, **kwargs):
            return f(*[kwargs[arg] for arg in _args])
        return func_wrapper
    return hook_decorator


@hook_arg_filter("g_z", "epoch")
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
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
        folder_path = os.path.join(config.output_dir, 'b-mnist')
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "%s.png"%str(fname))
    #pdb.set_trace()
    imsave(file_path, img_grid)


def generator(z, scope="generator"):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE): 
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(z, 150, scope = "h1")
        
            h2 = fc(h1, 300, scope = "h2")
            h3 = fc(h2, 784, activation_fn = None, scope = "h3")
        o = tf.nn.tanh(h3)
            
        return o

def discriminator(z, scope='discriminator'):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE):
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(z, 100, scope = "h1")
            h2 = fc(h1, 1000, scope = "h2")
            num_outputs = 11 if config.exp == 'semisupervised' else 1
            h3 = fc(h2, num_outputs, activation_fn = None, scope = "h3")

        return h3

#class and helper functions for DCGAN
def DCGANdiscriminator(z, scope='discriminator', train=True):

    K = 11
    # z: [?, 28, 28, 1]
    #pdb.set_trace()
    disc_strides = [2, 2, 2, 2]
    disc_kernel_sizes = [5, 3, 3, 3, 3]
    batch_size = z.get_shape()[0]
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

args = parse_arguments()
config = Config(args.config_file, args.loglevel)
hook1 = Hook(1, False, show_result)

data = Data()
data.build_graph(config)
m = SBGAN(DCGANgenerator, DCGANdiscriminator, n_g = config.n_g, n_d = config.n_d)

sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True)))
m.train(sess, config, data, summary=False, hooks = [hook1])


