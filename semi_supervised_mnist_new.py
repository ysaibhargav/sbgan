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
from utils import AttributeDict, read_from_yaml, setup_output_dir, Data
from sbgan import SBGAN
import pdb
fc = tf.contrib.layers.fully_connected
relu = tf.nn.relu
Hook = namedtuple("Hook", ["frequency", "is_joint", "function"])
import logging
from tensorflow.examples.tutorials.mnist import input_data
import skimage.transform

config = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='output_dir',
                        type=str,
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
    parser.add_argument('-arch', '--architecture', dest='arch', type=str, help='mlp or dcgan architecture')
    parser.set_defaults(render=False)
    return parser.parse_args()

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
        logger = logging.getLogger()
        logger.info(self.__dict__)

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
    if not os.path.exists(os.path.join("out", "cifar")):
        os.mkdir(os.path.join("out", "cifar"))

    img_height = img_width = 32
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], \
                img_height, img_width, 1)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, 1), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255.
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w, :] = img
        folder_path = os.path.join(config.save_dir, 'cifar')
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "%s.png"%str(fname))
    imsave(file_path, img_grid)


def MLPgenerator(z, scope="generator"):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE): 
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(z, 150, scope = "h1")
            h2 = fc(h1, 300, scope = "h2")
            h3 = fc(h2, 784, activation_fn = None, scope = "h3")
        o = tf.nn.tanh(h3)
            
        return o

def MLPdiscriminator(z, scope='discriminator'):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE):
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(z, 100, scope = "h1")
            h2 = fc(h1, 1000, scope = "h2")
            num_outputs = 11 if config.exp == 'semisupervised' else 1
            h3 = fc(h2, num_outputs, activation_fn = None, scope = "h3")

        return h3


#class and helper functions for DCGAN
def DCGANgenerator(x, scope="generator", isTrain=True): 
    with tf.variable_scope(scope):
        linear1 = fc(x, 2048, activation_fn=None, scope='l1', reuse=tf.AUTO_REUSE)
        linear1 = relu(tf.layers.batch_normalization(linear1, training=isTrain, reuse=tf.AUTO_REUSE, name='bn0')) 
        x = tf.reshape(linear1, [-1, 2, 2, 512])
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 384, [3, 3], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c1')
        relu1 = relu(tf.layers.batch_normalization(conv1, training=isTrain, reuse=tf.AUTO_REUSE, name='bn1'))
        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(relu1, 192, [3, 3], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c2')
        relu2 = relu(tf.layers.batch_normalization(conv2, training=isTrain, reuse=tf.AUTO_REUSE, name='bn2'))
        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(relu2, 96, [3, 3], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c3')
        relu3 = relu(tf.layers.batch_normalization(conv3, training=isTrain, reuse=tf.AUTO_REUSE, name='bn3'))
        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(relu3, 1, [5, 5], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c4')
        # output layer
        o = tf.nn.tanh(conv4)
        #import pdb; pdb.set_trace()
        return o

# D(x)
def DCGANdiscriminator(x, scope="discriminator", isTrain=True): 
    #x = tf.expand_dims(x, -1)
    with tf.variable_scope(scope):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 96, [5, 5], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c1')
        lrelu1 = lrelu(conv1, 0.2)
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 192, [3, 3], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain, reuse=tf.AUTO_REUSE, name='bn2'), 0.2)
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 384, [3, 3], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain, reuse=tf.AUTO_REUSE, name='bn3'), 0.2)
        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 512, [3, 3], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c4')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain, reuse=tf.AUTO_REUSE, name='bn4'), 0.2)
        # output layer
        
        linear1 = lrelu(fc(tf.reshape(lrelu4, [-1, 2048]), 512, activation_fn=None, scope='l1', reuse=tf.AUTO_REUSE))
        
        linear2 = fc(linear1, 11, activation_fn=None, scope='l2', reuse=tf.AUTO_REUSE)
        
        return linear2


args = parse_args()
config = Config(args.config_file, args.loglevel, args)
hook1 = Hook(1, False, show_result)

sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True)))
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

train_x_op = tf.image.resize_images(mnist.train.images, [32, 32])
train_x_op = (train_x_op - 0.5) / 0.5
test_x_op = tf.image.resize_images(mnist.test.images, [32, 32])
test_x_op = (test_x_op - 0.5) / 0.5
train_x, test_x = sess.run([train_x_op, test_x_op])

train_y = mnist.train.labels
test_y = mnist.test.labels
data = {'train': {'x': train_x.astype(np.float32), 'y': train_y.astype(np.float32)}, 
        'test': {'x': test_x.astype(np.float32), 'y': test_y.astype(np.float32)}}

data = Data(data, num_classes=10)
data.build_graph(config, shape=[32, 32, 1])
if not hasattr(config, 'arch'):
    setattr(config, 'arch', 'mlp')

if config.arch == 'dcgan':
    m = SBGAN(DCGANgenerator, DCGANdiscriminator, n_g = config.n_g, n_d = config.n_d)
else:
    m = SBGAN(MLPgenerator, MLPdiscriminator, n_g = config.n_g, n_d=config.n_d)
m.train(sess, config, data, summary=config.summary, hooks = [hook1])


