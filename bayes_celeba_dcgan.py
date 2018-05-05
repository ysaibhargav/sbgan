# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug


import tensorflow as tf 
import numpy as np
import os
import pdb
import glob
import argparse
import pickle
import h5py
from skimage.io import imsave
from collections import namedtuple
from collections import OrderedDict, defaultdict

from sbgan import SBGAN
from utils import *#AttributeDict, read_from_yaml, setup_output_dir, Data
from dcgan_ops import *

fc = tf.contrib.layers.fully_connected
c2d = tf.layers.conv2d
c2d_t = tf.layers.conv2d_transpose
bn = tf.layers.batch_normalization
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
    if not os.path.exists(os.path.join("out", "b-celeba")):
        os.mkdir(os.path.join("out", "b-celeba"))

    img_height = img_width = 64
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], \
            img_height, img_width, 3)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255.
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w, :] = img
        folder_path = os.path.join(config.save_dir, 'b-celeba')
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "%s.png"%str(fname))
    imsave(file_path, img_grid)

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def generator(x, scope="generator", isTrain=True): 
    with tf.variable_scope(scope):
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid', reuse=tf.AUTO_REUSE, name='c1')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain, reuse=tf.AUTO_REUSE, name='bn1'), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain, reuse=tf.AUTO_REUSE, name='bn2'), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain, reuse=tf.AUTO_REUSE, name='bn3'), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c4')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain, reuse=tf.AUTO_REUSE, name='bn4'), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 3, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c5')
        o = tf.nn.tanh(conv5)

        return o

# D(x)
def discriminator(x, scope="discriminator", isTrain=True): 
    #x = tf.expand_dims(x, -1)
    with tf.variable_scope(scope):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c1')
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain, reuse=tf.AUTO_REUSE, name='bn2'), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain, reuse=tf.AUTO_REUSE, name='bn3'), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE, name='c4')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain, reuse=tf.AUTO_REUSE, name='bn4'), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid', reuse=tf.AUTO_REUSE, name='c5')
        o = tf.nn.sigmoid(conv5)

        conv5 = tf.squeeze(conv5, [2, 3])

        return conv5
        c1_p = pooling(c1_a, 2, 2, name='c1_p')
        c2 = c2d(c1_p, 128, 5, reuse=tf.AUTO_REUSE, name='c2') 
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

pickled_data_path = os.path.join('data', 'pickled')
pickled_file_path = os.path.join(pickled_data_path, 'celeba.h5')

if not os.path.exists(out_path):
    os.makedirs(out_path)

if not os.path.isfile(pickled_file_path):
    if not os.path.exists(pickled_data_path):
        os.makedirs(pickled_data_path)

    data = glob.glob(os.path.join("data", "celebA", "*jpg"))
    N = 75000 
    images = [get_image(f, input_height=108, input_width=108, resize_height=64, 
        resize_width=64, crop=True, grayscale=False) for f in data[:N]]
    images = np.array(images, dtype=np.float32)
    h5f = h5py.File(pickled_file_path, 'w')
    h5f.create_dataset('dataset_1', data=images)
    h5f.close()
    #pickle.dump(images, open(pickled_file_path, 'wb'))
else:
    h5f = h5py.File(pickled_file_path, 'r')
    images = h5f['dataset_1'][:]
    h5f.close()
    #images = pickle.load(open(pickled_file_path, 'rb'))

data = {'train': {'x': images}} 
data = Data(data)
data.build_graph(config, shape=[64, 64, 3])

hook1 = Hook(1, False, show_result)

m = SBGAN(generator, discriminator, n_g=config.n_g)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, config, data, summary=False, hooks = [hook1])
