# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug


import tensorflow as tf 
import numpy as np
import os
from skimage.io import imsave
from collections import namedtuple

from sbgan import SBGAN

fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["n", "f"])


class _config(object):
    def __init__(self):
        self.x_batch_size = 256
        self.z_batch_size = 256
        self.z_dims = 100
        self.z_std = 1
        self.num_epochs = 100 
        self.prior_std = 1
        self.step_size = 1e-3 
        self.summary_savedir = 'summary'
        self.summary_n = 1


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
    imsave(os.path.join("out", "b-mnist", "%s.png"%str(fname)), img_grid)


def generator(z, scope="generator"):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE): 
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(z, 150, scope = "h1")
            h2 = fc(h1, 300, scope = "h2")
            h3 = fc(h2, 784, activation_fn = None, scope = "h3")
        o = tf.nn.tanh(h3)
            
        return o

# TODO: dropout
def discriminator(x, scope="discriminator"):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE): 
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(x, 200, scope = "h1")
            h2 = fc(h1, 150, scope = "h2")
            h3 = fc(h2, 1, activation_fn = None, scope = "h3")
        o = tf.nn.sigmoid(h3)

        #return o, h3
        return h3


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
real_data = mnist.train.images

config = _config()
hook1 = Hook(1, show_result)

m = SBGAN(generator, discriminator)
sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, real_data, config, summary=False, hooks = [hook1])
