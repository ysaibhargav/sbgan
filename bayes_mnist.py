# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug


import tensorflow as tf 
import numpy as np
import os
import pdb
from skimage.io import imsave
from collections import namedtuple

from sbgan import SBGAN

fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["frequency", "is_joint", "function"])


class Config(object):
    def __init__(self):
        self.x_batch_size = 256
        self.z_batch_size = 256
        self.z_dims = 100
        self.z_std = 1
        self.num_epochs = 100 
        self.prior_std = 1
        self.step_size = 1e-3 
        self.prior = 'xavier'
        self.summary_savedir = 'summary'
        self.summary_n = 20
        self.exp = 'semisupervised'
        self.n_supervised = 100
        self.n_g = 5
        self.n_d = 1
        self.test_batch_size = 256
        

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
        _x_train = mnist.train.images
        idx = np.random.choice(_x_train.shape[0], size=config.n_supervised, replace=False)
        _xs_train = _x_train[idx]
        _ys_train = mnist.train.labels[idx]

        dataset = tf.data.Dataset.from_tensor_slices(_x_train)
        dataset = dataset.shuffle(buffer_size=55000).batch(config.x_batch_size)
        self.unsupervised_iterator = dataset.make_initializable_iterator()
        self.x = [self.unsupervised_iterator.get_next() for _ in range(config.n_d)]
        self.z = tf.random.normal([2, config.n_g, config.z_batch_size, config.z_dims], stddev = config.z_std)
        if config.exp == 'semisupervised':
            self.n_classes = 10
            dataset = tf.data.Dataset.from_tensor_slices((_xs_train, _ys_train))
            dataset = dataset.batch(config.n_supervised)
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(indices = y, depth = 10)))
            self.supervised_iterator = dataset.make_initializable_iterator()
            self.xs, self.ys = self.supervised_iterator.get_next()
            
            dataset = tf.data.Dataset.from_tensor_slices((mnist.test.images, mnist.test.labels))
            dataset = dataset.batch(config.test_batch_size)
            dataset = dataset.map(lambda x, y: (x, tf.onehot(indices = y, depth = 10)))
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

config = Config()
hook1 = Hook(1, False, show_result)

data = Data()
data.build_graph()
m = SBGAN(generator, discriminator, n_g = config.n_g, n_d = config.n_d)

sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, config, data, summary=True, hooks = None)
