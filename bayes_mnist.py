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
from collections import OrderedDict, defaultdict
from dcgan_ops import *
#from bgan_util import AttributeDict

from sbgan import SBGAN

fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["frequency", "is_joint", "function"])

config = None

class AttributeDict(dict):
	def __getattr__(self, attr):
		return self[attr]
	def __setattr__(self, attr, value):
		self[attr] = value
	def __hash__(self):
		return hash(tuple(sorted(self.items())))

class Config(object):
	def __init__(self):
		self.x_batch_size = 200
		self.z_batch_size = 200
		self.z_dims = 100
		self.z_std = 1
		self.num_epochs = 100 
		self.prior_std = 1
		self.step_size = 1e-3 
		self.prior = 'normal'
		self.summary_savedir = 'summary'
		self.summary_n = 20
		self.exp = 'semisupervised'
		self.n_supervised = 100
		self.n_g = 5
		self.n_d = 1
		self.test_batch_size = 200
		

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
	disc_strides = [2, 2, 2, 2]
	disc_kernel_sizes = [5, 3, 3, 3, 3]
	batch_size = config.x_batch_size
	df_dim = 96
	output_kernels = [96, 192, 384, 512]

	with tf.variable_scope(scope) as scope:
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

		h_end = lrelu(linear(tf.reshape(h, [batch_size, -1]), df_dim*4, "d_h_end_lin")) # for feature norm
		h_out = linear(h_end, K, 'd_h_out_lin')
	return h_out


	def DCGANgenerator(z, scope='generator'):
		#z: [?, 100]
		gen_strides = [2, 2, 2, 2]
		gen_kernel_sizes = [5, 3, 3, 3, 3]
		gen_weight_dims = OrderedDict([('g_h4_W', (5, 5, 1, 96)), ('g_h3_W', (3, 3, 96, 192)), ('g_h2_W', (3, 3, 192, 384)), \
										('g_h1_W', (3, 3, 384, 512)), ('g_h0_lin_W', (100, 2048))])
		batch_size = config.z_batch_size
		gen_output_dims = OrderedDict([('g_h4_out', (28, 28)), ('g_h3_out', (14, 14)), ('g_h2_out', (7, 7)), ('g_h1_out', (4, 4)), ('g_h0_out', (2, 2))])

		with tf.variable_scope(scope) as scope:
			self.g_batch_norm = AttributeDict([("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) for gbn_i in range(len(gen_strides))])
			h = linear(z, gen_weight_dims["g_h0_lin_W"][-1], 'g_h0_lin')
			h = tf.nn.relu(g_batch_norm.g_bn0(h))
			h = tf.reshape(h, [batch_size, gen_output_dims["g_h0_out"][0], gen_output_dims["g_h0_out"][1], -1])

			for layer in range(1, len(gen_strides)+1):
				out_shape = [batch_size, gen_output_dims["g_h%i_out" % layer][0],
							 gen_output_dims["g_h%i_out" % layer][1], gen_weight_dims["g_h%i_W" % layer][-2]]

				h = deconv2d(h,
							 out_shape,
							 k_h=gen_kernel_sizes[layer-1], k_w=gen_kernel_sizes[layer-1],
							 d_h=gen_strides[layer-1], d_w=gen_strides[layer-1],
							 name='g_h%i' % layer)
				if layer < len(gen_strides):
					h = tf.nn.relu(g_batch_norm["g_bn%i" % layer](h))
		return tf.nn.tanh(h) 

config = Config()
hook1 = Hook(1, False, show_result)

data = Data()
data.build_graph(config)
m = SBGAN(generator, discriminator, n_g = config.n_g, n_d = config.n_d)

sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True)))
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, config, data, summary=False, hooks = [hook1])