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
#from dcgan_ops import *
#from bgan_util import AttributeDict

from sbgan import SBGAN

fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["frequency", "is_joint", "function"])

config = None

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
'''
class DCGAN(object):
	def __init__(self, x_dim, z_dim, dataset_size, batch_size=64, gf_dim=64, df_dim=64, num_layers=4):
		assert len(x_dim) == 3, "invalid image dims"
		c_dim = x_dim[2]
		self.is_grayscale = (c_dim == 1)
		self.dataset_size = dataset_size
		self.batch_size = batch_size
		
		self.K = 2 # fake and real classes
		self.x_dim = x_dim
		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim
		self.c_dim = c_dim

		def get_strides(num_layers, num_pool):
			interval = int(math.floor(num_layers/float(num_pool)))
			strides = np.array([1]*num_layers)
			strides[0:interval*num_pool:interval] = 2
			return strides
		
		self.num_pool = 4
		self.max_num_dfs = 512
		self.gen_strides = get_strides(num_layers, self.num_pool)
		self.disc_strides = self.gen_strides
		num_dfs = np.cumprod(np.array([self.df_dim] + list(self.disc_strides)))[:-1]
		num_dfs[num_dfs >= self.max_num_dfs] = self.max_num_dfs # memory
		self.num_dfs = list(num_dfs)
		self.num_gfs = self.num_dfs[::-1]

		self.construct_from_hypers(gen_strides=self.gen_strides, disc_strides=self.disc_strides,
								   num_gfs=self.num_gfs, num_dfs=self.num_dfs)
		
		self.build_bgan_graph()

	def construct_from_hypers(self, gen_kernel_size=5, gen_strides=[2,2,2,2],
							  disc_kernel_size=5, disc_strides=[2,2,2,2],
							  num_dfs=None, num_gfs=None):

		
		self.d_batch_norm = AttributeDict([("d_bn%i" % dbn_i, batch_norm(name='d_bn%i' % dbn_i)) for dbn_i in range(len(disc_strides))])
		self.sup_d_batch_norm = AttributeDict([("sd_bn%i" % dbn_i, batch_norm(name='sup_d_bn%i' % dbn_i)) for dbn_i in range(5)])
		self.g_batch_norm = AttributeDict([("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) for gbn_i in range(len(gen_strides))])

		if num_dfs is None:
			num_dfs = [self.df_dim, self.df_dim*2, self.df_dim*4, self.df_dim*8]
			
		if num_gfs is None:
			num_gfs = [self.gf_dim*8, self.gf_dim*4, self.gf_dim*2, self.gf_dim]

		assert len(gen_strides) == len(num_gfs), "invalid hypers!"
		assert len(disc_strides) == len(num_dfs), "invalid hypers!"

		s_h, s_w = self.x_dim[0], self.x_dim[1]
		ks = gen_kernel_size
		self.gen_output_dims = OrderedDict()
		self.gen_weight_dims = OrderedDict()
		num_gfs = num_gfs + [self.c_dim]
		self.gen_kernel_sizes = [ks]
		for layer in range(len(gen_strides))[::-1]:
			self.gen_output_dims["g_h%i_out" % (layer+1)] = (s_h, s_w)
			assert gen_strides[layer] <= 2, "invalid stride"
			assert ks % 2 == 1, "invalid kernel size"
			self.gen_weight_dims["g_h%i_W" % (layer+1)] = (ks, ks, num_gfs[layer+1], num_gfs[layer])
			self.gen_weight_dims["g_h%i_b" % (layer+1)] = (num_gfs[layer+1],)
			s_h, s_w = conv_out_size(s_h, gen_strides[layer]), conv_out_size(s_w, gen_strides[layer])
			ks = kernel_sizer(ks, gen_strides[layer])
			self.gen_kernel_sizes.append(ks)
		self.gen_weight_dims.update(OrderedDict([("g_h0_lin_W", (self.z_dim, num_gfs[0] * s_h * s_w)),
												 ("g_h0_lin_b", (num_gfs[0] * s_h * s_w,))]))
		self.gen_output_dims["g_h0_out"] = (s_h, s_w)

		self.disc_weight_dims = OrderedDict()
		s_h, s_w = self.x_dim[0], self.x_dim[1]
		num_dfs = [self.c_dim] + num_dfs
		ks = disc_kernel_size
		self.disc_kernel_sizes = [ks]
		for layer in range(len(disc_strides)):
			assert disc_strides[layer] <= 2, "invalid stride"
			assert ks % 2 == 1, "invalid kernel size"
			self.disc_weight_dims["d_h%i_W" % layer] = (ks, ks, num_dfs[layer], num_dfs[layer+1])
			self.disc_weight_dims["d_h%i_b" % layer] = (num_dfs[layer+1],)
			s_h, s_w = conv_out_size(s_h, disc_strides[layer]), conv_out_size(s_w, disc_strides[layer])
			ks = kernel_sizer(ks, disc_strides[layer])
			self.disc_kernel_sizes.append(ks)

		self.disc_weight_dims.update(OrderedDict([("d_h_end_lin_W", (num_dfs[-1] * s_h * s_w, num_dfs[-1])),
												  ("d_h_end_lin_b", (num_dfs[-1],)),
												  ("d_h_out_lin_W", (num_dfs[-1], self.K)),
												  ("d_h_out_lin_b", (self.K,))]))


		for k, v in self.gen_output_dims.items():
			print "%s: %s" % (k, v)
		print '****'
		for k, v in self.gen_weight_dims.items():
			print "%s: %s" % (k, v)
		print '****'
		for k, v in self.disc_weight_dims.items():
			print "%s: %s" % (k, v)
	

	def initialize_wgts(self, scope_str):

		if scope_str == "generator":
			weight_dims = self.gen_weight_dims
			numz = self.num_gen
		elif scope_str == "discriminator":
			weight_dims = self.disc_weight_dims
			numz = self.num_disc
		else:
			raise RuntimeError("invalid scope!")

		param_list = []
		with tf.variable_scope(scope_str) as scope:
			for zi in xrange(numz):
				for m in xrange(self.num_mcmc):
					wgts_ = AttributeDict()
					for name, shape in weight_dims.iteritems():
						wgts_[name] = tf.get_variable("%s_%04d_%04d" % (name, zi, m),
													  shape, initializer=tf.random_normal_initializer(stddev=0.02))
					param_list.append(wgts_)
			return param_list

	def generator(self, z, gen_params):

		with tf.variable_scope("generator") as scope:

			h = linear(z, self.gen_weight_dims["g_h0_lin_W"][-1], 'g_h0_lin',
					   matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b)
			h = tf.nn.relu(self.g_batch_norm.g_bn0(h))

			h = tf.reshape(h, [self.batch_size, self.gen_output_dims["g_h0_out"][0],
							   self.gen_output_dims["g_h0_out"][1], -1])

			for layer in range(1, len(self.gen_strides)+1):

				out_shape = [self.batch_size, self.gen_output_dims["g_h%i_out" % layer][0],
							 self.gen_output_dims["g_h%i_out" % layer][1], self.gen_weight_dims["g_h%i_W" % layer][-2]]

				h = deconv2d(h,
							 out_shape,
							 k_h=self.gen_kernel_sizes[layer-1], k_w=self.gen_kernel_sizes[layer-1],
							 d_h=self.gen_strides[layer-1], d_w=self.gen_strides[layer-1],
							 name='g_h%i' % layer,
							 w=gen_params["g_h%i_W" % layer], biases=gen_params["g_h%i_b" % layer])
				if layer < len(self.gen_strides):
					h = tf.nn.relu(self.g_batch_norm["g_bn%i" % layer](h))

			return tf.nn.tanh(h)

	def discriminator(self, image, K, disc_params, train=True):

		with tf.variable_scope("discriminator") as scope:

			h = image
			for layer in range(len(self.disc_strides)):
				if layer == 0:
					h = lrelu(conv2d(h,
									 self.disc_weight_dims["d_h%i_W" % layer][-1],
									 name='d_h%i_conv' % layer,
									 k_h=self.disc_kernel_sizes[layer], k_w=self.disc_kernel_sizes[layer],
									 d_h=self.disc_strides[layer], d_w=self.disc_strides[layer],
									 w=disc_params["d_h%i_W" % layer], biases=disc_params["d_h%i_b" % layer]))
				else:
					h = lrelu(self.d_batch_norm["d_bn%i" % layer](conv2d(h,
																		 self.disc_weight_dims["d_h%i_W" % layer][-1],
																		 name='d_h%i_conv' % layer,
																		 k_h=self.disc_kernel_sizes[layer], k_w=self.disc_kernel_sizes[layer],
																		 d_h=self.disc_strides[layer], d_w=self.disc_strides[layer],
																		 w=disc_params["d_h%i_W" % layer], biases=disc_params["d_h%i_b" % layer]), train=train))

			h_end = lrelu(linear(tf.reshape(h, [self.batch_size, -1]),
							  self.df_dim*4, "d_h_end_lin",
							  matrix=disc_params.d_h_end_lin_W, bias=disc_params.d_h_end_lin_b)) # for feature norm
			h_out = linear(h_end, K,
						   'd_h_out_lin',
						   matrix=disc_params.d_h_out_lin_W, bias=disc_params.d_h_out_lin_b)
			
			return tf.nn.softmax(h_out), h_out, [h_end]
'''
	
config = Config()
hook1 = Hook(1, False, show_result)

data = Data()
data.build_graph(config)
m = SBGAN(generator, discriminator, n_g = config.n_g, n_d = config.n_d)

sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, config, data, summary=False, hooks = [hook1])