import os 
import glob
import numpy as np
import six
import cPickle
import tensorflow as tf
import scipy.io as sio
import pdb
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn import mixture
from collections import namedtuple

from sbgan import SBGAN

fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["frequency", "is_joint", "function"])

class Config(object):
    def __init__(self):
        self.x_batch_size = 256
        self.z_batch_size = 256
        self.x_dims = 100
        self.z_dims = 10
        self.z_std = 1
        self.n_g = 15
        self.num_epochs = 100
        self.prior_std = 1
        self.prior = 'xavier'
        self.step_size = 1e-3
        self.summary_savedir = 'summary'
        self.summary_n = 1

class FigPrinter():
    
    def __init__(self, subplot_args):
        import matplotlib as mpl
        mpl.use('Agg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig, self.ax_arr = plt.subplots(*subplot_args)
        
    def print_to_file(self, file_name, close_on_exit=True):
        import matplotlib as mpl
        mpl.use('Agg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig.savefig(file_name, bbox_inches='tight')
        if close_on_exit:
            plt.close("all")

class SynthDataset():
    
    def __init__(self, x_dim=100, num_clusters=10, seed=1234):
        
        np.random.seed(seed)
        
        self.x_dim = x_dim
        self.N = 10000
        self.true_z_dim = 2
        # generate synthetic data
        self.Xs = []
        for _ in xrange(num_clusters):
            cluster_mean = np.random.randn(self.true_z_dim) * 5 # to make them more spread
            A = np.random.randn(self.x_dim, self.true_z_dim) * 5
            X = np.dot(np.random.randn(self.N / num_clusters, self.true_z_dim) + 
                    cluster_mean,
                       A.T)
            self.Xs.append(X)
        X_raw = np.concatenate(self.Xs)
        self.X = (X_raw - X_raw.mean(0)) / (X_raw.std(0))
        print self.X.shape
        
        
    def next_batch(self, batch_size):

        rand_idx = np.random.choice(range(self.N), size=(batch_size,), replace=False)
        return self.X[rand_idx]

def gmm_ms(X):
    aics = []
    n_components_range = range(1, 20)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GMM(n_components=n_components,
                          covariance_type="full")
        gmm.fit(X)
        aics.append(gmm.aic(X))
    return np.array(aics)

VAR_CACHE = {} 
def analyze_div(X_real, X_sample):
    global VAR_CACHE

    def kl_div(p, q):
        eps = 1e-10
        p_safe = np.copy(p)
        p_safe[p_safe < eps] = eps
        q_safe = np.copy(q)
        q_safe[q_safe < eps] = eps
        return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))

    def js_div(p, q):
        m = (p + q) / 2.
        return (kl_div(p, m) + kl_div(q, m))/2.

    from sklearn.decomposition import PCA
    if 'X_trans_real' in VAR_CACHE:
        X_trans_real = VAR_CACHE['X_trans_real']
    else:
        pca = PCA(n_components=2)
        X_trans_real = pca.fit_transform(X_real)

        def cartesian_prod(x, y):
            return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

        dx = 0.1
        dy = 0.1

        xmin1 = np.min(X_trans_real[:, 0]) - 3.0
        xmax1 = np.max(X_trans_real[:, 0]) + 3.0

        xmin2 = np.min(X_trans_real[:, 1]) - 3.0
        xmax2 = np.max(X_trans_real[:, 1]) + 3.0

        space = cartesian_prod(np.arange(xmin1,xmax1,dx), np.arange(xmin2,xmax2,dy)).T

    from scipy import stats

    real_kde = stats.gaussian_kde(X_trans_real.T)
    real_density = real_kde(space) * dx * dy

    X_trans_fake = pca.transform(X_sample)
    fake_kde = stats.gaussian_kde(X_trans_fake.T)
    fake_density = fake_kde(space) * dx * dy

    return js_div(real_density, fake_density), X_trans_real, X_trans_fake

def hook_arg_filter(*_args):
    def hook_decorator(f):
        def func_wrapper(*args, **kwargs):
            return f(*[kwargs[arg] for arg in _args])
        return func_wrapper
    return hook_decorator

@hook_arg_filter("g_z", "real_data", "epoch")
def show_result(g_z, X_real, epoch):
    X_sample = np.concatenate(g_z)
    aics_fake = gmm_ms(X_sample)
    print "Fake number of clusters (AIC estimate):", aics_fake.argmin()
    dist, X_trans_real, X_trans_fake = analyze_div(X_real, X_sample)
    print "JS div:", dist
    fp = FigPrinter((1,2))
    xmin1 = np.min(X_trans_real[:, 0]) - 1.0
    xmax1 = np.max(X_trans_real[:, 0]) + 1.0
    xmin2 = np.min(X_trans_real[:, 1]) - 1.0
    xmax2 = np.max(X_trans_real[:, 1]) + 1.0
    fp.ax_arr[0].plot(X_trans_real[:, 0], X_trans_real[:, 1], '.r')
    fp.ax_arr[0].set_xlim([xmin1, xmax1]); fp.ax_arr[0].set_ylim([xmin2, xmax2])
    fp.ax_arr[1].plot(X_trans_fake[:, 0], X_trans_fake[:, 1], '.g')
    fp.ax_arr[1].set_xlim([xmin1, xmax1]); fp.ax_arr[1].set_ylim([xmin2, xmax2])
    fp.ax_arr[0].set_aspect('equal', adjustable='box')
    fp.ax_arr[1].set_aspect('equal', adjustable='box')
    fp.ax_arr[1].set_title("Epoch %s" % (epoch))
    fp.print_to_file(os.path.join("pca_distribution_%s.png" % (epoch)))

def generator(z, scope='generator'):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE):
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(z, 10, scope = "h1")
            h2 = fc(h1, 1000, scope = "h2")
            h3 = fc(h2, 100, activation_fn = None, scope = "h3")
        o = h3 

        return h3

def discriminator(z, scope='discriminator'):
    with tf.variable_scope(scope):
        with tf.contrib.framework.arg_scope([fc], reuse=tf.AUTO_REUSE):
                #weights_initializer=tf.random_normal_initializer(0, 1)):
            h1 = fc(z, 100, scope = "h1")
            h2 = fc(h1, 1000, scope = "h2")
            h3 = fc(h2, 1, activation_fn = None, scope = "h3")
        o = h3 

        return h3


config = Config()
real_data = SynthDataset(config.x_dims).X 
real_data = np.array(real_data, dtype='float32')
hook = Hook(1, True, show_result)

m = SBGAN(generator, discriminator, n_g=config.n_g)
sess = tf.Session()
m.train(sess, real_data, config, summary=False, hooks=[hook])
