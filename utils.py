import yaml
import yaml
import os
import time
import logging
from subprocess import Popen, PIPE
import sys
import tensorflow as tf
import numpy as np
import pdb


class Data(object):
    def __init__(self, data, num_classes=None):
        self._x_train = None
        self._xs_train = None
        self._ys_train = None
        self._xs_test = None
        self._ys_test = None
        self._data = data
        self.n_classes = num_classes

    def build_graph(self, config, shape=None):
        '''
        Modify this function according to the dataset.
        Builds the computation graph for the data
        '''

        _x_train = self._data['train']['x']
        idx = np.random.choice(_x_train.shape[0], size=config.n_supervised, replace=False)
        _xs_train = _x_train[idx]

        keep_idx = list(set(range(_x_train.shape[0])) - set(idx))
        _x_train = _x_train[keep_idx]

        round_sz = config.x_batch_size*(_x_train.shape[0]//config.x_batch_size)
        _x_train = _x_train[:round_sz]

        dataset = tf.data.Dataset.from_tensor_slices(_x_train)
        if shape is not None:
            dataset = dataset.map(lambda x: tf.reshape(x, shape))
        dataset = dataset.shuffle(buffer_size=55000).batch(config.x_batch_size)
        self.unsupervised_iterator = dataset.make_initializable_iterator()
        self.x = [self.unsupervised_iterator.get_next() for _ in range(config.n_d)]
        self.z = tf.random_normal([2, config.n_g, config.z_batch_size, config.z_dims], 
                stddev = config.z_std)

        if config.exp == 'semisupervised':
            _y_train = self._data['train']['y']
            _ys_train = _y_train[idx]

            dataset = tf.data.Dataset.from_tensor_slices((_xs_train, _ys_train))
            dataset = dataset.batch(config.n_supervised)
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(indices = y, 
                depth = self.n_classes)))
            self.supervised_iterator = dataset.make_initializable_iterator()
            self.xs, self.ys = self.supervised_iterator.get_next()
            
            dataset = tf.data.Dataset.from_tensor_slices((self._data['test']['x'], 
                self._data['test']['y']))
            dataset = dataset.batch(config.test_batch_size)
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(indices = y, 
                depth = self.n_classes)))
            self.test_iterator = dataset.make_initializable_iterator()
            self.x_test, self.y_test = self.test_iterator.get_next()



def read_from_yaml(filepath):
    with open(filepath, 'r') as fd:
        data = yaml.load(fd)
    return data

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def make_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_directory(dirname):
    """
    Makes a directory if it does not exist
    """
    try:
        os.makedirs(dirname)
    except OSError:
        if not os.path.isdir(dirname):
            raise

def write_to_yaml(filepath, data):
    with open(filepath, 'w') as fd:
        yaml.dump(data=data, stream=fd, default_flow_style=False)


def setup_output_dir(output_dir, config, loglevel):
    """
        Takes in the output_dir. Note that the output_dir stores each run as run-1, ....
        Makes the next run directory. This also sets up the logger
        A run directory has the following structure
        run-1:
            |_ best_model
                     |_ model_params_and_metrics.tar.gz
                     |_ validation paths.txt
            |_ last_model_params_and_metrics.tar.gz
            |_ config.yaml
            |_ githash.log of current run
            |_ gitdiff.log of current run
            |_ logfile.log (the log of the current run)
        This also changes the config, to add the save directory
    """
    make_directory(output_dir)
    last_run = -1
    for dirname in os.listdir(output_dir):
        if dirname.startswith('run-'):
            last_run = max(last_run, int(dirname.split('-')[1]))
    new_dirname = os.path.join(output_dir, 'run-%d' % (last_run + 1))
    make_directory(new_dirname)
    best_model_dirname = os.path.join(new_dirname, 'best_model')
    make_directory(best_model_dirname)
    config_file = os.path.join(new_dirname, 'config.yaml')
    config['save_dir'] = new_dirname
    write_to_yaml(config_file, config)
    # Save the git hash
    process = Popen('git log -1 --format="%H"'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('ascii').strip('\n').strip('"')
    with open(os.path.join(new_dirname, "githash.log"), "w") as fp:
        fp.write(stdout)
    # Save the git diff
    process = Popen('git diff'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    with open(os.path.join(new_dirname, "gitdiff.log"), "w") as fp:
        stdout = stdout.decode('ascii')
        fp.write(stdout)
    # Set up the logger
    numeric_level = getattr(logging, loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logfile = os.path.join(new_dirname, 'logfile.log')
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=numeric_level, stream=sys.stdout)
    fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    logfile_handle = logging.FileHandler(logfile, 'w')
    logfile_handle.setFormatter(fmt)
    logger.addHandler(logfile_handle)
    return new_dirname, config
