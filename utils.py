import yaml
import yaml
import os
import time
import logging
from subprocess import Popen, PIPE
import sys


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