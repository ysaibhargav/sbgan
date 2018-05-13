from utils import read_from_yaml
import os
import argparse
import itertools
import sys


def parse_arguments():
	parser = argparse.ArgumentParser(description='SBGAN Argument Parser')
	parser.add_argument('-cf', '--config_file',dest='config_file', type=str, default = 'reinforce_configs/lunar1.yaml')
	parser.add_argument('-of', '--output_dir', dest='output_dir', type=str, default= 'hyper_configs')
	return parser.parse_args()

def main(args):
	args = parse_arguments()
	config = read_from_yaml(args.config_file)
	hyper_params = {'step_size': ["0.00001", "0.000001", "0.000005"]}
	hyper_params = [[(key, val) for val in hyper_params[key]] for key in hyper_params]
	iterator = itertools.product(*hyper_params)
	i = -1
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	for l in iterator:
		i += 1
		for (x, y) in l:
			config[x] = y
		with open(os.path.join(args.output_dir, 'config%d.yaml' % i), 'w') as f:
			for k in config:
				f.write('%s: %s\n'% (k, config[k]))

if __name__ == '__main__':
	main(sys.argv)
