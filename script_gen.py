import os
import glob
import argparse
import sys


def parse_arguments():
    parse = argparse.ArgumentParser(description='Script generator')
    parser.add_argument('-cdir', '--config_dir', dest='config_dir', type=str,
            default='config')
    parser.add_argument('-ldir', '--log_dir', dest='log_dir', type=str, default='logs')
    parser.add_argument('-sdir', '--scripts_dir', dest='scripts_dir', type=str,
            default='scripts')
    parser.add_argument('-gpus', '--gpus', dest='gpus', type=str, default='0')
    parser.add_argument('-py', '--py', dest='py', type=str, default='test.py')
    parser.add_argument('-n', '--num_threads', dest='n', type=int, default=1)

    return parser.parse_args()

def main(args):
    args = parse_arguments()
    config_files = glob.glob(os.path.join(args.config_dir, 'config*.yaml'))
    gpus = args.gpus.split(",")
    num_scripts = args.n * len(gpus)
    num_files_per_script = len(config_files) // num_scripts

    for i in range(num_scripts):
        file_list = config_files[i*num_scripts: min(len(config_files), (i+1)*num_scripts)]
        gpu = gpus[i // len(gpus)]

        with open(os.path.join(args.sdir, 'script_%d.sh'%i), 'w') as f:
            #f.write('#!/bin/bash\n\n')
            f.write('export CUDA_VISIBLE_DEVICES=%d;\n\n'%(int(gpu)))

            for _file in file_list:
                cidx = [s for s in _file.split(os.path.sep)[-1] if s.isdigit()].join("")
                f.write('python3 %s -cf %s &> %s\n'%(args.py, _file,
                    os.path.join(args.ldir, 'out_%s.txt'%cidx)))
                f.write('sleep 5\n\n')

    with open(os.path.join(args.sdir, 'run.sh'), 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(['sh %s'%(os.path.join("script_%d.sh"%i)) for i in
            range(num_scripts)].join('\n\n'))


if __name__ == '__main__':
    main(sys.argv)
