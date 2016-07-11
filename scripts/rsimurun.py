from __future__ import print_function, division

import argparse
import sys
import os

from rsimusim.simulation import RollingShutterImuSimulation

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('out')
parser.add_argument('--dataset-dir', default=None)
args = parser.parse_args()

if os.path.exists(args.out):
    print('Outfile {} already exists'.format(args.out))
    sys.exit(-1)

simulator = RollingShutterImuSimulation.from_config(args.config, datasetdir=args.dataset_dir)
result = simulator.run()
result.save(args.out)