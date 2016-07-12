from __future__ import print_function, division

import argparse
import logging
import sys
import os

from rsimusim.simulation import RollingShutterImuSimulation

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('out')
parser.add_argument('--dataset-dir', default=None)
parser.add_argument('--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
parser.add_argument('--show-progress', action='store_true')
args = parser.parse_args()

# Setup logging
loglevel = getattr(logging, args.loglevel.upper())
logging.basicConfig(level=loglevel)
logger = logging.getLogger("rsimurun")


if os.path.exists(args.out):
    logger.error('Outfile {} already exists'.format(args.out))
    sys.exit(-1)

logger.info('Simulation configuration: {}'.format(args.config))
if args.dataset_dir is not None:
    logger.info('Dataset directory: {}'.format(args.dataset_dir))
simulator = RollingShutterImuSimulation.from_config(args.config, datasetdir=args.dataset_dir)
logger.info('Used dataset: {}'.format(simulator.config.dataset_path))
result = simulator.run(progress=args.show_progress)
logger.info('Saving results to {}'.format(args.out))
result.save(args.out)
logger.info('All done')