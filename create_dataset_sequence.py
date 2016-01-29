from __future__ import print_function
import argparse
import time

from rsimusim.openmvg_io import SfMData
from rsimusim.nvm import NvmModel
from rsimusim.dataset import DatasetBuilder
from rsimusim.misc import CalibratedGyroStream

parser = argparse.ArgumentParser()
parser.add_argument('sequence')
parser.add_argument("landmark_source")
parser.add_argument("output")
args = parser.parse_args()

sfm_source = 'nvm' if args.landmark_source.endswith('.nvm') else 'openmvg'

db = DatasetBuilder()

dataset_root = '/home/hannes/Datasets/gopro-gyro-dataset'
gyro_stream = CalibratedGyroStream.from_directory(dataset_root, args.sequence)
db.add_source_gyro(gyro_stream.data, gyro_stream.timestamps)
db.set_orientation_source('imu')

if sfm_source == 'nvm':
    nvm_model = NvmModel.from_file(args.landmark_source)
    nvm_model = NvmModel.create_autoscaled_walk(nvm_model)
    db.add_source_nvm(nvm_model)
    db.set_position_source('nvm')
    db.set_landmark_source('nvm')
else:
    sfm_data = SfMData.from_json(args.landmark_source, color=True)
    sfm_data = SfMData.create_autoscaled_walk(sfm_data)
    db.add_source_openmvg(sfm_data)
    db.set_landmark_source('openmvg')
    db.set_position_source('openmvg')

ds = db.build()
#out_fname = '/home/hannes/Code/rs-imusim/walk_dataset_color_full.h5'
print('Saving dataset to', args.output)
t0 = time.time()
ds.save(args.output, args.sequence)
elapsed = time.time() - t0
print('Saved {:d} landmarks in {:.1f} seconds'.format(len(ds.landmarks), elapsed))
#ds.visualize()