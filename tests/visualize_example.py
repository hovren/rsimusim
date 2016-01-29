from rsimusim.openmvg_io import SfMData
from rsimusim.nvm import NvmModel
from rsimusim.dataset import DatasetBuilder

from test_dataset import GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES, GYRO_DT, OPENMVG_EXAMPLE, NVM_EXAMPLE

nvm_model = NvmModel.from_file(NVM_EXAMPLE)
nvm_model = NvmModel.create_autoscaled_walk(nvm_model)

db = DatasetBuilder()
db.add_source_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
db.add_source_nvm(nvm_model)
db.set_orientation_source('imu')
db.set_position_source('nvm')
db.set_landmark_source('nvm')

ds = db.build()

ds.visualize()