import os, json

import numpy as np
from crisp.l3g4200d import post_process_L3G4200D_data
import crisp
import crisp.rotations
from rsimusim_legacy.dataset import Dataset
from rsimusim_legacy.world import NvmModel

from imusim.maths.quaternions import QuaternionArray, Quaternion

import matplotlib.pyplot as plt

def load_gyro(filepath):
    return crisp.GyroStream.from_csv(filepath)

def load_params(filepath):
    _, ext = os.path.splitext(filepath)
    print ext
    if ext == '.csv':
        arr = np.loadtxt(filepath, delimiter=',')
        param_names = ('gyro_rate', 'time_offset', 'rot_x', 'rot_y', 'rot_z', 'gbias_x', 'gbias_y', 'gbias_z')
        data = {key : float(val) for key, val in zip(param_names, arr)}
    elif ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Don't know how to open {}".format(filepath))
    return data

calibration_params = load_params('/home/hannes/Datasets/gopro-gyro-dataset/walk_reference.csv')
gyro = load_gyro('/home/hannes/Datasets/gopro-gyro-dataset/walk_gyro.csv')
gyro.data = crisp.l3g4200d.post_process_L3G4200D_data(gyro.data.T).T

print 'gyro: Applying bias'
bias = np.array([calibration_params['gbias_{}'.format(axis)] for axis in 'xyz'])
gyro.data = gyro.data + bias.reshape(1,3) # Yes, actually plus

# Unpack R_g2c
r = np.array([calibration_params['rot_{}'.format(axis)] for axis in 'xyz'])
theta = np.linalg.norm(r)
v = r / theta
R_g2c = crisp.rotations.axis_angle_to_rotation_matrix(v, theta)
gyro.data = np.dot(R_g2c, gyro.data.T).T
gyro_times = np.arange(gyro.num_samples) / calibration_params['gyro_rate'] - calibration_params['time_offset']

SAVE_EXAMPLES = False
if SAVE_EXAMPLES:
    np.save('tests/example_gyro_data.npy', gyro.data)
    np.save('tests/example_gyro_times.npy', gyro_times)

nvm_model = NvmModel.from_file('/home/hannes/Code/rs-imusim/walk_model.nvm')
nvm_model.autoscale_walking()
nvm_model.normalize_world()

sfm_data = None

def plot_traj(traj, **kwargs):
    t = np.linspace(traj.startTime, traj.endTime, num=200)
    orient = traj.rotation(t)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(t, orient.array[:, i], **kwargs)
        for y in [-1, 0, 1]:
            plt.axhline(y, linestyle='--', color='k')

ds = Dataset()
ds.position_from_nvm(nvm_model, camera_fps=30.0)
#print 'Orientations from NVM'
ds.orientation_from_nvm(nvm_model, camera_fps=30.0)
plt.figure()
plot_traj(ds.trajectory, color='g', label='NVM')
print 'Orientations from gyro'
ds.orientation_from_gyro(gyro.data, gyro_times)
plot_traj(ds.trajectory, color='r', label='gyro')
plt.subplot(2,2,4)
plt.legend(loc='best')
plt.suptitle('Orientation from NVM vs gyro')
plt.show()
#ds.landmarks_from_nvm(nvm_model, camera_fps=30.0)
#ds.visualize()