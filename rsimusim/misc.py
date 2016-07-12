from __future__ import print_function, division

import os
import logging

import numpy as np
from crisp import GyroStream
import crisp.rotations

class CalibratedGyroStream(GyroStream):
    @classmethod
    def from_directory(cls, directory, sequence_name):
        data_path = os.path.join(directory, sequence_name + '_gyro.csv')
        param_path = os.path.join(directory, sequence_name + '_reference.csv')
        params = cls.load_params(param_path)
        instance = cls.from_csv(data_path)
        instance.params = params
        data = instance.data
        if sequence_name in ('rccar', 'walk', 'rotation'):
            import crisp.l3g4200d
            logging.info('Applying L3G4200D post processing')
            data = crisp.l3g4200d.post_process_L3G4200D_data(data.T).T
        bias = np.array([params['gbias_{}'.format(axis)] for axis in 'xyz']).reshape(1,3)
        data -= bias
        r = np.array([params['rot_{}'.format(axis)] for axis in 'xyz'])
        theta = np.linalg.norm(r)
        v = r / theta
        R_g2c = crisp.rotations.axis_angle_to_rotation_matrix(v, theta)
        data = R_g2c.dot(data.T).T
        instance.data = data
        dt = 1. / params['gyro_rate']
        timestamps = np.arange(data.shape[0]) * dt - params['time_offset']
        instance.timestamps = timestamps
        instance.integrate(dt)
        return instance

    @staticmethod
    def load_params(filepath):
        _, ext = os.path.splitext(filepath)
        if ext == '.csv':
            arr = np.loadtxt(filepath, delimiter=',')
            param_names = ('gyro_rate', 'time_offset', 'rot_x', 'rot_y', 'rot_z', 'gbias_x', 'gbias_y', 'gbias_z')
            data = {key : float(val) for key, val in zip(param_names, arr)}
        else:
            raise ValueError("Don't know how to open {}".format(filepath))
        return data

    def orientation_at(self, t):
        q = self._GyroStream__last_q
        if q is None:
            raise RuntimeError('Must integrate the stream before extracting orientation')
        n = np.flatnonzero(self.timestamps > t)[0]
        t1 = self.timestamps[n - 1]
        t2 = self.timestamps[n]
        q1 = q[n - 1]
        q2 = q[n]
        tau = (t - t1) / (t2 - t1)
        assert 0 <= tau <= 1
        if np.isclose(tau, 0.0):
            return q1
        elif np.isclose(tau, 1.0):
            return q2
        else:
            return crisp.rotations.slerp(q1, q2, tau)

    @property
    def orientations(self):
        return self._GyroStream__last_q