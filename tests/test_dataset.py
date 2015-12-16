from __future__ import print_function, division

import unittest
import random

import numpy.testing as nt
import numpy as np

from imusim.maths.quaternions import Quaternion, QuaternionArray

from rsimusim.nvm import NvmModel, NvmError
from rsimusim.dataset import Dataset, resample_quaternion_array

def unpack_quat(q):
    return np.array([q.w, q.x, q.y, q.z])

class DatasetTests(unittest.TestCase):
    EXAMPLE_NVM = 'example.nvm'

    def test_position_from_nvm(self):
        nvm = NvmModel.from_file(self.EXAMPLE_NVM)
        ds = Dataset()
        camera_fps = 30.0
        ds.position_from_nvm(nvm, camera_fps=camera_fps)

        cameras = [nvm.cameras[i] for i in
                   np.random.choice(len(nvm.cameras), size=10, replace=False)]
        for camera in cameras:
            camera_time = camera.framenumber / camera_fps
            ds_pos = ds.trajectory.position(camera_time).flatten()
            if np.all(np.isnan(ds_pos)):
                continue
            nt.assert_almost_equal(ds_pos, camera.position, decimal=1)

    def DISABLED_notest_orientation_from_nvm(self):
        nvm = NvmModel.from_file(self.EXAMPLE_NVM)
        camera_fps = 30.0
        ds = Dataset()
        ds.orientation_from_nvm(nvm, camera_fps=camera_fps)

        cameras = [nvm.cameras[i] for i in
                   np.random.choice(len(nvm.cameras), size=10, replace=False)]

        for camera in cameras:
            camera_time = camera.framenumber / camera_fps
            ds_rot = ds.trajectory.rotation(camera_time)
            if np.all(np.isnan(unpack_quat(ds_rot))):
                continue
            cam_rot = camera.orientation
            dq = ds_rot.conjugate * cam_rot
            v, theta = dq.toAxisAngle()
            self.assertLessEqual(abs(theta - np.pi), np.deg2rad(10.0))


    def test_resample_quaternion_array(self):
        nvm = NvmModel.from_file(self.EXAMPLE_NVM)
        cameras = sorted(nvm.cameras, key=lambda c: c.framenumber)
        Q = QuaternionArray([c.orientation for c in cameras])
        Q = Q.unflipped()
        camera_fps = 30.
        camera_times = np.array([c.framenumber / camera_fps for c in cameras])
        new_size = 500
        Q_resamp, Q_t = resample_quaternion_array(Q, camera_times, resize=new_size)
        self.assertEqual(len(Q_resamp), len(Q_t))
        self.assertEqual(len(Q_resamp), new_size)
        nt.assert_almost_equal(Q_t[0], camera_times[0])
        nt.assert_almost_equal(Q_t[-1], camera_times[-1])
        nt.assert_almost_equal(unpack_quat(Q_resamp[0]), unpack_quat(Q[0]))
        nt.assert_almost_equal(unpack_quat(Q_resamp[-1]), unpack_quat(Q[-1]))

def notest_bounds():
    times = [1.0, 2.0, 5.0, 9.0]
    bounds = rsimusim_legacy.dataset.create_bounds(times)
    expected_bounds = [0.5, 1.5, 3.5, 7.0, 11.0]
    nt.assert_almost_equal(bounds, expected_bounds)