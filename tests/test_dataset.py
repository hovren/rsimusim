from __future__ import print_function, division

import unittest
import random

import numpy.testing as nt
import numpy as np

from imusim.maths.quaternions import Quaternion, QuaternionArray

from rsimusim.nvm import NvmModel, NvmError
from rsimusim.dataset import Dataset, DatasetBuilder, DatasetError, resample_quaternion_array
from tests.helpers import random_orientation, unpack_quat, gyro_data_to_quaternion_array


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

    def test_orientation_from_nvm(self):
        nvm = NvmModel.from_file(self.EXAMPLE_NVM)
        camera_fps = 30.0
        ds = Dataset()
        ds.orientation_from_nvm(nvm, camera_fps=camera_fps)

        for camera in nvm.cameras:
            camera_time = camera.framenumber / camera_fps
            ds_rot = ds.trajectory.rotation(camera_time)
            if np.all(np.isnan(unpack_quat(ds_rot))):
                continue
            cam_rot = camera.orientation
            # Normalize sign before comparison
            # Normalize by largest element to avoid near-zero sign problems
            i = np.argmax(np.abs(unpack_quat(cam_rot)))
            ds_rot *= np.sign(unpack_quat(ds_rot)[i])
            cam_rot *= np.sign(unpack_quat(cam_rot)[i])
            nt.assert_almost_equal(unpack_quat(ds_rot), unpack_quat(cam_rot), decimal=1)

    def test_landmarks_from_nvm(self):
        nvm = NvmModel.from_file(self.EXAMPLE_NVM)
        camera_fps = 30.0
        ds = Dataset()
        ds.landmarks_from_nvm(nvm)

        self.assertEqual(len(ds.landmarks), len(nvm.points))
        for ds_p, nvm_p in zip(ds.landmarks, nvm.points):
            nt.assert_equal(nvm_p.position, ds_p.position)

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

class DatasetBuilderTests(unittest.TestCase):
    NVM_EXAMPLE = 'example.nvm'
    GYRO_EXAMPLE = 'example_gyro.npy'

    def test_source_types(self):
        valid = ('nvm', 'imu')
        landmark_valid = ('nvm', )
        invalid = ('gyro', 'acc', 'bacon')
        db = DatasetBuilder()
        for s in valid:
            db.set_orientation_source(s)
            db.set_position_source(s)

        for s in invalid:
            with self.assertRaises(DatasetError):
                db.set_orientation_source(s)
            with self.assertRaises(DatasetError):
                db.set_position_source(s)
            with self.assertRaises(DatasetError):
                db.set_landmark_source(s)

        for s in landmark_valid:
            db.set_landmark_source(s)

    def test_missing_source_fail(self):
        db = DatasetBuilder()
        nvm = NvmModel.from_file(self.NVM_EXAMPLE)
        db.add_source_nvm(nvm)
        with self.assertRaises(DatasetError):
            db.build()
        db.set_position_source('nvm')
        with self.assertRaises(DatasetError):
            db.build()
        db.set_orientation_source('nvm')
        with self.assertRaises(DatasetError):
            db.build()
        db.set_landmark_source('nvm')
        db.build() # all sources selected: OK

    def test_add_nvm_twice_fail(self):
        db = DatasetBuilder()
        nvm1 = NvmModel()
        nvm2 = NvmModel()
        db.add_source_nvm(nvm1)
        with self.assertRaises(DatasetError):
            db.add_source_nvm(nvm2)

    def test_nvm_full(self):
        db = DatasetBuilder()
        nvm = NvmModel.from_file(self.NVM_EXAMPLE)
        camera_fps = 30.0
        db.add_source_nvm(nvm, camera_fps=camera_fps)
        db.set_orientation_source('nvm')
        db.set_position_source('nvm')
        db.set_landmark_source('nvm')
        ds = db.build()

        cameras = sorted(nvm.cameras, key=lambda c: c.framenumber)
        for camera in cameras:
            t = camera.framenumber / camera_fps
            ds_pos = ds.trajectory.position(t).flatten()
            if np.all(np.isnan(ds_pos)):
                continue
            nt.assert_almost_equal(ds_pos, camera.position, decimal=2)

            ds_rot = ds.trajectory.rotation(t)
            cam_rot = camera.orientation
            i = np.argmax(np.abs(unpack_quat(cam_rot)))
            ds_rot *= np.sign(unpack_quat(ds_rot)[i])
            cam_rot *= np.sign(unpack_quat(cam_rot)[i])
            nt.assert_almost_equal(unpack_quat(ds_rot), unpack_quat(cam_rot), decimal=1)

        # Assume landmark order intact
        self.assertEqual(len(ds.landmarks), len(nvm.points))
        for nvm_p, ds_p in zip(nvm.points, ds.landmarks):
            nt.assert_equal(nvm_p.position, ds_p.position)




def notest_bounds():
    times = [1.0, 2.0, 5.0, 9.0]
    bounds = rsimusim_legacy.dataset.create_bounds(times)
    expected_bounds = [0.5, 1.5, 3.5, 7.0, 11.0]
    nt.assert_almost_equal(bounds, expected_bounds)