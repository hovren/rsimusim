from __future__ import print_function, division

import unittest

import crisp.rotations
import numpy as np
import numpy.testing as nt
from crisp.fastintegrate import integrate_gyro_quaternion_uniform
from imusim.maths.quaternions import Quaternion, QuaternionArray

from rsimusim.dataset import Dataset, DatasetBuilder, DatasetError, \
    resample_quaternion_array, quaternion_slerp, quaternion_array_interpolate
from rsimusim.nvm import NvmModel
from tests.helpers import random_orientation, unpack_quat, gyro_data_to_quaternion_array

NVM_EXAMPLE = 'example.nvm'
GYRO_EXAMPLE_DATA = np.load('example_gyro_data.npy')
GYRO_EXAMPLE_TIMES = np.load('example_gyro_times.npy')
GYRO_DT = float(GYRO_EXAMPLE_TIMES[1] - GYRO_EXAMPLE_TIMES[0])
GYRO_EXAMPLE_DATA_INT = integrate_gyro_quaternion_uniform(GYRO_EXAMPLE_DATA, GYRO_DT)
GYRO_EXAMPLE_DATA_Q = QuaternionArray(GYRO_EXAMPLE_DATA_INT)
assert len(GYRO_EXAMPLE_DATA_Q) == len(GYRO_EXAMPLE_TIMES)

class DatasetTests(unittest.TestCase):
    def test_position_from_nvm(self):
        nvm = NvmModel.from_file(NVM_EXAMPLE)
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
        nvm = NvmModel.from_file(NVM_EXAMPLE)
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

    def test_orientation_from_gyro(self):
        ds = Dataset()
        ds.orientation_from_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
        t0 = max(GYRO_EXAMPLE_TIMES[0], ds.trajectory.startTime)
        t1 = min(GYRO_EXAMPLE_TIMES[-1], ds.trajectory.endTime)
        i0 = np.flatnonzero(GYRO_EXAMPLE_TIMES >= t0)[0]
        i1 = np.flatnonzero(GYRO_EXAMPLE_TIMES <= t1)[-1]
        t0 = GYRO_EXAMPLE_TIMES[i0]
        t1 = GYRO_EXAMPLE_TIMES[i1]
        gyro_times_valid = GYRO_EXAMPLE_TIMES[i0:i1+1]
        gyro_data_valid = GYRO_EXAMPLE_DATA[i0:i1+1]
        rotvel_world = ds.trajectory.rotationalVelocity(gyro_times_valid)
        rotvel_body = ds.trajectory.rotation(gyro_times_valid).rotateFrame(rotvel_world)
        """
        import matplotlib.pyplot as plt
        for i in range(3):
            plt.subplot(3,1,1+i)
            plt.plot(gyro_times_valid, rotvel_body[i], color='g', linewidth=2)
            plt.plot(gyro_times_valid, gyro_data_valid.T[i], color='k', alpha=0.5)
        plt.show()
        """
        rotvel_err = rotvel_body - gyro_data_valid.T
        self.assertLess(np.mean(rotvel_err), 0.01)

    def test_orientation_from_gyro_shapes(self):
        N = 100
        gyro_times = np.arange(N) / 0.1
        valid_shapes = [(N, 3), (N, 4)]
        invalid_shapes = [(3, N), (4, N), (N, 2), (N, 5)]

        for shape in valid_shapes:
            gyro_data = np.zeros(shape)
            gyro_data[:, 0] = 1. # To get an OK quaternion
            ds = Dataset()
            ds.orientation_from_gyro(gyro_data, gyro_times)

        for shape in invalid_shapes:
            gyro_data = np.zeros(shape)
            ds = Dataset()
            with self.assertRaises(DatasetError):
                ds.orientation_from_gyro(gyro_data, gyro_times)

    def test_orientation_from_gyro_uniform_only(self):
        N = 100
        gyro_data = np.zeros((N, 3))
        gyro_times_valid = np.arange(N) / 0.1
        gyro_times_invalid = np.random.uniform(0, 10.0, size=N)
        gyro_times_invalid.sort()

        ds = Dataset()
        ds.orientation_from_gyro(gyro_data, gyro_times_valid)

        ds = Dataset()
        with self.assertRaises(DatasetError):
            ds.orientation_from_gyro(gyro_data, gyro_times_invalid)

    def test_landmarks_from_nvm(self):
        nvm = NvmModel.from_file(NVM_EXAMPLE)
        camera_fps = 30.0
        ds = Dataset()
        ds.landmarks_from_nvm(nvm)

        self.assertEqual(len(ds.landmarks), len(nvm.points))
        for ds_p, nvm_p in zip(ds.landmarks, nvm.points):
            nt.assert_equal(nvm_p.position, ds_p.position)

    def test_resample_quaternion_array(self):
        nvm = NvmModel.from_file(NVM_EXAMPLE)
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
        nvm = NvmModel.from_file(NVM_EXAMPLE)
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

    def test_selected_sources(self):
        db = DatasetBuilder()
        db.set_landmark_source('nvm')
        db.set_position_source('imu')
        db.set_orientation_source('imu')

        expected = {
            'orientation' : 'imu',
            'position' : 'imu',
            'landmark' : 'nvm'
        }

        self.assertEqual(db.selected_sources, expected)

    def test_add_nvm_twice_fail(self):
        db = DatasetBuilder()
        nvm1 = NvmModel()
        nvm2 = NvmModel()
        db.add_source_nvm(nvm1)
        with self.assertRaises(DatasetError):
            db.add_source_nvm(nvm2)

    def test_nvm_full(self):
        db = DatasetBuilder()
        nvm = NvmModel.from_file(NVM_EXAMPLE)
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

    def test_add_gyro_shape(self):
        N = 100
        gyro_times = np.arange(N) / 100.
        valid_shapes = [(N, 3)]
        invalid_shapes = [(3, N), (4, N), (N, 2), (N, 5)]

        for sh in valid_shapes:
            db = DatasetBuilder()
            gyro_data = np.random.normal(size=sh)
            db.add_source_gyro(gyro_data, gyro_times)

        for sh in invalid_shapes:
            db = DatasetBuilder()
            gyro_data = np.random.normal(size=sh)
            with self.assertRaises(DatasetError):
                db.add_source_gyro(gyro_data, gyro_times)

    def test_add_gyro_twice_fail(self):
        db = DatasetBuilder()
        N = 100
        gdata = np.zeros((N, 3))
        gtimes = np.zeros(N)
        db.add_source_gyro(gdata, gtimes)
        with self.assertRaises(DatasetError):
            db.add_source_gyro(gdata, gtimes)

    def test_add_gyro_generate_quat(self):
        db = DatasetBuilder()
        N = 100
        gdata = np.zeros((N, 3))
        gtimes = np.zeros(N)
        db.add_source_gyro(gdata, gtimes)
        self.assertEqual(len(db._gyro_quat), len(gtimes))

    def test_nvm_gyro(self):
        db = DatasetBuilder()
        camera_fps = 30.0
        db.add_source_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
        nvm = NvmModel.from_file(NVM_EXAMPLE)
        db.add_source_nvm(nvm, camera_fps=camera_fps)
        db.set_landmark_source('nvm')
        db.set_position_source('nvm')
        db.set_orientation_source('imu')
        ds = db.build()

        # 1) Dataset rotational velocity should match gyro (almost)
        t0 = max(GYRO_EXAMPLE_TIMES[0], ds.trajectory.startTime)
        t1 = min(GYRO_EXAMPLE_TIMES[-1], ds.trajectory.endTime)
        i0 = np.flatnonzero(GYRO_EXAMPLE_TIMES >= t0)[0]
        t0 = GYRO_EXAMPLE_TIMES[i0]
        i1 = np.flatnonzero(GYRO_EXAMPLE_TIMES <= t1)[-1]
        t1 = GYRO_EXAMPLE_TIMES[i1]
        gyro_part_data = GYRO_EXAMPLE_DATA[i0:i1+1]
        gyro_part_times = GYRO_EXAMPLE_TIMES[i0:i1+1]
        rotvel_ds_world = ds.trajectory.rotationalVelocity(gyro_part_times)
        rotvel_ds = ds.trajectory.rotation(gyro_part_times).rotateFrame(rotvel_ds_world)
        rotvel_err = rotvel_ds - gyro_part_data.T
        self.assertLess(np.mean(rotvel_err), 0.01)

        # 2) Dataset orientations should match approximately with NVM cameras
        for camera in nvm.cameras:
            t = camera.framenumber / camera_fps
            if not ds.trajectory.startTime <= t <= ds.trajectory.endTime:
                continue
            orientation_ds = ds.trajectory.rotation(t)
            if orientation_ds.dot(camera.orientation) < 0:
                orientation_ds = -orientation_ds
            nt.assert_almost_equal(unpack_quat(orientation_ds),
                                   unpack_quat(camera.orientation),
                                   decimal=1)

    def test_nvm_aligned_imu_orientations(self):
        nvm = NvmModel.from_file(NVM_EXAMPLE)
        db = DatasetBuilder()
        camera_fps = 30.0
        db.add_source_nvm(nvm, camera_fps=camera_fps)
        db.add_source_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
        orientations_aligned, new_times = db._nvm_aligned_imu_orientations()
        Q_aligned = QuaternionArray(orientations_aligned).unflipped()
        for camera in nvm.cameras:
            t = camera.framenumber / camera_fps
            qt = quaternion_array_interpolate(Q_aligned, new_times, t)
            if qt.dot(camera.orientation) < 0:
                qt = -qt
            nt.assert_almost_equal(unpack_quat(qt), unpack_quat(camera.orientation), decimal=1)

    def test_gyro_uniform(self):
        N = 1000
        gyro_data = np.zeros((3, N))
        gyro_times = np.random.uniform(0, 100, size=N)
        gyro_times.sort()
        db = DatasetBuilder()
        with self.assertRaises(DatasetError):
            db.add_source_gyro(gyro_data, gyro_times)

def test_crisp_slerp():
    "Check that crisp.rotations.slerp() works as intended"
    def constrain_angle(phi):
        while phi < -np.pi:
            phi += 2*np.pi
        while phi > np.pi:
            phi -= 2*np.pi
        return phi

    for _ in range(100):
        n = np.random.uniform(-1, 1, size=3)
        n /= np.linalg.norm(n)
        phi1 = np.random.uniform(-np.pi, np.pi)
        dphi = np.random.uniform(-np.pi, np.pi)
        phi2 = constrain_angle(phi1 + dphi)
        q0 = Quaternion.fromAxisAngle(n, phi1)
        q1 = Quaternion.fromAxisAngle(n, phi2)
        nt.assert_almost_equal(q0.magnitude, 1.0)
        nt.assert_almost_equal(q1.magnitude, 1.0)
        tau = 0.5
        qintp = Quaternion(*crisp.rotations.slerp(unpack_quat(q0), unpack_quat(q1), tau))
        nt.assert_almost_equal(qintp.magnitude, 1.0)
        axis, angle = qintp.toAxisAngle()
        if np.dot(axis, n) < 0:
            axis = -axis
            angle = -angle
        nt.assert_almost_equal(axis, n, decimal=3)

        constrained_result_angle = constrain_angle(angle)
        expected_angle = constrain_angle(phi1 + tau * dphi)
        nt.assert_almost_equal(constrained_result_angle, expected_angle, decimal=3)


def test_quaternion_slerp():
    for i in range(100):
        q0 = random_orientation()
        q1 = random_orientation()
        nt.assert_almost_equal(q0.magnitude, 1.0)
        nt.assert_almost_equal(q1.magnitude, 1.0)
        tau = 0.45
        qintp = quaternion_slerp(q0, q1, tau)
        assert not np.all(np.isnan(unpack_quat(qintp))), "qintp={}, q0={}, q1={}".format(qintp, q0, q1)

        qintp_ref = crisp.rotations.slerp(unpack_quat(q0), unpack_quat(q1), tau)

        nt.assert_almost_equal(unpack_quat(qintp), qintp_ref)
        nt.assert_almost_equal(qintp.magnitude, 1.0)

def test_quaternion_array_interpolate():
    N = 100
    qa = QuaternionArray([random_orientation() for _ in range(N)])
    qtimes = np.random.uniform(-2, 2, size=N)
    qtimes.sort()

    t_intp = np.random.uniform(qtimes[0], qtimes[-1])
    q_intp = quaternion_array_interpolate(qa, qtimes, t_intp)

    i = np.flatnonzero(qtimes > t_intp)[0]
    q0 = qa[i-1]
    q1 = qa[i]
    t0 = qtimes[i-1]
    t1 = qtimes[i]
    tau = (t_intp - t0) / (t1 - t0)
    qslerp = quaternion_slerp(q0, q1, tau)
    nt.assert_almost_equal(unpack_quat(q_intp), unpack_quat(qslerp))

def notest_bounds():
    times = [1.0, 2.0, 5.0, 9.0]
    bounds = rsimusim_legacy.dataset.create_bounds(times)
    expected_bounds = [0.5, 1.5, 3.5, 7.0, 11.0]
    nt.assert_almost_equal(bounds, expected_bounds)