from __future__ import print_function, division
import os
import shutil
import tempfile
import unittest

import h5py
import crisp.rotations
import numpy as np
import numpy.testing as nt
from crisp.fastintegrate import integrate_gyro_quaternion_uniform
from imusim.maths.quaternions import Quaternion, QuaternionArray

from rsimusim.dataset import Dataset, DatasetBuilder, DatasetError, \
    resample_quaternion_array, quaternion_slerp, quaternion_array_interpolate, create_bounds
from rsimusim.sfm import SfmResult, VisualSfmResult, OpenMvgResult
from tests.helpers import random_orientation, unpack_quat, gyro_data_to_quaternion_array, find_landmark

NVM_EXAMPLE = 'example.nvm'
OPENMVG_EXAMPLE = 'example_sfm_data.json'
GYRO_EXAMPLE_DATA = np.load('example_gyro_data.npy')
GYRO_EXAMPLE_TIMES = np.load('example_gyro_times.npy')
GYRO_DT = float(GYRO_EXAMPLE_TIMES[1] - GYRO_EXAMPLE_TIMES[0])
GYRO_EXAMPLE_DATA_INT = integrate_gyro_quaternion_uniform(GYRO_EXAMPLE_DATA, GYRO_DT)
GYRO_EXAMPLE_DATA_Q = QuaternionArray(GYRO_EXAMPLE_DATA_INT)
assert len(GYRO_EXAMPLE_DATA_Q) == len(GYRO_EXAMPLE_TIMES)
CAMERA_FPS = 30.

class DatasetGyroTests(unittest.TestCase):
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

    def test_resample_quaternion_array(self):
        sfm = VisualSfmResult.from_file(NVM_EXAMPLE, camera_fps=CAMERA_FPS)
        Q = QuaternionArray([view.orientation for view in sfm.views])
        Q = Q.unflipped()
        view_times = np.array([view.time for view in sfm.views])
        new_size = 500
        Q_resamp, Q_t = resample_quaternion_array(Q, view_times, resize=new_size)
        self.assertEqual(len(Q_resamp), len(Q_t))
        self.assertEqual(len(Q_resamp), new_size)
        nt.assert_almost_equal(Q_t[0], view_times[0])
        nt.assert_almost_equal(Q_t[-1], view_times[-1])
        nt.assert_almost_equal(unpack_quat(Q_resamp[0]), unpack_quat(Q[0]))
        nt.assert_almost_equal(unpack_quat(Q_resamp[-1]), unpack_quat(Q[-1]))


class AbstractDatasetSfmTestMixin(object):
    def load_dataset(self):
        self.ds = Dataset()
        self.ds.landmarks_from_sfm(self.sfm)
        self.ds.position_from_sfm(self.sfm)
        self.ds.orientation_from_sfm(self.sfm)

    def test_landmarks_loaded(self):
        self.assertEqual(len(self.ds.landmarks), len(self.sfm.landmarks))

    def test_positions(self):
        position = self.ds.trajectory.position
        num_tried = 0
        for view in self.sfm.views:
            if self.ds.trajectory.startTime <= view.time <= self.ds.trajectory.endTime:
                p = position(view.time).ravel()
                nt.assert_almost_equal(p, view.position, decimal=2)
                num_tried += 1
        self.assertGreater(num_tried, 10)

    def test_orientations(self):
        num_tried = 0
        trajectory = self.ds.trajectory
        for view in self.sfm.views:
            if trajectory.startTime < view.time < trajectory.endTime:
                vq = view.orientation
                q = trajectory.rotation(view.time)
                if view.orientation.dot(q) < 0:
                    q = -q
                nt.assert_almost_equal(q.components, vq.components, decimal=1)
                num_tried += 1
        self.assertGreater(num_tried, 10)


    def notest_projection(self):
        max_mean_reproj_error = 15.0 # Pixels
        num_test = min(500, len(self.ds.landmarks))
        chosen_landmarks = [self.ds.landmarks[i] for i in np.random.choice(len(self.ds.landmarks), num_test)]
        corresp_landmarks = [self.sfm.landmarks[lm.id] for lm in chosen_landmarks]
        distance_list = []
        for ds_lm, sfm_lm in zip(chosen_landmarks, corresp_landmarks):
            self.assertEqual(ds_lm.id, sfm_lm.id)
            nt.assert_equal(ds_lm.position, sfm_lm.position)
            X = ds_lm.position.reshape(3,1)
            for bound_id in ds_lm.visibility:
                ta, tb = self.ds._landmark_bounds[bound_id:bound_id+2]
                matching_views = [v for v in self.sfm.views if ta <= v.time <= tb]
                self.assertEqual(len(matching_views), 1)
                view = matching_views[0]
                t = view.time
                y_expected = sfm_lm.observations[view.id].reshape(2,1)
                if self.ds.trajectory.startTime <= t <= self.ds.trajectory.endTime:
                    q = self.ds.trajectory.rotation(t)
                    R = q.toMatrix()
                    p = self.ds.trajectory.position(t).reshape(3,1)
                    nt.assert_almost_equal(p, view.position.reshape(3,1), decimal=1)
                    qtest = q if q.dot(view.orientation) > 0 else -q
                    nt.assert_almost_equal(qtest.components, view.orientation.components, decimal=1)
                    X_view = np.dot(R, X - p)
                    self.assertEqual(X_view.shape, (3,1))
                    y = np.dot(self.CAMERA_MATRIX, X_view)
                    self.assertEqual(y.size, 3)
                    y = y[:2] / y[2,0]
                    self.assertEqual(y.size, 2)
                    distance = np.linalg.norm(y - y_expected)
                    distance_list.append(distance)
                    #self.assertLess(distance, max_reproj_error)
        import matplotlib.pyplot as plt
        plt.hist(distance_list, bins=np.linspace(0,70))
        plt.title(self.__class__.__name__)
        plt.show()
        self.assertLess(np.mean(distance_list), max_mean_reproj_error)

    def notest_plot_trajectory(self):
        view_pos = np.vstack([v.position for v in self.sfm.views]).T
        view_times = np.array([v.time for v in self.sfm.views])
        assert view_pos.shape[0] == 3 and view_pos.ndim == 2
        view_q = QuaternionArray([v.orientation for v in self.sfm.views])
        view_q = view_q.unflipped()

        t = view_times #np.linspace(self.ds.trajectory.startTime, self.ds.trajectory.endTime,
                       # num=2000)
        traj_pos = self.ds.trajectory.position(t)
        traj_q = self.ds.trajectory.rotation(t)

        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(3):
            plt.subplot(3,1,1+i)
            plt.plot(view_times, view_pos[i], label='views')
            plt.plot(t, traj_pos[i], label='trajectory')
        plt.suptitle(self.__class__.__name__)
        plt.legend()

        plt.figure()
        for i in range(4):
            plt.subplot(5,1,1+i)
            plt.plot(view_times, view_q.array[:, i], label='views')
            plt.plot(t, traj_q.array[:, i], label='trajectory')
        dq = view_q.conjugate * traj_q
        angles = [q.toAxisAngle()[1] for q in dq]
        plt.subplot(5,1,5)
        plt.plot(t, np.rad2deg(angles))
        plt.ylabel('degrees of error')
        plt.suptitle(self.__class__.__name__)
        plt.legend()
        plt.show()


class DatasetFromNvmTests(AbstractDatasetSfmTestMixin, unittest.TestCase):
    CAMERA_MATRIX = np.array(
        [[ 850.051391602,    0.        ,  0],
         [ 0.        ,  850.051391602,  0],
     [   0.        ,    0.        ,    1.        ]]
)
    def setUp(self):
        self.sfm = VisualSfmResult.from_file(NVM_EXAMPLE, CAMERA_FPS)
        self.load_dataset()


class DatasetFromOpenMvgTests(AbstractDatasetSfmTestMixin, unittest.TestCase):
    CAMERA_MATRIX = np.array(
            [[ 862.43356025,    0.        ,  987.89341878],
       [   0.        ,  862.43356025,  525.14469927],
       [   0.        ,    0.        ,    1.        ]])

    def setUp(self):
        self.sfm = OpenMvgResult.from_file(OPENMVG_EXAMPLE, CAMERA_FPS)
        self.load_dataset()

class DatasetBuilderGeneralTests(unittest.TestCase):
    def test_source_types(self):
        valid = ('imu', 'sfm')
        landmark_valid = ('sfm', )
        invalid = ('gyro', 'acc', 'bacon', 'openmvg', 'nvm')
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
        sfm = VisualSfmResult.from_file(NVM_EXAMPLE, camera_fps=30.)
        db.add_source_sfm(sfm)
        with self.assertRaises(DatasetError):
            db.build()
        db.set_position_source('sfm')
        with self.assertRaises(DatasetError):
            db.build()
        db.set_orientation_source('sfm')
        with self.assertRaises(DatasetError):
            db.build()
        db.set_landmark_source('sfm')
        db.build() # all sources selected: OK

    def test_selected_sources(self):
        db = DatasetBuilder()
        db.set_landmark_source('sfm')
        db.set_position_source('imu')
        db.set_orientation_source('imu')

        expected = {
            'orientation' : 'imu',
            'position' : 'imu',
            'landmark' : 'sfm'
        }

        self.assertEqual(db.selected_sources, expected)

    def test_add_sfm_twice_fail(self):
        db = DatasetBuilder()
        sfm1 = SfmResult()
        sfm2 = SfmResult()
        db.add_source_sfm(sfm1)
        with self.assertRaises(DatasetError):
            db.add_source_sfm(sfm2)

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

    def test_gyro_uniform(self):
        N = 1000
        gyro_data = np.zeros((3, N))
        gyro_times = np.random.uniform(0, 100, size=N)
        gyro_times.sort()
        db = DatasetBuilder()
        with self.assertRaises(DatasetError):
            db.add_source_gyro(gyro_data, gyro_times)

class DatasetBuilderSfmMixin(object):
    def test_sfm_aligned_imu_orientations(self):
        db = DatasetBuilder()
        camera_fps = 30.0
        db.add_source_sfm(self.sfm)
        db.add_source_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
        orientations_aligned, new_times = db._sfm_aligned_imu_orientations()
        Q_aligned = QuaternionArray(orientations_aligned).unflipped()
        for view in self.sfm.views:
            qt = quaternion_array_interpolate(Q_aligned, new_times, view.time)
            if qt.dot(view.orientation) < 0:
                qt = -qt
            nt.assert_almost_equal(qt.components, view.orientation.components, decimal=1)

    def test_sfm_only(self):
        db = DatasetBuilder()
        db.add_source_sfm(self.sfm)
        db.set_orientation_source('sfm')
        db.set_position_source('sfm')
        db.set_landmark_source('sfm')
        ds = db.build()

        for view in self.sfm.views:
            t = view.time
            if ds.trajectory.startTime <= t <= ds.trajectory.endTime:
                p_ds = ds.trajectory.position(t).flatten()
                q_ds = ds.trajectory.rotation(t)
                nt.assert_almost_equal(p_ds, view.position, decimal=2)
                q_sfm = view.orientation
                if q_ds.dot(q_sfm) < 0:
                    q_sfm *= -1
                nt.assert_almost_equal(q_ds.components, q_sfm.components, decimal=1)

        # Assume landmark order intact
        self.assertEqual(len(ds.landmarks), len(self.sfm.landmarks))
        for sfm_lm, ds_lm in zip(self.sfm.landmarks, ds.landmarks):
            nt.assert_equal(sfm_lm.position, ds_lm.position)

    def test_with_gyro_orientation(self):
        db = DatasetBuilder()
        db.add_source_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
        db.add_source_sfm(self.sfm)
        db.set_landmark_source('sfm')
        db.set_position_source('sfm')
        db.set_orientation_source('imu')
        ds = db.build()

        # 2) Dataset orientations should match approximately with NVM cameras
        for view in self.sfm.views:
            t = view.time
            if ds.trajectory.startTime <= t <= ds.trajectory.endTime:
                orientation_ds = ds.trajectory.rotation(t)
                if orientation_ds.dot(view.orientation) < 0:
                    orientation_ds = -orientation_ds
                nt.assert_almost_equal(orientation_ds.components,
                                       view.orientation.components,
                                       decimal=1)

    def notest_with_gyro_velocity(self):
        # NOTE: Since the gyro orientations are transported into the
        # SfM coordinate frame, comparing the gyro velocity is not
        # so simple, so we deactivate this test for now
        db = DatasetBuilder()
        db.add_source_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
        db.add_source_sfm(self.sfm)
        db.set_landmark_source('sfm')
        db.set_position_source('sfm')
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
        rotvel_err = np.linalg.norm(rotvel_ds - gyro_part_data.T, axis=1)
        self.assertLess(np.mean(rotvel_err), 0.01)

        import matplotlib.pyplot as plt
        fig1 = plt.figure()
        for i in range(3):
            plt.subplot(3,1,1+i)
            plt.plot(gyro_part_times, gyro_part_data[:,i], label='gyro', linewidth=3)
            plt.plot(gyro_part_times, rotvel_ds[i, :], label='ds')
            plt.legend(loc='upper right')
        plt.suptitle(self.__class__.__name__)
        fig1.savefig('/tmp/{}_w.pdf'.format(self.__class__.__name__))
        view_q = QuaternionArray([v.orientation for v in self.sfm.views])
        view_q = view_q.unflipped()
        view_times = [v.time for v in self.sfm.views]
        traj_q = ds.trajectory.rotation(gyro_part_times)
        fig2 = plt.figure()
        for i in range(4):
            plt.subplot(4,1,1+i)
            plt.plot(view_times, view_q.array[:, i], '-o')
            plt.plot(gyro_part_times, traj_q.array[:, i])
        plt.suptitle(self.__class__.__name__)
        fig2.savefig('/tmp/{}_q.pdf'.format(self.__class__.__name__))
        plt.show()

class DatasetBuilderNvm(DatasetBuilderSfmMixin, unittest.TestCase):
    def setUp(self):
        self.sfm = VisualSfmResult.from_file(NVM_EXAMPLE, camera_fps=CAMERA_FPS)

class DatasetBuilderOpenMvg(DatasetBuilderSfmMixin, unittest.TestCase):
    def setUp(self):
        self.sfm = OpenMvgResult.from_file(OPENMVG_EXAMPLE, CAMERA_FPS)

class DatasetSaveTests(unittest.TestCase):
    def setUp(self):
        db = DatasetBuilder()
        db.add_source_gyro(GYRO_EXAMPLE_DATA, GYRO_EXAMPLE_TIMES)
        sfm = VisualSfmResult.from_file(NVM_EXAMPLE, camera_fps=CAMERA_FPS)
        db.add_source_sfm(sfm)
        db.set_landmark_source('sfm')
        db.set_orientation_source('imu')
        db.set_position_source('sfm')
        self.ds = db.build()

        self.testdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.testdir)

    def test_save_dont_overwrite(self):
        dataset_name = 'test_dataset'
        outfilename = os.path.join(self.testdir, 'dataset1.h5')
        with open(outfilename, 'w') as f:
            f.write("Hello\n")
        with self.assertRaises(DatasetError):
            self.ds.save(outfilename, dataset_name)

    def test_save(self):
        outfilename = os.path.join(self.testdir, 'dataset1.h5')
        dataset_name = 'test_dataset'
        self.ds.save(outfilename, dataset_name)
        self.assertTrue(os.path.exists(outfilename))

        with h5py.File(outfilename, 'r') as h5f:
            self.assertEqual(h5f.attrs['name'], dataset_name)
            for key in ('position', 'orientation'):
                self.assertTrue(key in h5f.keys())
                group = h5f[key]
                for gkey in ('data', 'timestamps'):
                    self.assertTrue(gkey in group.keys())

            landmarks_group = h5f['landmarks']
            self.assertTrue('visibility_bounds' in landmarks_group)
            self.assertEqual(len(landmarks_group['visibility'].keys()), len(self.ds.landmarks))
            self.assertTrue('positions' in landmarks_group.keys())
            self.assertTrue('colors' in landmarks_group.keys())
            self.assertTrue('visibility' in landmarks_group.keys())

    def test_save_reload_multi(self):
        t = np.linspace(self.ds.trajectory.startTime, self.ds.trajectory.endTime, num=200)
        original_positions = self.ds.trajectory.position(t)
        original_rotations = self.ds.trajectory.rotation(t)
        original_landmarks = self.ds.landmarks
        t_vis = np.linspace(t[0], t[-1], 10)
        original_visibles = [self.ds.visible_landmarks(_t) for _t in t_vis]

        outfilename = os.path.join(self.testdir, 'dataset_save.h5')
        dataset_name = 'test_dataset'
        ds = self.ds
        for i in range(2):
            if os.path.exists(outfilename):
                os.unlink(outfilename)
            # Save previous dataset
            ds.save(outfilename, dataset_name)

            # Load it again, check that it is the same
            ds = Dataset.from_file(outfilename)
            positions = ds.trajectory.position(t)
            rotations = ds.trajectory.rotation(t)
            nt.assert_equal(positions, original_positions)
            nt.assert_equal(rotations.array, original_rotations.array)

            self.assertEqual(len(ds.landmarks), len(original_landmarks))
            for old, new in zip(original_landmarks, ds.landmarks):
                nt.assert_equal(new.position, old.position)
                nt.assert_equal(new.color, old.color)
                self.assertEqual(new.visibility, old.visibility)

            visibles = [ds.visible_landmarks(_t) for _t in t_vis]
            for old_vis, new_vis in zip(original_visibles, visibles):
                self.assertEqual(len(new_vis), len(old_vis))
                for old_lm, new_lm in zip(old_vis, new_vis):
                    nt.assert_equal(new_lm.position, old_lm.position)
                    self.assertEqual(new_lm.visibility, old_lm.visibility)




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
    for _ in range(100):
        N = 100
        qa = QuaternionArray([random_orientation() for _ in range(N)])
        qtimes = np.random.uniform(-10, 10, size=N)
        qtimes.sort()

        t_intp = np.random.uniform(qtimes[0], qtimes[-1])
        q_intp = quaternion_array_interpolate(qa, qtimes, t_intp)

        i = np.flatnonzero(qtimes > t_intp)[0]
        q0 = qa[i-1]
        q1 = qa[i]
        t0 = qtimes[i-1]
        t1 = qtimes[i]
        tau = np.clip((t_intp - t0) / (t1 - t0), 0, 1)
        qslerp = quaternion_slerp(q0, q1, tau)
        if qslerp.dot(q_intp) < 0:
            qslerp = -qslerp
        nt.assert_almost_equal(unpack_quat(q_intp), unpack_quat(qslerp))

def test_bounds():
    times = [1.0, 2.0, 5.0, 9.0]
    bounds = create_bounds(times)
    #expected_bounds = [0.5, 1.5, 3.5, 7.0, 11.0]
    expected_bounds = [-float('inf'), 1.5, 3.5, 7.0, float('inf')]
    nt.assert_almost_equal(bounds, expected_bounds)