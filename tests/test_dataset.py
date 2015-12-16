import unittest
import random

import numpy.testing as nt
import numpy as np

from imusim.maths.quaternions import Quaternion, QuaternionArray

from rsimusim.nvm import NvmModel, NvmError

def random_position():
    return np.random.uniform(-10, 10, size=3)

def random_orientation():
    q = np.random.uniform(-1, 1, size=4)
    return Quaternion(*(q / np.linalg.norm(q)))

class NvmTests(unittest.TestCase):
    def test_start_empty(self):
        nvm = NvmModel()
        self.assertEqual(len(nvm.cameras), 0)
        self.assertEqual(len(nvm.points), 0)

    def test_add_camera(self):
        nvm = NvmModel()
        cam_id = 0
        cam_filename = 'somepic.jpg'
        q = Quaternion(1, 0, 0, 0)
        p = np.array([1., 2., 3.])
        nvm.add_camera(cam_id, cam_filename, q, p)
        self.assertEqual(len(nvm.cameras), 1)

    def test_add_point(self):
        nvm = NvmModel()
        p = np.random.uniform(-1, 1, size=3)
        color = (1, 0, 0)
        visibility = []
        nvm.add_point(p, color, visibility)
        self.assertEqual(len(nvm.points), 1)

    def test_add_same_camera_id(self):
        nvm = NvmModel()
        nvm.add_camera(0, 'somefile1.jpg', random_orientation(), random_position())
        nvm.add_camera(1, 'somefile2.jpg', random_orientation(), random_position())
        with self.assertRaises(NvmError):
            nvm.add_camera(0, 'somefile3.jpg', random_orientation(), random_position())

    def test_add_point_camera_id_missing(self):
        nvm = NvmModel()
        camera_id = 0
        nvm.add_camera(camera_id, 'somefile.jpg', random_orientation(), random_position())
        # Cameras with id 12 and 14 does not exist, this is an error
        with self.assertRaises(NvmError):
            nvm.add_point(random_position(), (1, 0, 0), [12, 14])

    def test_camera_by_id(self):
        nvm = NvmModel()
        camera_data = [
            (i, 'somefile{:04d}.jpg'.format(i), random_orientation(), random_position())
            for i in range(100)
        ]
        random.shuffle(camera_data)

        for cam_id, filename, q, p in camera_data:
            nvm.add_camera(cam_id, filename, q, p)

        cam50 = nvm.camera_by_id(50)
        self.assertEqual(cam50.id, 50)
        self.assertEqual(cam50.filename, 'somefile0050.jpg')


class NvmExampleTests(unittest.TestCase):
    nvm = None

    def setUp(self):
        if not self.nvm:
            self.nvm = NvmModel.from_file('example.nvm', load_measurements=True)

    def test_cameras_loaded(self):
        self.assertEqual(len(self.nvm.cameras), 188)

        # Cameras should have an ID that is ever increasing
        camera_ids = [camera.id for camera in self.nvm.cameras]
        self.assertEqual(camera_ids, range(len(camera_ids)))

    def test_focal(self):
        self.assertEqual(self.nvm.focal, 850.051391602)

    def test_points_loaded(self):
        self.assertEqual(len(self.nvm.points), 2682)

    def test_reprojection(self):
        def project_point_camera(world_point, camera, focal):
            Xw = world_point.position
            R = camera.orientation.toMatrix()
            Xc = np.dot(R, (Xw - camera.position)).reshape(3,1)
            K = np.array([[focal, 0, 0],
                         [0, focal, 0],
                         [0, 0, 1.]])
            y = np.dot(K, Xc)
            y /= Xc[2]
            return y[:2].flatten()

        max_pixel_distance = 7.0
        for p in self.nvm.points[:5]:
            for camera_id, measurement in zip(p.visibility, p.measurements.T):
                camera = self.nvm.camera_by_id(camera_id)
                yhat = project_point_camera(p, camera, self.nvm.focal)
                distance = np.linalg.norm(yhat - measurement)
                self.assertLessEqual(distance, max_pixel_distance)


def notest_bounds():
    times = [1.0, 2.0, 5.0, 9.0]
    bounds = rsimusim_legacy.dataset.create_bounds(times)
    expected_bounds = [0.5, 1.5, 3.5, 7.0, 11.0]
    nt.assert_almost_equal(bounds, expected_bounds)
