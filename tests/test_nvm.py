import unittest
import random

import numpy.testing as nt
import numpy as np

from imusim.maths.quaternions import Quaternion, QuaternionArray

from rsimusim.nvm import NvmModel, NvmError, NvmCamera
from tests.helpers import random_position, random_orientation, random_focal

class NvmTests(unittest.TestCase):
    def test_start_empty(self):
        nvm = NvmModel()
        self.assertEqual(len(nvm.cameras), 0)
        self.assertEqual(len(nvm.points), 0)

    def test_camera_framenumber(self):
        test_data = [('house_123.jpg,', 123),
                     ('building_00234.jpeg', 234),
                     ('crocodile0345.jpg', 345),
                     ('horse456.png', 456),
                     ('cat045frame0005.jpg', 5)]

        for i, (filename, framenumber) in enumerate(test_data):
            camera = NvmCamera(i, filename, random_focal(), random_orientation(), random_position())
            self.assertEqual(camera.framenumber, framenumber)

        with self.assertRaises(NvmError):
            camera = NvmCamera(99, 'f23noframes.jpg', random_focal(), random_orientation(), random_position())
            fn = camera.framenumber

    def test_add_camera(self):
        nvm = NvmModel()
        cam_id = 0
        cam_filename = 'somepic.jpg'
        q = Quaternion(1, 0, 0, 0)
        p = np.array([1., 2., 3.])
        focal = random_focal()
        nvm.add_camera(cam_id, cam_filename, focal, q, p)
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
        nvm.add_camera(0, 'somefile1.jpg', random_focal(), random_orientation(), random_position())
        nvm.add_camera(1, 'somefile2.jpg', random_focal(), random_orientation(), random_position())
        with self.assertRaises(NvmError):
            nvm.add_camera(0, 'somefile3.jpg', random_focal(), random_orientation(), random_position())

    def test_add_point_camera_id_missing(self):
        nvm = NvmModel()
        camera_id = 0
        nvm.add_camera(camera_id, 'somefile.jpg', random_focal(), random_orientation(), random_position())
        # Cameras with id 12 and 14 does not exist, this is an error
        with self.assertRaises(NvmError):
            nvm.add_point(random_position(), (1, 0, 0), [12, 14])

    def test_camera_by_id(self):
        nvm = NvmModel()
        camera_data = [
            (i, 'somefile{:04d}.jpg'.format(i), random_focal(), random_orientation(), random_position())
            for i in range(100)
        ]
        random.shuffle(camera_data)

        for cam_id, filename, focal, q, p in camera_data:
            nvm.add_camera(cam_id, filename, focal, q, p)

        cam50 = nvm.camera_by_id(50)
        self.assertEqual(cam50.id, 50)
        self.assertEqual(cam50.filename, 'somefile0050.jpg')


class NvmExampleTests(unittest.TestCase):
    NVM_FILENAME = 'example.nvm'

    def setUp(self):
        self.nvm = NvmModel.from_file(self.NVM_FILENAME, load_measurements=True)

    def test_no_global_focal(self):
        with self.assertRaises(AttributeError):
            focal = self.nvm.focal

    def test_not_load_measurements_default(self):
        nvm_default = NvmModel.from_file(self.NVM_FILENAME)
        nvm_no_meas = NvmModel.from_file(self.NVM_FILENAME, load_measurements=False)

        num_meas_default = sum(p.measurements.shape[1] for p in nvm_default.points)
        num_meas_no_meas = sum(p.measurements.shape[1] for p in nvm_no_meas.points)
        num_meas_yes = sum(p.measurements.shape[1] for p in self.nvm.points)

        self.assertEqual(num_meas_default, 0)
        self.assertEqual(num_meas_no_meas, 0)
        self.assertGreater(num_meas_yes, 0)

    def test_cameras_loaded(self):
        self.assertEqual(len(self.nvm.cameras), 188)

        # Cameras should have an ID that is ever increasing
        camera_ids = [camera.id for camera in self.nvm.cameras]
        self.assertEqual(camera_ids, range(len(camera_ids)))

        # Shared focals
        shared_focal = 850.051391602
        camera_focals = [camera.focal for camera in self.nvm.cameras]
        nt.assert_almost_equal(camera_focals, shared_focal)

    def test_points_loaded(self):
        self.assertEqual(len(self.nvm.points), 2682)

    def test_reprojection(self):
        max_pixel_distance = 12.0
        for p in self.nvm.points:
            for camera_id, measurement in zip(p.visibility, p.measurements.T):
                camera = self.nvm.camera_by_id(camera_id)
                yhat = NvmModel.project_point_camera(p, camera)
                distance = np.linalg.norm(yhat - measurement)
                self.assertLessEqual(distance, max_pixel_distance)

    def test_rescale(self):
        scale_factor = np.random.uniform(0.1, 0.7)
        rescaled = NvmModel.create_rescaled(self.nvm, scale_factor)
        self.assertEqual(len(rescaled.cameras), len(self.nvm.cameras))
        self.assertEqual(len(rescaled.points), len(self.nvm.points))

        for cam_original, cam_scaled in zip(self.nvm.cameras, rescaled.cameras):
            self.assertEqual(cam_scaled.id, cam_original.id)
            self.assertEqual(cam_scaled.filename, cam_original.filename)
            self.assertEqual(cam_scaled.orientation, cam_original.orientation)
            nt.assert_almost_equal(cam_scaled.position, scale_factor * cam_original.position)

        for p_original, p_scaled in zip(self.nvm.points, rescaled.points):
            nt.assert_equal(p_scaled.color, p_original.color)
            nt.assert_equal(p_scaled.measurements, p_original.measurements)
            nt.assert_almost_equal(p_scaled.position, scale_factor * p_original.position)

        max_pixel_distance = 12.0
        for p in rescaled.points:
            for camera_id, measurement in zip(p.visibility, p.measurements.T):
                camera = rescaled.camera_by_id(camera_id)
                yhat = NvmModel.project_point_camera(p, camera)
                distance = np.linalg.norm(yhat - measurement)
                self.assertLessEqual(distance, max_pixel_distance)

    def test_extract_framenumbers(self):
        frames = self.nvm.camera_frame_numbers
        expected_frames = [397, 373, 391, 394, 389, 387, 385, 381, 375, 377,
                           363, 366, 361, 359, 357, 401, 403, 406, 414,
                           417, 419, 421, 423, 426, 431, 433 ,435, 438, 446,
                           448, 450, 453, 463, 457, 466, 470, 475, 480, 485,
                           492, 494, 497, 501, 509, 513, 519, 524, 541, 539,
                           537, 535, 543, 545, 547, 549, 551, 554, 558, 334,
                           314, 316, 331, 329, 318, 320, 324, 312, 310, 308,
                           343, 346, 350, 303, 290, 266, 263, 269, 297, 274,
                           252, 240, 242, 237, 235, 220, 217, 209, 226, 213,
                           205, 201, 222, 185, 181, 178, 159, 162, 165, 170,
                           155, 139, 128, 118, 109, 114, 105, 102,  97,  93,
                            91, 560,  90,  89, 562,  87,  68,  50,  66,  69,
                            71,  74,  73, 564,  84,  75,  81, 568, 566, 570,
                            76,  78, 572, 815, 705, 818, 785, 782, 748, 744,
                           728, 714, 721, 700, 779, 695, 687, 684, 682, 690,
                           670, 667, 661, 657, 651, 646, 643, 640, 673, 638,
                           635, 632, 625, 622, 619, 612, 608, 605, 601, 593,
                           587, 584, 582, 580, 579, 578, 577, 576, 575, 574,
                           812, 796, 843, 839, 859, 834, 809, 763, 776]

        self.assertEqual(frames, expected_frames)

    def test_rescale_walk(self):
        #TODO: Should test if scaling was actually correct not just that it *is* scaled
        rescaled = NvmModel.create_autoscaled_walk(self.nvm)

        max_pixel_distance = 12.0
        for p in rescaled.points:
            for camera_id, measurement in zip(p.visibility, p.measurements.T):
                camera = rescaled.camera_by_id(camera_id)
                yhat = NvmModel.project_point_camera(p, camera)
                distance = np.linalg.norm(yhat - measurement)
                self.assertLessEqual(distance, max_pixel_distance)

