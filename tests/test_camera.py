from __future__ import print_function, division

import unittest

import numpy as np

from imusim.simulation.base import Simulation
from imusim.platforms.imus import IdealIMU
from imusim.behaviours.imu import BasicIMUBehaviour

from rsimusim.dataset import Dataset
from rsimusim.scene import SceneEnvironment
from rsimusim.camera import CameraPlatform, PinholeModel, BasicCameraBehaviour

CAMERA_MATRIX = np.array(
        [[ 850.051391602,    0.        ,  0],
         [ 0.        ,  850.051391602,  0],
     [   0.        ,    0.        ,    1.        ]]
    )


class CameraSimulationTest(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset.from_file('example_dataset.h5')

        environment = SceneEnvironment(dataset=self.ds)
        sim = Simulation(environment=environment)
        trajectory = self.ds.trajectory
        camera_model = PinholeModel(CAMERA_MATRIX, (1920, 1080), 1./35, 30.0)
        camera = CameraPlatform(camera_model, simulation=sim,
                                trajectory=trajectory)
        camera_behaviour = BasicCameraBehaviour(camera)
        imu = IdealIMU(simulation=sim, trajectory=trajectory)
        imu_behaviour = BasicIMUBehaviour(imu, 1./100) # 100Hz sample rate

        sim.time = trajectory.startTime
        self.simulation = sim
        self.camera = camera

    def test_observations(self):
        run_time = 3.0 # seconds
        stop_time = self.ds.trajectory.startTime + 3.0
        expected_frames = run_time * self.camera.camera.camera_model.frame_rate
        self.simulation.run(stop_time) # seconds
        camera = self.camera.camera

        self.assertTrue(expected_frames-1 <= len(camera.measurements) <= expected_frames+1)
        for t, observations in zip(camera.measurements.timestamps, camera.measurements.values):
            visible_landmark_ids = set(lm.id for lm in self.ds.visible_landmarks(t))
            observation_ids = set(observations.keys())
            self.assertTrue(observation_ids.issubset(visible_landmark_ids))
            for landmark_id, image_point in observations.items():
                x, y = image_point.flat
                self.assertTrue(0 <= x < camera.camera_model.columns)
                self.assertTrue(0 <= y < camera.camera_model.rows)





