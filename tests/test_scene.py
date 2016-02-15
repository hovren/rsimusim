from __future__ import print_function, division

import unittest

import numpy as np

from rsimusim.dataset import Dataset, DatasetBuilder
from rsimusim.sfm import VisualSfmResult
from rsimusim.scene import SceneEnvironment

def bound_index(t, boundaries):
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i:i+2]
        if a < t <= b:
            return i
    raise ValueError("No such bound")

class MiscSceneTests(unittest.TestCase):
    def test_bound_index(self):
        bounds = [-np.inf, 1.0, 2.0, 4.0, np.inf]
        data = ((0.5, 0), (1.0, 0), (1.5,  1))
        for t, expected_bound in data:
            bound = bound_index(t, bounds)
            self.assertEqual(bound, expected_bound)

class SceneTests(unittest.TestCase):
    def setUp(self):
        db = DatasetBuilder()
        sfm = VisualSfmResult.from_file('example.nvm', 30, load_measurements=True)
        db.add_source_sfm(sfm)
        db.set_position_source('sfm')
        db.set_orientation_source('sfm')
        db.set_landmark_source('sfm')

        self.ds = db.build()

    def test_visibility(self):
        env = SceneEnvironment(dataset=self.ds)

        times = np.random.uniform(self.ds.trajectory.startTime, self.ds.trajectory.endTime, size=10)
        for t in times:
            expected_visible = [lm.id for lm in self.ds.landmarks if bound_index(t, self.ds._landmark_bounds) in lm.visibility]
            obs = env.observe(t, None, None) # Position and orientation is not currently used
            obs_ids = [o.id for o in obs]
            self.assertEqual(sorted(expected_visible), sorted(obs_ids))

