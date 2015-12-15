import unittest
import numpy.testing as nt

import rsimusim_legacy.dataset
import rsimusim_legacy.nvm

class NvmTests(unittest.TestCase):
    def setUp(self):
        self.model = nvm.NvmModel.from_file('example.nvm')

    def test_load_ok(self):
        self.assertEqual(len(self.model.points), )

def test_bounds():
    times = [1.0, 2.0, 5.0, 9.0]
    bounds = rsimusim_legacy.dataset.create_bounds(times)
    expected_bounds = [0.5, 1.5, 3.5, 7.0, 11.0]
    nt.assert_almost_equal(bounds, expected_bounds)
