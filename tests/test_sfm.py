from __future__ import division, print_function

import unittest

import numpy.testing as nt
import numpy as np

from rsimusim.nvm import NvmLoader
from rsimusim.sfm import SfmResult, SfmResultError
from imusim.maths.quaternions import Quaternion

CAMERA_FPS = 30.

class SfmLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = SfmResult()

    def test_add_view(self):
        id1 = self.loader.add_view(0, 0, 0)
        id2 = self.loader.add_view(0, 0, 0)
        self.assertNotEqual(id1, id2)
        self.assertEqual(len(self.loader.views), 2)
        self.assertEqual(self.loader.views[id1].id, id1)
        self.assertEqual(self.loader.views[id2].id, id2)

    def test_add_landmark_no_view(self):
        self.loader.add_view(0, 0, 0) # view_id = 0
        self.loader.add_view(0, 0, 0) # view_id = 1
        pos = 0
        visibility = [0, 1, 2]
        with self.assertRaises(SfmResultError):
            self.loader.add_landmark(pos, visibility)

    def test_add_landmark(self):
        self.loader.add_view(0, 0, 0)
        self.loader.add_view(0, 0, 0)
        self.loader.add_landmark(0, [0, 1])
        self.loader.add_landmark(0, [0])
        self.loader.add_landmark(0, [1])
        self.assertEqual(len(self.loader.landmarks), 3)


class LoaderTestsMixin(object):
    def test_num_cameras_loaded(self):
        self.assertEqual(len(self.loader.views), self.NUM_VIEWS)

    def test_num_landmarks_loaded(self):
        self.assertEqual(len(self.loader.landmarks), self.NUM_LANDMARKS)

    def test_view_data(self):
        for orig_id, (pos, rot) in self.VIEW_TEST_DATA.items():
            v_id = self.VIEW_REMAP[orig_id]
            view = self.loader.views[v_id]
            self.assertEqual(view.id, v_id)
            nt.assert_almost_equal(view.position, pos)
            dq = view.orientation - rot
            self.assertAlmostEqual(dq.magnitude, 0, places=5)

    def test_view_time_order(self):
        ids = [v.id for v in self.loader.views]
        times = [v.time for v in self.loader.views]
        ids_diff = np.diff(ids)
        times_diff = np.diff(times)
        self.assertTrue(np.all(ids_diff == 1))
        self.assertTrue(np.all(times_diff > 0))

    def test_landmark_data(self):
        for lm_id, (pos, observations) in self.LANDMARK_TEST_DATA.items():
            lm = self.loader.landmarks[lm_id]
            self.assertEqual(lm.id, lm_id)
            nt.assert_almost_equal(lm.position, pos)
            remapped_obs = {self.VIEW_REMAP[v_id] : measurement for v_id, measurement in observations.items()}
            self.assertEqual(sorted(lm.visibility), sorted(remapped_obs.keys()))
            for lm_id, measurement in observations.items():
                nt.assert_almost_equal(lm.observations[lm_id], measurement)

class NvmLoaderTests(LoaderTestsMixin, unittest.TestCase):
    EXAMPLE_NVM_FILE = 'example.nvm'
    NUM_VIEWS = 188
    NUM_LANDMARKS = 2682
    VIEW_TEST_DATA = {
        0: ([-0.0304724108428, -0.0147310020402, 0.00404673162848],
             Quaternion(0.992442210987, -0.0560751916672, -0.105804611464, -0.0268352282614)),

        40: ([1.09610497952, -0.0696913599968, 3.22010850906],
              Quaternion(0.984561221356, -0.0532728451582, -0.166185207461, 0.013549630795))
    }

    LANDMARK_TEST_DATA = {
        0: ([1.22031700611, 0.769853770733, 3.58396601677],
             {0: [123.318847656, 266.784362793],
              1: [534.956176758, 364.917297363],
              2: [250.174804688, 280.94128418],
              3: [177.635498047, 272.447937012],
              4: [330.296508789, 285.646850586],
              5: [459.054321289, 299.74609375],
              6: [588.260498047, 308.647033691],
              7: [734.854736328, 390.82421875],
              8: [604.747802734, 435.141479492],
              9: [669.421630859, 482.711181641],
              10: [424.889404297, 358.722900391],
              11: [349.652099609, 328.390380859],
              12: [472.379516602, 342.169616699],
              13: [540.979858398, 321.333312988],
              15: [248.307250977, 267.757263184],
              17: [422.686035156, 368.409118652],
              18: [333.8359375, 378.549072266],
              19: [196.943115234, 356.749572754],
              22: [-103.813964844, 388.336303711],
              23: [-183.988220215, 402.432861328],
              58: [466.869995117, 214.889709473],
              59: [296.767456055, 197.85546875],
              61: [526.762084961, 228.758117676],
              62: [564.068969727, 256.866088867],
              63: [503.201171875, 186.373291016],
              64: [588.295288086, 211.475585938],
              65: [650.608886719, 245.094116211],
              66: [196.457275391, 192.248779297],
              67: [119.320556641, 178.516479492],
              68: [32.5877685547, 165.536621094],
              69: [658.449707031, 230.507141113],
              71: [799.218383789, 348.678527832],
              73: [-13.3076782227, 256.155517578],
              74: [115.8125, 267.198425293],
              75: [209.794921875, 296.114440918],
              76: [18.6792602539, 284.726318359],
              81: [193.463256836, 219.585693359],
              87: [508.649536133, 239.822387695],
              91: [439.633544922, 227.285888672]}),

        1236: ([1.03074204922, 1.11799681187, -6.69829416275],
                 {137: [-160.770263672, 339.70904541],
                  138: [-228.46282959, 389.761169434],
                  139: [-260.504638672, 350.328063965],
                  140: [-218.321350098, 326.826416016],
                  141: [-147.86505127, 323.537475586],
                  142: [-202.450073242, 361.780883789],
                  144: [-96.3673095703, 315.953979492],
                  145: [-187.434082031, 372.191955566],
                  146: [-291.932189941, 355.8671875],
                  147: [-378.505493164, 353.123352051],
                  148: [-108.198364258, 358.3984375],
                  149: [-432.921081543, 298.705810547],
                  150: [-335.725036621, 315.267089844],
                  151: [-218.79095459, 320.045593262],
                  152: [-237.279296875, 314.228942871],
                  157: [-510.5390625, 339.002380371]})

    }

    VIEW_REMAP = {0: 81, 1: 72, 2: 79, 3: 80, 4: 78, 5: 77, 6: 76, 7: 75, 8: 73, 9: 74, 10: 70, 11: 71, 12: 69, 13: 68, 14: 67, 15: 82,
                  16: 83, 17: 84, 18: 85, 19: 86, 20: 87, 21: 88, 22: 89, 23: 90, 24: 91, 25: 92, 26: 93, 27: 94, 28: 95, 29: 96, 30: 97,
                  31: 98, 32: 100, 33: 99, 34: 101, 35: 102, 36: 103, 37: 104, 38: 105, 39: 106, 40: 107, 41: 108, 42: 109, 43: 110,
                  44: 111, 45: 112, 46: 113, 47: 117, 48: 116, 49: 115, 50: 114, 51: 118, 52: 119, 53: 120, 54: 121, 55: 122, 56: 123,
                  57: 124, 58: 63, 59: 56, 60: 57, 61: 62, 62: 61, 63: 58, 64: 59, 65: 60, 66: 55, 67: 54, 68: 53, 69: 64, 70: 65, 71: 66,
                  72: 52, 73: 50, 74: 47, 75: 46, 76: 48, 77: 51, 78: 49, 79: 45, 80: 43, 81: 44, 82: 42, 83: 41, 84: 38, 85: 37, 86: 35,
                  87: 40, 88: 36, 89: 34, 90: 33, 91: 39, 92: 32, 93: 31, 94: 30, 95: 26, 96: 27, 97: 28, 98: 29, 99: 25, 100: 24, 101: 23,
                  102: 22, 103: 20, 104: 21, 105: 19, 106: 18, 107: 17, 108: 16, 109: 15, 110: 125, 111: 14, 112: 13, 113: 126, 114: 12,
                  115: 2, 116: 0, 117: 1, 118: 3, 119: 4, 120: 6, 121: 5, 122: 127, 123: 11, 124: 7, 125: 10, 126: 129, 127: 128, 128: 130,
                  129: 8, 130: 9, 131: 131, 132: 182, 133: 168, 134: 183, 135: 178, 136: 177, 137: 173, 138: 172, 139: 171, 140: 169, 141: 170,
                  142: 167, 143: 176, 144: 166, 145: 164, 146: 163, 147: 162, 148: 165, 149: 160, 150: 159, 151: 158, 152: 157, 153: 156,
                  154: 155, 155: 154, 156: 153, 157: 161, 158: 152, 159: 151, 160: 150, 161: 149, 162: 148, 163: 147, 164: 146, 165: 145,
                  166: 144, 167: 143, 168: 142, 169: 141, 170: 140, 171: 139, 172: 138, 173: 137, 174: 136, 175: 135, 176: 134, 177: 133,
                  178: 132, 179: 181, 180: 179, 181: 186, 182: 185, 183: 187, 184: 184, 185: 180, 186: 174, 187: 175}

    def setUp(self):
        self.loader = NvmLoader.from_file(self.EXAMPLE_NVM_FILE, CAMERA_FPS)
