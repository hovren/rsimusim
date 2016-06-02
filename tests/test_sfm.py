from __future__ import division, print_function

import unittest

import numpy.testing as nt
import numpy as np

from rsimusim.sfm import SfmResult, SfmResultError, VisualSfmResult, OpenMvgResult
from imusim.maths.quaternions import Quaternion

CAMERA_FPS = 30.

class SfmLoaderTests(unittest.TestCase):
    def setUp(self):
        self.sfm = SfmResult()

    def test_add_view(self):
        id1 = self.sfm.add_view(0, 0, 0)
        id2 = self.sfm.add_view(0, 0, 0)
        self.assertNotEqual(id1, id2)
        self.assertEqual(len(self.sfm.views), 2)
        self.assertEqual(self.sfm.views[id1].id, id1)
        self.assertEqual(self.sfm.views[id2].id, id2)

    def test_add_landmark_no_view(self):
        self.sfm.add_view(0, 0, 0) # view_id = 0
        self.sfm.add_view(0, 0, 0) # view_id = 1
        pos = 0
        visibility = [0, 1, 2]
        with self.assertRaises(SfmResultError):
            self.sfm.add_landmark(pos, visibility)

    def test_add_landmark(self):
        self.sfm.add_view(0, 0, 0)
        self.sfm.add_view(0, 0, 0)
        self.sfm.add_landmark(0, [0, 1])
        self.sfm.add_landmark(0, [0])
        self.sfm.add_landmark(0, [1])
        self.assertEqual(len(self.sfm.landmarks), 3)


class LoaderTestsMixin(object):
    def test_num_cameras_loaded(self):
        self.assertEqual(len(self.sfm.views), self.NUM_VIEWS)

    def test_num_landmarks_loaded(self):
        self.assertEqual(len(self.sfm.landmarks), self.NUM_LANDMARKS)

    def test_view_data(self):
        for orig_id, (pos, rot) in self.VIEW_TEST_DATA.items():
            v_id = self.VIEW_REMAP[orig_id]
            view = self.sfm.views[v_id]
            self.assertEqual(view.id, v_id)
            nt.assert_almost_equal(view.position, pos)
            # Orientation is conjugated due to different coordinate frames
            dq = view.orientation.conjugate - rot
            self.assertAlmostEqual(dq.magnitude, 0, places=5)

    def test_view_time_order(self):
        ids = [v.id for v in self.sfm.views]
        times = [v.time for v in self.sfm.views]
        ids_diff = np.diff(ids)
        times_diff = np.diff(times)
        self.assertTrue(np.all(ids_diff == 1))
        self.assertTrue(np.all(times_diff > 0))

    def test_landmark_handpicked(self):
        for lm_id, (pos, observations) in self.LANDMARK_TEST_DATA.items():
            lm = self.sfm.landmarks[lm_id]
            self.assertEqual(lm.id, lm_id)
            nt.assert_almost_equal(lm.position, pos)
            remapped_obs = {self.VIEW_REMAP[v_id] : measurement for v_id, measurement in observations.items()}
            self.assertEqual(sorted(lm.visibility), sorted(remapped_obs.keys()))
            X = np.array(pos).reshape(3,1)
            for view_id, measurement in remapped_obs.items():
                view = self.sfm.views[view_id]
                Rws = view.orientation.toMatrix()
                pws = view.position.reshape(3,1)
                y_proj = np.dot(self.CAMERA_MATRIX, np.dot(Rws.T, X - pws))
                y_proj = y_proj[:2] / y_proj[2,0]
                y_proj = y_proj.reshape(2,1)
                y_expected = np.array(measurement).reshape(2,1)
                distance = np.linalg.norm(y_proj - y_expected)
                self.assertLess(distance, 10.0)

    def test_landmark_all(self):
        distance_list = []
        for lm in self.sfm.landmarks:
            X = lm.position.reshape(3,1)
            for view_id, measurement in lm.observations.items():
                view = self.sfm.views[view_id]
                Rws = view.orientation.toMatrix()
                pws = view.position.reshape(3,1)
                y_proj = np.dot(self.CAMERA_MATRIX, np.dot(Rws.T, X - pws))
                y_proj = y_proj[:2] / y_proj[2,0]
                y_proj = y_proj.reshape(2,1)
                y_expected = np.array(measurement).reshape(2,1)
                distance = np.linalg.norm(y_proj - y_expected)
                distance_list.append(distance)
        #import matplotlib.pyplot as plt
        #plt.hist(distance_list)
        #plt.title(self.__class__.__name__)
        #plt.show()
        self.assertLess(np.mean(distance_list), 10)

    def test_rescale(self):
        scales = [0.1, 10.0]
        for scale_factor in scales:
            sfm_r = self.sfm.rescaled(scale_factor)

            self.assertEqual(len(sfm_r.views), len(self.sfm.views))
            for v, v_r in zip(self.sfm.views, sfm_r.views):
                self.assertEqual(v_r.id, v.id)
                self.assertEqual(v_r.time, v.time)
                nt.assert_almost_equal(v_r.position, scale_factor * v.position)
                nt.assert_equal(v_r.orientation.components, v.orientation.components)

            self.assertEqual(len(sfm_r.landmarks), len(self.sfm.landmarks))
            for lm, lm_r in zip(self.sfm.landmarks, sfm_r.landmarks):
                self.assertEqual(lm_r.id, lm.id)
                nt.assert_almost_equal(lm_r.position, scale_factor * lm.position)
                self.assertEqual(sorted(lm_r.visibility), sorted(lm.visibility))


class NvmLoaderTests(LoaderTestsMixin, unittest.TestCase):
    EXAMPLE_NVM_FILE = 'example.nvm'
    NUM_VIEWS = 188
    NUM_LANDMARKS = 2682
    CAMERA_MATRIX = np.array(
        [[ 850.051391602,    0.        ,  0],
         [ 0.        ,  850.051391602,  0],
     [   0.        ,    0.        ,    1.        ]]
    )
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
        self.sfm = VisualSfmResult.from_file(self.EXAMPLE_NVM_FILE, CAMERA_FPS)

class OpenMvgLoaderTests(LoaderTestsMixin, unittest.TestCase):
    EXAMPLE_JSON_FILE = 'example_sfm_data.json'
    NUM_VIEWS = 155
    NUM_LANDMARKS = 9455
    CAMERA_MATRIX = np.array(
            [[ 862.43356025,    0.        ,  987.89341878],
       [   0.        ,  862.43356025,  525.14469927],
       [   0.        ,    0.        ,    1.        ]])
    VIEW_TEST_DATA = {6: ([-0.0066135463687062891, 0.013110920711255216, 0.0054072737344219874],
                          Quaternion(0.9599244193709499, 0.23327067225011502, 0.06151246987361936, 0.14263982127502603)),
                      2: ([-0.0066137870300444221, 0.013110339773949207, 0.0054072891053267057],
                          Quaternion(0.9971081514035066, 0.07364772448126028, -0.015545974958163908, -0.010472332405643518)),
                      133: ([0.34388455081445496, 0.12327583235340298, 1.2324705149816222],
                            Quaternion(0.32847610220097523, 0.03320448390508638, 0.9162407892228537, -0.2269443295077749)),
                      122: ([0.74123424111363245, 0.2113942275019377, 1.6924068726636425],
                            Quaternion(0.32957241455083724, 0.03125572389356408, 0.9310795494453806, -0.15328397142430536))
                      }

    LANDMARK_TEST_DATA = {
        110: ([-1.8301917253379576, 0.17858422733881618, 1.7575707358917012],
  {0: [88.970703125, 597.48699951171875],
   1: [68.536598205566406, 565.15899658203125],
   3: [49.993400573730469, 335.427001953125],
   4: [134.62800598144531, 260.7550048828125],
   13: [115.16699981689453, 437.16000366210938],
   14: [57.579601287841797, 535.92498779296875],
   16: [90.096099853515625, 589.989013671875],
   25: [57.378101348876953, 558.448974609375],
   26: [232.10800170898438, 533.4110107421875],
   29: [274.84298706054688, 566.55902099609375]}),
 3800: ([0.17614115740727326, 0.36611671358235287, 1.5322134980217552],
  {58: [1225.239990234375, 884.6619873046875],
   59: [1219.1199951171875, 958.041015625],
   60: [1055.1400146484375, 949.46197509765625],
   61: [1009.1799926757812, 962.10198974609375],
   63: [923.56298828125, 1060.9200439453125]}),
 4874: ([0.6931900643748532, 0.36680427728932175, 1.6509404810136892],
  {61: [1749.4599609375, 926.48199462890625],
   71: [1414.1500244140625, 856.9110107421875],
   72: [1389.5699462890625, 846.447998046875]}),
 5470: ([-0.023667470986766217, 0.46416957579865487, 3.1836766325835555],
  {74: [311.75399780273438, 616.3480224609375],
   80: [322.13699340820312, 531.11199951171875],
   81: [124.52999877929688, 501.6300048828125],
   84: [13.078900337219238, 491.76300048828125]}),
 5529: ([2.7260641913540993, -0.083104783854676012, 5.5843447910482213],
  {54: [1497.010009765625, 393.93600463867188],
   56: [1394.0999755859375, 418.01400756835938],
   59: [1708.77001953125, 472.87399291992188],
   63: [1329.8599853515625, 516.76898193359375],
   68: [1423.9300537109375, 424.32699584960938],
   82: [594.95001220703125, 440.81100463867188],
   85: [901.7230224609375, 438.49398803710938],
   86: [703.49798583984375, 442.0369873046875]}),
 6543: ([1.7826277020717853, -0.11553762349330475, 2.6822758354074163],
  {97: [1385.8599853515625, 261.66400146484375],
   98: [1425.239990234375, 277.48599243164062],
   99: [1480.3699951171875, 287.7550048828125],
   100: [1477.3599853515625, 290.3699951171875],
   101: [1419.949951171875, 299.8070068359375],
   102: [1268.4200439453125, 304.7919921875],
   105: [880.9229736328125, 267.33499145507812],
   106: [716.08001708984375, 262.72799682617188],
   107: [476.38800048828125, 331.8280029296875],
   108: [414.32699584960938, 320.77999877929688],
   109: [370.92999267578125, 334.29000854492188],
   110: [234.22500610351562, 321.927001953125],
   111: [120.72000122070312, 285.86300659179688]}),
 6967: ([1.185226833635985, 0.43459029155665507, 2.0353546432899656],
  {96: [1746.25, 969.9940185546875],
   97: [1831.68994140625, 981.5689697265625],
   104: [1264.25, 1014.0999755859375],
   105: [1144.5400390625, 992.63897705078125],
   106: [981.8480224609375, 1001.0999755859375]}),
 7416: ([0.42946526292622439, 0.31996086704851645, 1.2091637549430783],
  {116: [1230.9599609375, 784.20599365234375],
   117: [1148.9000244140625, 784.10797119140625],
   119: [1108.8900146484375, 887.697998046875],
   120: [1062.68994140625, 912.8280029296875],
   121: [984.72198486328125, 898.9949951171875],
   122: [936.14501953125, 962.80902099609375]})
    }

    VIEW_REMAP = {i : i for i in range(NUM_VIEWS)} # Already ordered

    def setUp(self):
        self.sfm = OpenMvgResult.from_file(self.EXAMPLE_JSON_FILE, CAMERA_FPS)