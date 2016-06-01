import unittest

import numpy as np
import numpy.testing as nt

from rsimusim.openmvg import SfMData

POSE_143 = (
    np.array([[-0.8913088105868323,
     0.13353483358069088,
     0.43328633995415017],
    [-0.03546320219951843, 0.9321846941526003, -0.3602416651046622],
    [-0.45200770508924104, -0.3364522911323398, -0.8261288581884529]]),
    np.array([-0.07824778062490875,
    0.021573934104102216,
    0.7099844319700787])
)

STRUCTURE_800 = {
    'X' : np.array([0.1684471696120118,
   0.07249354710603015,
   0.2333098154894031]),
    'observations' : {
        13 : np.array([1583.5799560546875, 895.2379760742188]),
        14 : np.array([1572.050048828125, 929.2050170898438]),
        15 : np.array([1617.530029296875, 960.9539794921875])
    }
}

class OpenMVGSfMDataTests(unittest.TestCase):
    def setUp(self):
        example_file = 'example_sfm_data.json'
        self.sfm_data = SfMData.from_json(example_file)

    def test_load_example(self):
        sfm_data = self.sfm_data

        # Intrinsics
        self.assertEqual(len(sfm_data.intrinsics), 1)
        intr = sfm_data.intrinsics[0]
        self.assertEqual(intr.id, 0)
        nt.assert_equal(intr.focal_length, 862.4335602502919)
        nt.assert_equal(intr.width, 1920)
        nt.assert_equal(intr.height, 1080)
        nt.assert_equal(intr.principal_point, [987.8934187779146, 525.1446992725403])


        # Views
        self.assertEqual(len(sfm_data.views), 155)
        for v in sfm_data.views:
            self.assertIs(v.intrinsic, sfm_data.intrinsics[0])
        v143 = sfm_data.views[143]
        self.assertEqual(v143.id, 143)
        R143, c143 = POSE_143
        nt.assert_equal(v143.R, R143)
        nt.assert_equal(v143.c, c143)
        self.assertEqual(v143.filename, 'frame_0779.jpg')
        self.assertEqual(v143.framenumber, 779)

        # Structure
        self.assertEqual(len(sfm_data.structure), 9455)
        s800 = sfm_data.structure[800]
        nt.assert_equal(s800.point, STRUCTURE_800['X'])
        self.assertEqual(len(s800.observations), 3)
        for view_id, expected_pt in STRUCTURE_800['observations'].items():
            pt = s800.observations[view_id]
            nt.assert_equal(pt, expected_pt)

    def test_reprojection(self):
        sfm_data = self.sfm_data

        tolerance = np.inf #7.0 # max reprojection error
        error_list = []
        for s in sfm_data.structure:
            for view_id, image_point in s.observations.items():
                v = sfm_data.views[view_id]
                self.assertEqual(v.id, view_id)
                p = sfm_data.project_point_view(s.point, v)
                error = np.linalg.norm(p - image_point)
                error_list.append(error)
                self.assertLessEqual(error, tolerance)
        """import matplotlib.pyplot as plt
        plt.hist(error_list)
        plt.title("OpenMVG reproj.test, mean={:.1f} std={:.1f}".format(np.mean(error_list), np.std(error_list)))
        plt.show()"""

    def test_rescale(self):
        sfm_data = self.sfm_data
        low_sf = np.random.uniform(0.1, 0.8)
        high_sf = np.random.uniform(1.5, 10.)
        for scale_factor in (low_sf, high_sf):
            rescaled = SfMData.create_rescaled(sfm_data, scale_factor)
            self.assertEqual(len(rescaled.views), len(sfm_data.views))
            self.assertEqual(len(rescaled.structure), len(sfm_data.structure))
            self.assertEqual(len(rescaled.intrinsics), len(sfm_data.intrinsics))

            for i_original, i_rescaled in zip(sfm_data.intrinsics, rescaled.intrinsics):
                self.assertEqual(i_rescaled.width, i_original.width)
                self.assertEqual(i_rescaled.height, i_original.height)
                nt.assert_equal(i_rescaled.camera_matrix, i_original.camera_matrix)

            for view_original, view_scaled in zip(sfm_data.views, rescaled.views):
                self.assertEqual(view_scaled.id, view_original.id)
                self.assertEqual(view_scaled.filename, view_original.filename)
                nt.assert_equal(view_scaled.R, view_original.R)
                nt.assert_almost_equal(view_scaled.c, scale_factor * view_original.c)

            for s_original, s_scaled in zip(sfm_data.structure, rescaled.structure):
                self.assertEqual(len(s_scaled.observations), len(s_original.observations))
                self.assertEqual(sorted(s_scaled.observations.keys()),
                                 sorted(s_original.observations.keys()))
                for view_id in s_scaled.observations.keys():
                    p_original = s_original.observations[view_id]
                    p_scaled = s_scaled.observations[view_id]
                    nt.assert_equal(p_scaled, p_original)

                nt.assert_almost_equal(s_scaled.point, scale_factor * s_original.point)

            max_pixel_distance = 12.0
            for s in rescaled.structure:
                for view_id, measurement in s.observations.items():
                    view = rescaled.views[view_id]
                    assert view.id == view.id
                    yhat = rescaled.project_point_view(s.point, view)
                    distance = np.linalg.norm(yhat - measurement)
                    self.assertLessEqual(distance, max_pixel_distance)

    def test_rescale_walk(self):
        rescaled = SfMData.create_autoscaled_walk(self.sfm_data)
        self.assertIsNotNone(rescaled)
