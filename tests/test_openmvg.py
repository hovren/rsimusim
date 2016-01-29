import unittest

import numpy as np
import numpy.testing as nt

from rsimusim.openmvg_io import SfMData

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
    def test_load_example(self):
        example_file = 'example_sfm_data.json'
        sfm_data = SfMData(example_file)

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
        example_file = 'example_sfm_data.json'
        sfm_data = SfMData(example_file)
        tolerance = 7.0 # max reprojection error
        for s in sfm_data.structure:
            for view_id, image_point in s.observations.items():
                v = sfm_data.views[view_id]
                self.assertEqual(v.id, view_id)
                p = sfm_data.project_point_view(s.point, v)
                error = np.linalg.norm(p - image_point)
                self.assertLessEqual(error, tolerance)
