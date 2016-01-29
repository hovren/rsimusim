import json
from collections import namedtuple
import re

import numpy as np


class View(namedtuple('View', ['id', 'filename', 'intrinsic', 'R', 'c'])):
    @property
    def framenumber(self):
        m = re.findall(r'(\d+)\.[\w]+\b', self.filename)
        try:
            return int(m[-1])
        except (ValueError, IndexError):
            raise ValueError("Could not extract frame number from {}".format(self.filename))

Structure = namedtuple('Structure', ['point', 'observations'])
class Intrinsic(namedtuple('Intrinsic',
                           ['id', 'focal_length', 'principal_point', 'width', 'height'])):
    @property
    def camera_matrix(self):
        f = self.focal_length
        u0, v0 = self.principal_point
        return np.array([[f, 0., u0],
                         [0., f, v0],
                         [0., 0., 1.]])

class SfMData(object):
    def __init__(self, sfm_data_path):
        data = json.load(open(sfm_data_path, 'r'))
        self.intrinsics = self._unpack_intrinsics(data)
        self.views = self._unpack_views(data)
        self.structure = self._unpack_structure(data)

    def project_point_view(self, p, view):
        intr = view.intrinsic
        K = intr.camera_matrix
        p_view = np.dot(view.R, (p - view.c))
        im_pt = np.dot(K, p_view)
        return im_pt[:2] / im_pt[2]

    def _unpack_intrinsics(self, data):
        def parse_intr(d):
            assert d['value']['polymorphic_name'] == 'pinhole'
            intr_id = d['key']
            idata = d['value']['ptr_wrapper']['data']
            focal = idata['focal_length']
            princp = idata['principal_point']
            width = idata['width']
            height = idata['height']
            return Intrinsic(intr_id, focal, princp, width, height)

        return [parse_intr(d) for d in data['intrinsics']]


    def _unpack_views(self, data):
        views_data = data['views']
        poses = self._unpack_poses(data)

        def parse_view(d):
            vdata = d['value']['ptr_wrapper']['data']
            v_id = vdata['id_view']
            i_id = vdata['id_intrinsic']
            intr = self.intrinsics[i_id]
            filename = vdata['filename']
            p_id = vdata['id_pose']
            R, c = poses[p_id]
            view = View(v_id, filename, intr, R, c)
            return view
        return [parse_view(d) for d in views_data]

    def _unpack_poses(self, data):
        poses_data = data['extrinsics']
        def parse_pose(d):
            p_id = d['key']
            pdata = d['value']
            R = np.array(pdata['rotation'])
            c = np.array(pdata['center'])
            return p_id, R, c
        return {p_id : (R, c) for p_id, R, c in (parse_pose(d) for d in poses_data)}


    def _unpack_structure(self, data):
        structure_data = data['structure']
        def parse_observation(d):
            view_id = d['key']
            image_point = np.array(d['value']['x'])
            return view_id, image_point

        def parse_structure(d):
            sdata = d['value']
            point = np.array(sdata['X'])
            observations = {view_id : pt for view_id, pt in (parse_observation(od) for od in sdata['observations'])}
            structure = Structure(point, observations)
            return structure

        return [parse_structure(d) for d in structure_data]
