import json
from collections import namedtuple
import re

import numpy as np
import cv2
import os
from imusim.utilities.time_series import TimeSeries
from imusim.trajectories.splined import SplinedPositionTrajectory, SampledPositionTrajectory


class View(namedtuple('View', ['id', 'filename', 'intrinsic', 'R', 'c'])):
    @property
    def framenumber(self):
        m = re.findall(r'(\d+)\.[\w]+\b', self.filename)
        try:
            return int(m[-1])
        except (ValueError, IndexError):
            raise ValueError("Could not extract frame number from {}".format(self.filename))


class Structure(object):
    __slots__ = ('point', 'observations', 'color')

    def __init__(self, point, observations, color=None):
        self.point = point
        self.observations = observations
        self.color = color


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
    def __init__(self):
        self.intrinsics = []
        self.views = []
        self.structure = []

    @classmethod
    def from_json(cls, sfm_data_path, color=False):
        data = json.load(open(sfm_data_path, 'r'))
        instance = cls()
        instance.intrinsics = instance._unpack_intrinsics(data)
        instance.views = instance._unpack_views(data)
        instance.structure = instance._unpack_structure(data)

        if color:
            image_path = os.path.join(os.path.split(sfm_data_path)[0],
                                      data['root_path'])
            SfMData.colorize(instance, image_path)

        return instance

    @classmethod
    def colorize(cls, instance, root_path):
        image_structure_map = {}
        for s_id, s in enumerate(instance.structure):
            view_id, image_point = s.observations.items()[0]
            view = instance.views[view_id]
            if view.filename in image_structure_map:
                image_structure_map[view.filename].append((s_id, image_point))
            else:
                image_structure_map[view.filename] = [(s_id, image_point), ]

        #img = np.zeros((1080, 1920, 3))
        for filename, structures in image_structure_map.iteritems():
            filepath = os.path.join(root_path, filename)
            img = cv2.imread(filepath)
            assert img.ndim == 3
            for s_id, image_point in structures:
                x, y = map(int, image_point)
                b, g, r = img[y, x]
                s = instance.structure[s_id]
                s.color = np.array([r, g, b], dtype='uint8')
                #img = None


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
            structure = Structure(point, observations, None)
            return structure

        return [parse_structure(d) for d in structure_data]

    @classmethod
    def create_rescaled(cls, original, scale_factor):
        rescaled = cls()

        for intr in original.intrinsics:
            new_intr = Intrinsic(*intr)
            rescaled.intrinsics.append(new_intr)

        for v in original.views:
            scaled_c = scale_factor * v.c
            scaled_intr = rescaled.intrinsics[v.intrinsic.id]
            assert scaled_intr.id == v.intrinsic.id
            scaled_v = View(v.id, v.filename, scaled_intr, v.R, scaled_c)
            rescaled.views.append(scaled_v)

        for s in original.structure:
            scaled_pt = scale_factor * s.point
            scaled_obs = {view_id : measurement for view_id, measurement in s.observations.items()}
            scaled_s = Structure(scaled_pt, scaled_obs, s.color)
            rescaled.structure.append(scaled_s)

        return rescaled

    @classmethod
    def create_autoscaled_walk(cls, sfm_data, walk_speed=1.4, camera_fps=30.0):
        views = sorted(sfm_data.views, key=lambda v: v.framenumber)
        view_times = np.array([v.framenumber / camera_fps for v in views])
        view_pos = np.vstack([v.c for v in views]).T

        timeseries = TimeSeries(view_times, view_pos)
        samp_traj = SampledPositionTrajectory(timeseries)
        splined_traj = SplinedPositionTrajectory(samp_traj)
        num_samples = 50 * (splined_traj.endTime - splined_traj.startTime)
        integration_times = np.linspace(splined_traj.startTime, splined_traj.endTime, num=num_samples)
        vel = splined_traj.velocity(integration_times)
        speed = np.linalg.norm(vel, axis=0)
        travel_time = integration_times[-1] - integration_times[0]
        dt = float(travel_time) / num_samples
        distance_traveled = np.trapz(speed, dx=dt)
        scale_factor = (walk_speed * travel_time) / distance_traveled

        return cls.create_rescaled(sfm_data, scale_factor)

