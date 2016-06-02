from __future__ import division, print_function

import os
from collections import namedtuple
import re

import numpy as np
from imusim.maths.quaternions import Quaternion

from .dataset import Landmark
from .openmvg import SfMData

class View(object):
    __slots__ = ('id', 'time', 'position', 'orientation')

    def __init__(self, _id, time, position, orientation):
        self.id = _id
        self.time = time
        self.position = position
        self.orientation = orientation

class SfmResultError(Exception):
    pass

class SfmResult(object):
    next_view_id = 0
    next_landmark_id = 0

    @staticmethod
    def frame_from_filename(filename):
        m = re.findall(r'(\d+)\.[\w]+\b', filename)
        try:
            return int(m[-1])
        except (ValueError, IndexError):
            raise SfmResultError("Could not extract frame number from {}".format(filename))

    def __init__(self):
        self.views = []
        self.landmarks = []

    def add_view(self, time, position, orientation):
        _id = self.next_view_id
        self.next_view_id += 1

        view = View(_id, time, position, orientation)
        self.views.append(view)
        return _id

    def remap_views(self):
        remap = {}
        sorted_views = []
        for new_id, view in enumerate(sorted(self.views, key=lambda v: v.time)):
            remap[view.id] = new_id
            view.id = new_id
            sorted_views.append(view)
        self.views = sorted_views

        for lm in self.landmarks:
            remapped_observations = {
                remap[v_id] : measurement for v_id, measurement in lm.observations.items()
            }
            lm.observations = remapped_observations

    def add_landmark(self, position, visibility, color=None):
        for view_id in visibility:
            try:
                view = self.views[view_id]
            except IndexError:
                raise SfmResultError("No such view: {:d}".format(view_id))
        _id = self.next_landmark_id
        self.next_landmark_id += 1
        lm = Landmark(_id, position, visibility, color=color)
        self.landmarks.append(lm)
        return _id

    def rescaled(self, scale_factor):
        sfm_r = SfmResult()
        for view in self.views:
            new_pos = scale_factor * view.position
            v_r = View(view.id, view.time, new_pos, view.orientation)
            sfm_r.views.append(v_r)

        for lm in self.landmarks:
            new_pos = scale_factor * lm.position
            lm_r = Landmark(lm.id, new_pos, lm.observations, color=lm.color)
            sfm_r.landmarks.append(lm_r)

        return sfm_r


class VisualSfmResult(SfmResult):
    @property
    def camera_frame_numbers(self):
        return [camera.framenumber for camera in self.cameras]

    @classmethod
    def from_file(cls, filename, camera_fps, load_measurements=True):
        instance = cls()
        num_cameras = 0
        num_points = 0
        state = 'header'
        for i, line in enumerate(open(filename, 'r')):
            line = line.strip()
            if not line or line[0] == '#':
                continue

            if state == 'header':
                if line == 'NVM_V3':
                    state = 'num_cameras'
                else:
                    raise SfmResultError("Expected NVM_V3, got {}".format(line))

            elif state == 'num_cameras':
                try:
                    num_cameras = int(line)
                    state = 'cameras'
                except ValueError:
                    raise SfmResultError("Expected number of cameras, got {}".format(line))

            elif state == 'cameras':
                tokens = line.split()
                try:
                    params = map(float, tokens[-10:])
                except IndexError:
                    raise SfmResultError("Failed to parse camera on line {:d}".format(i))
                focal, qw, qx, qy, qz, px, py, pz, radial, _ = params
                filename = ''.join(tokens[:-10])
                filename = os.path.split(filename)[-1]
                q = Quaternion(qw, qx, qy, qz).conjugate # Change coordinate frame
                q.normalise()
                if not np.isclose(q.magnitude, 1.0):
                    raise SfmResultError("{} had norm {}".format(q, q.magnitude))
                p = np.array([px, py, pz])
                frame_number = cls.frame_from_filename(filename)
                t = frame_number / camera_fps
                view_id = instance.add_view(t, p, q)

                if len(instance.views) >= num_cameras:
                    state = 'num_points'

            elif state == 'num_points':
                try:
                    num_points = int(line)
                    state = 'points'
                except ValueError:
                    raise SfmResultError("Expected number of points, got {}".format(line))

            elif state == 'points':
                tokens = line.split()
                position = np.array(map(float, tokens[:3]))
                color = np.array(map(int, tokens[3:6]), dtype='uint8')
                num_meas = int(tokens[6])
                if not len(tokens) == 7 + num_meas * 4:
                    raise SfmResultError("Number of tokens: {}, expected {}".format(len(tokens), 7+num_meas*4))
                image_indices = map(int, tokens[7::4])
                visibility = image_indices
                if load_measurements:
                    meas_x = map(float, tokens[9::4])
                    meas_y = map(float, tokens[10::4])
                    #measurements = np.array(zip(meas_x, meas_y)).T
                    observations = {v_id: np.array([x, y]) for v_id, x, y
                                    in zip(image_indices, meas_x, meas_y)}
                else:
                    observations = image_indices

                lm_id = instance.add_landmark(position, observations)

                if len(instance.landmarks) >= num_points:
                    state = 'model_done'

            elif state == 'model_done':
                if not line == '0':
                    raise SfmResultError("Expected 0 to end model section, got {}".format(line))
                state = 'all_done'

            elif state == 'all_done':
                if not line == '0':
                    raise SfmResultError("Expected 0 to end file, got {}".format(line))
                state = 'finish'

            elif state == 'finish':
                if line:
                    raise SfmResultError("Expected nothing, got {}".format(line))

            else:
                raise SfmResultError("Unknown state {}".format(state))

        # Reorder views in time ascending order
        instance.remap_views()
        return instance


class OpenMvgResult(SfmResult):
    @classmethod
    def from_file(cls, filename, camera_fps, color=False):
        sfm_data = SfMData.from_json(filename, color=color)
        instance = cls()
        view_remap = {}
        for omvg_view in sfm_data.views:
            time = omvg_view.framenumber / camera_fps
            q = Quaternion.fromMatrix(omvg_view.R)
            p = omvg_view.c
            new_id = instance.add_view(time, p, q)
            view_remap[omvg_view.id] = new_id

        for s in sfm_data.structure:
            X = s.point
            observations = { view_remap[view_id] : xy for view_id, xy in s.observations.items()}
            instance.add_landmark(X, observations, color=s.color)

        # Rearrage views in ascending time order
        instance.remap_views()
        return instance
