from __future__ import print_function, division

import os
from collections import namedtuple

import numpy as np
from imusim.maths.quaternions import Quaternion, QuaternionArray

NvmCamera = namedtuple('CameraPose', ['id', 'filename', 'orientation', 'position'])
NvmPoint = namedtuple('WorldPoint', ['position', 'color', 'visibility', 'measurements'])


class NvmError(Exception):
    pass


class NvmModel(object):
    def __init__(self):
        self.cameras = []
        self.points = []
        self._camera_ids = set()
        self._camera_map = {}

    def add_camera(self, cam_id, filename, q, p):
        if cam_id in self._camera_ids:
            raise NvmError("There was already a camera with id {:d}".format(cam_id))
        camera = NvmCamera(cam_id, filename, q, p)
        self.cameras.append(camera)
        self._camera_ids.add(cam_id)
        self._camera_map[cam_id] = camera

    def add_point(self, xyz, rgb, visibility, measurements=None):
        for cam_id in visibility:
            if cam_id not in self._camera_ids:
                raise NvmError("Camera with id {:d} not in camera list".format(cam_id))
        measurements = measurements if measurements is not None else np.array([])
        self.points.append(NvmPoint(xyz, rgb, visibility, measurements))

    def camera_by_id(self, cam_id):
        return self._camera_map[cam_id]


    @classmethod
    def from_file(cls, filename, load_measurements=False):
        instance = cls()
        instance.focal = None
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
                    raise NvmError("Expected NVM_V3, got {}".format(line))

            elif state == 'num_cameras':
                try:
                    num_cameras = int(line)
                    state = 'cameras'
                except ValueError:
                    raise NvmError("Expected number of cameras, got {}".format(line))

            elif state == 'cameras':
                tokens = line.split()
                try:
                    params = map(float, tokens[-10:])
                except IndexError:
                    raise NvmError("Failed to parse camera on line {:d}".format(i))
                focal, qw, qx, qy, qz, px, py, pz, radial, _ = params
                if not instance.focal:
                    instance.focal = focal
                else:
                    if not focal == instance.focal:
                        raise ValueError("Got new focal {:.3f}, but already had {:.3f}".format(
                            focal, instance.focal
                        ))
                filename = ''.join(tokens[:-10])
                filename = os.path.split(filename)[-1]
                q = Quaternion(qw, qx, qy, qz)
                q.normalise()
                qnorm = np.linalg.norm([q.w, q.x, q.y, q.z])
                if not np.isclose(qnorm, 1.0):
                    raise ValueError("{} had norm {}".format(q, qnorm))
                p = np.array([px, py, pz])
                instance.add_camera(len(instance.cameras), filename, q, p)

                if len(instance.cameras) >= num_cameras:
                    state = 'num_points'

            elif state == 'num_points':
                try:
                    num_points = int(line)
                    state = 'points'
                except ValueError:
                    raise ValueError("Expected number of points, got {}".format(line))

            elif state == 'points':
                tokens = line.split()
                position = np.array(map(float, tokens[:3]))
                color = np.array(map(int, tokens[3:6]), dtype='uint8')
                num_meas = int(tokens[6])
                if not len(tokens) == 7 + num_meas * 4:
                    raise ValueError("Number of tokens: {}, expected {}".format(len(tokens), 7+num_meas*4))
                image_indices = map(int, tokens[7::4])
                visibility = image_indices
                if load_measurements:
                    meas_x = map(float, tokens[9::4])
                    meas_y = map(float, tokens[10::4])
                    measurements = np.array(zip(meas_x, meas_y)).T
                else:
                    measurements = []

                instance.add_point(position, color, visibility, measurements)

                if len(instance.points) >= num_points:
                    state = 'model_done'

            elif state == 'model_done':
                if not line == '0':
                    raise ValueError("Expected 0 to end model section, got {}".format(line))
                state = 'all_done'

            elif state == 'all_done':
                if not line == '0':
                    raise ValueError("Expected 0 to end file, got {}".format(line))
                state = 'finish'

            elif state == 'finish':
                if line:
                    raise ValueError("Expected nothing, got {}".format(line))

            else:
                raise ValueError("Unknown state {}".format(state))

        # Done, return normalized instance
        #instance._normalize_world()
        return instance