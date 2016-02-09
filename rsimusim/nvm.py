from __future__ import print_function, division

import os
import re
from collections import namedtuple

import numpy as np
from imusim.maths.quaternions import Quaternion, QuaternionArray
from imusim.utilities.time_series import TimeSeries
from imusim.trajectories.splined import SplinedPositionTrajectory, SampledPositionTrajectory
from rsimusim.sfm import SfmResult

NvmPoint = namedtuple('WorldPoint', ['position', 'color', 'visibility', 'measurements'])

class NvmCamera(namedtuple('CameraPose',
                           ['id', 'filename', 'focal', 'orientation', 'position'])):

    @property
    def framenumber(self):
        m = re.findall(r'(\d+)\.[\w]+\b', self.filename)
        try:
            return int(m[-1])
        except (ValueError, IndexError):
            raise NvmError("Could not extract frame number from {}".format(self.filename))


class NvmError(Exception):
    pass


class NvmModel(object):
    def __init__(self):
        self.cameras = []
        self.points = []
        self._camera_ids = set()
        self._camera_map = {}

    def add_camera(self, cam_id, filename, focal, q, p):
        if cam_id in self._camera_ids:
            raise NvmError("There was already a camera with id {:d}".format(cam_id))
        camera = NvmCamera(cam_id, filename, focal, q, p)
        self.cameras.append(camera)
        self._camera_ids.add(cam_id)
        self._camera_map[cam_id] = camera

    def add_point(self, xyz, rgb, visibility, measurements=None):
        for cam_id in visibility:
            if cam_id not in self._camera_ids:
                raise NvmError("Camera with id {:d} not in camera list".format(cam_id))
        measurements = measurements if measurements is not None else np.empty((2,0))
        self.points.append(NvmPoint(xyz, rgb, visibility, measurements))

    def camera_by_id(self, cam_id):
        return self._camera_map[cam_id]

    @property
    def camera_frame_numbers(self):
        return [camera.framenumber for camera in self.cameras]

    @classmethod
    def from_file(cls, filename, load_measurements=False):
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
                filename = ''.join(tokens[:-10])
                filename = os.path.split(filename)[-1]
                q = Quaternion(qw, qx, qy, qz)
                q.normalise()
                qnorm = np.linalg.norm([q.w, q.x, q.y, q.z])
                if not np.isclose(qnorm, 1.0):
                    raise ValueError("{} had norm {}".format(q, qnorm))
                p = np.array([px, py, pz])
                instance.add_camera(len(instance.cameras), filename, focal, q, p)

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
                    measurements = np.empty((2,0))

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

        return instance

    @staticmethod
    def project_point_camera(world_point, camera):
            Xw = world_point.position
            R = camera.orientation.toMatrix()
            Xc = np.dot(R, (Xw - camera.position)).reshape(3,1)
            K = np.array([[camera.focal, 0, 0],
                         [0, camera.focal, 0],
                         [0, 0, 1.]])
            y = np.dot(K, Xc)
            y /= Xc[2]
            return y[:2].flatten()

    @classmethod
    def create_rescaled(cls, nvm, scale_factor):
        rescaled = NvmModel()

        for camera in nvm.cameras:
            new_pos = scale_factor * camera.position
            rescaled.add_camera(camera.id, camera.filename, camera.focal, camera.orientation, new_pos)

        for p in nvm.points:
            new_pos = scale_factor * p.position
            rescaled.add_point(new_pos, p.color, p.visibility, p.measurements)

        return rescaled

    @classmethod
    def create_autoscaled_walk(cls, nvm, walk_speed=1.4, camera_fps=30.0):
        camera_frames = nvm.camera_frame_numbers
        timestamps = []
        positions = []
        for idx in np.argsort(camera_frames):
            camera = nvm.cameras[idx]
            frame_number = camera_frames[idx]
            timestamps.append(frame_number / camera_fps)
            positions.append(camera.position)

        timestamps = np.array(timestamps)
        positions = np.vstack(positions).T
        timeseries = TimeSeries(timestamps, positions)
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

        return cls.create_rescaled(nvm, scale_factor)


class NvmLoader(SfmResult):
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
                filename = ''.join(tokens[:-10])
                filename = os.path.split(filename)[-1]
                q = Quaternion(qw, qx, qy, qz)
                q.normalise()
                qnorm = np.linalg.norm([q.w, q.x, q.y, q.z])
                if not np.isclose(qnorm, 1.0):
                    raise ValueError("{} had norm {}".format(q, qnorm))
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

        instance.remap_views()
        return instance
