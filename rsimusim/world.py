from collections import namedtuple
import os

import numpy as np
from imusim.environment.base import Environment
from imusim.utilities.time_series import TimeSeries
from imusim.trajectories.sampled import SampledPositionTrajectory
from imusim.trajectories.splined import SplinedPositionTrajectory
from imusim.maths.quaternions import Quaternion

WorldObservation = namedtuple('WorldObservation', ['id', 'world_point'])

CameraPose = namedtuple('CameraPose', ['frame_num', 'orientation', 'position'])
WorldPoint = namedtuple('WorldPoint', ['position','color', 'visibility'])

class NvmModel(object):
    def __init__(self):
        self.cameras = []
        self.points = []

    def add_camera(self, frame_num, q, p):
        self.cameras.append(CameraPose(frame_num, q, p))

    def add_point(self, xyz, rgb, visibility):
        self.points.append(WorldPoint(xyz, rgb, visibility))

    def _normalize_world(self):
        # Sort cameras by filename
        self.cameras.sort(key=lambda c: c.frame_num)

        return

        # Rotate such that first camera is q=(1, 0, 0, 0)
        q0 = self.cameras[0].orientation
        print 'Before'
        print self.cameras[0].orientation, self.cameras[0].position
        print self.cameras[170].orientation, self.cameras[170].position
        p0 = self.cameras[0].position
        def transform_position(x, p0, q0):
            return q0.rotateFrame((x-p0).reshape(3,1)).reshape(-1)

        self.points = [WorldPoint(transform_position(p.position, p0, q0), p.color, p.visibility) for p in self.points]
        self.cameras = [CameraPose(c.frame_num, q0.conjugate * c.orientation, transform_position(c.position, p0, q0))
                        for c in self.cameras]
        print 'After'
        print self.cameras[0].orientation, self.cameras[0].position
        print self.cameras[170].orientation, self.cameras[170].position

    def scale_world(self, factor):
        print 'Scaling with', factor
        self.cameras = [CameraPose(c.frame_num, c.orientation, c.position * factor) for c in self.cameras]
        self.points = [WorldPoint(p.position * factor, p.color, p.visibility) for p in self.points]

    def autoscale_walking(self, walk_speed=1.4, camera_fps=30.0):
        camera_timestamps = np.array(self.camera_framenums) / camera_fps
        camera_timeseries = TimeSeries(camera_timestamps, self.camera_positions)
        samp_traj = SampledPositionTrajectory(camera_timeseries)
        splined_traj = SplinedPositionTrajectory(samp_traj)
        num_samples = 50 * (splined_traj.endTime - splined_traj.startTime)
        integration_times = np.linspace(splined_traj.startTime, splined_traj.endTime, num=num_samples)
        vel = splined_traj.velocity(integration_times)
        speed = np.linalg.norm(vel, axis=0)
        distance_traveled = np.sum(speed)
        travel_time = integration_times[-1] - integration_times[0]
        mean_speed = 1.4 # From WP
        scale_factor = mean_speed * travel_time / distance_traveled
        self.scale_world(scale_factor)

    @property
    def camera_positions(self):
        return np.vstack([camera.position for camera in self.cameras]).T

    @property
    def camera_framenums(self):
        return [c.frame_num for c in self.cameras]

    @property
    def world_points(self):
        return np.vstack([point.position for point in self.points]).T

    @classmethod
    def from_file(cls, filename, load_measurements=False):
        instance = cls()
        instance.focal = None
        if load_measurements:
            instance.measurements = []
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
                    raise ValueError("Expected NVM_V3, got {}".format(line))

            elif state == 'num_cameras':
                try:
                    num_cameras = int(line)
                    state = 'cameras'
                    current_camera = 0
                except ValueError:
                    raise ValueError("Expected number of cameras, got {}".format(line))

            elif state == 'cameras':
                tokens = line.split()
                try:
                    #print tokens
                    params = map(float, tokens[-10:])
                except IndexError:
                    raise ValueError("Failed to parse camera")
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
                frame_num = int(os.path.splitext(os.path.basename(filename))[0].split("_")[-1])
                #q = np.array([qw, qx, qy, qz])
                q = Quaternion(qw, qx, qy, qz)
                q.normalise()
                qnorm = np.linalg.norm([q.w, q.x, q.y, q.z])
                if not np.isclose(qnorm, 1.0):
                    raise ValueError("{} had norm {}".format(q, qnorm))
                p = np.array([px, py, pz])
                instance.add_camera(frame_num, q, p)

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
                visible_set = image_indices #set(image_indices)
                if load_measurements:
                    meas_x = np.array(map(float, tokens[9::4]))
                    meas_y = np.array(map(float, tokens[10::4]))
                    if (len(meas_x) != len(meas_y)) or (len(meas_x) != num_meas):
                        raise ValueError("Failed to load measurements")
                    meas_xy = np.vstack((meas_x, meas_y))
                    instance.measurements.append(meas_xy)
                instance.add_point(position, color, visible_set)

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
        instance._normalize_world()
        return instance

class NonBlockableWorld(object):
    def __init__(self, world_points=None):
        self.points = world_points if world_points is not None else np.array([])

    def observe(self, *args):
        return [WorldObservation(i, p) for i, p in enumerate(self.points.T)]

class NvmWorld(object):
    def __init__(self, nvm_model, time_to_frame_func):
        self.model = nvm_model
        self.time_to_frame_func = time_to_frame_func

    def observe(self, t, position, orientation):
        frame_num = self.time_to_frame_func(t)
        points = [p for p in self.model if frame_num in p.visibility]
        return points

class WorldEnvironment(Environment):
    def __init__(self, world=NonBlockableWorld(), **kwargs):
        self.world = world
        super(WorldEnvironment, self).__init__(**kwargs)

    def __call__(self, t, position, orientation):
        return self.world.observe(t, position, orientation)
