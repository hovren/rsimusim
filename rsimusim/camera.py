from collections import namedtuple

import numpy as np
import scipy.optimize

from imusim.platforms.base import Platform, Component
from imusim.platforms.timers import IdealTimer
from imusim.utilities.time_series import TimeSeries


class PinholeModel(object):
    def __init__(self, K, size, readout, frame_rate):
        self.K = K
        self.size = size
        self.readout = readout
        self.frame_rate = frame_rate

    @property
    def rows(self):
        return self.size[1]

    @property
    def columns(self):
        return self.size[0]

    def project(self, points):
        """Project points to image plane"""
        y = np.dot(self.K, points)
        y /= y[2]
        return y[:2]

class CameraPlatform(Platform):
    def __init__(self, camera_model, simulation=None, trajectory=None):
        self.timer = IdealTimer(self)
        self.camera = Camera(camera_model, self)

        Platform.__init__(self, simulation, trajectory)

    @property
    def components(self):
        return [self.timer, self.camera]

class Camera(Component):
    def __init__(self, camera_model, platform):
        self.current_frame = 0
        self.camera_model = camera_model
        super(Camera, self).__init__(platform)

    @property
    def frame_rate(self):
        return self.camera_model.frame_rate

    def sample(self, t):
        framenum = self.current_frame
        self.current_frame += 1
        environment = self.platform.simulation.environment
        pos = self.platform.trajectory.position(t)
        orientation = self.platform.trajectory.rotation(t)

        landmarks = environment.observe(t, pos, orientation)
        image_observations = {}
        for lm in landmarks:
            image_point, point_t = self.project_point_rs(lm.position, t)
            x, y = image_point
            if 0 <= x < self.camera_model.columns and 0 <= y < self.camera_model.rows:
                image_observations[lm.id] = image_point

        return framenum, t, image_observations

    def project_point_rs(self, X, t0):
        t_min = t0
        t_max = t0 + self.camera_model.readout
        readout_delta = self.camera_model.readout / self.camera_model.rows
        Xh = np.ones((4, 1))
        Xh[:3] = X.reshape(3,1)
        
        def point_and_time(t):
            pos = self.platform.trajectory.position(t)
            orientation = self.platform.trajectory.rotation(t)
            R = orientation.toMatrix()
            P = np.hstack((R.T, np.dot(R.T, -pos)))
            PX = np.dot(P, Xh)
            y = self.camera_model.project(PX)
            t_p = float(y[1]) * readout_delta + t_min
            return y, t_p

        opt_func = lambda t: np.abs(point_and_time(t)[1] - t)
        t = scipy.optimize.fminbound(opt_func, t_min, t_max)
        y, t = point_and_time(t)
        return y, t

class BasicCameraBehaviour(object):
    def __init__(self, camera_platform):
        self.camera_platform = camera_platform
        camera = self.camera_platform.camera
        camera.measurements = TimeSeries()
        # Start the sampling process
        timer = self.camera_platform.timer
        timer.callback = self._timer_callback
        period = 1. / camera.frame_rate
        timer.start(period, repeat=True)

    def _timer_callback(self):
        sim_time = self.camera_platform.simulation.time
        sensor_time = sim_time
        framenum, t, observations = self.camera_platform.camera.sample(sensor_time)
        assert t == sensor_time
        camera = self.camera_platform.camera
        camera.measurements.add(sensor_time, observations)


