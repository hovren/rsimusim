from collections import namedtuple

import numpy as np
import scipy.optimize

from imusim.platforms.base import Platform, Component
from imusim.platforms.timers import IdealTimer

ImageObservation = namedtuple('ImageObservation', ['id', 'image_point'])

class PinholeModel(object):
    def __init__(self, K, size, readout, frame_rate):
        self.K = K
        self.size = size
        self.readout = readout
        self.frame_rate = frame_rate

    @property
    def rows(self):
        return self.size[1]

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
        self.camera_model = camera_model
        super(Camera, self).__init__(platform)

    @property
    def frame_rate(self):
        return self.camera_model.frame_rate

    def sample(self, t):
        world = self.platform.simulation.environment.world

        pos = self.platform.trajectory.position(t)
        orientation = self.platform.trajectory.rotation(t)

        world_observations = world.observe(t, pos, orientation)
        image_observations = [ImageObservation(wo.id, self.project_point_rs(wo.world_point, t)[0]) for wo in world_observations]
        return image_observations

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

class DefaultCameraBehaviour(object):
    def __init__(self, camera_platform):
        self.camera_platform = camera_platform

        # Start the sampling process
        timer = self.camera_platform.timer
        timer.callback = self._timer_callback
        period = 1. / self.camera_platform.camera.frame_rate
        timer.start(period, repeat=True)

        self.timestamps = []

    def _timer_callback(self):
        sim_time = self.camera_platform.simulation.time
        sensor_time = sim_time
        samples = self.camera_platform.camera.sample(sim_time)
        #print 'See3:', samples[:3]
        self.timestamps.append(sensor_time)

