from imusim.platforms.base import Platform, Component
from imusim.platforms.timers import IdealTimer
from imusim.behaviours.timing import VirtualTimer
from imusim.environment.base import Environment

class CameraPlatform(Platform):
    def __init__(self, simulation=None, trajectory=None):
        self.timer = IdealTimer(self)
        self.camera = Camera(self)

        Platform.__init__(self, simulation, trajectory)

    @property
    def components(self):
        return [self.timer, self.camera]

class Camera(Component):
    @property
    def frame_rate(self):
        return 30.0

    def sample(self, t):
        pos = self.platform.trajectory.position(t)
        orientation = self.platform.trajectory.rotation(t)
        print 'At time {:.2f}  with position {}, and orientation {}'.format(t, pos.reshape(-1), orientation)

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
        self.timestamps.append(sensor_time)

