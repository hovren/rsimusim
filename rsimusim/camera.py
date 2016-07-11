from collections import namedtuple
import multiprocessing
import logging

import numpy as np
import scipy.optimize

from imusim.platforms.base import Platform, Component
from imusim.platforms.timers import IdealTimer
from imusim.utilities.time_series import TimeSeries

USE_MULTIPROC = True

logger = logging.getLogger()

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

    def unproject(self, image_points):
        xh = np.ones((3, image_points.shape[1]))
        xh[:2] = image_points
        Y = np.dot(np.linalg.inv(self.K), xh)
        Y /= Y[2]
        return Y

class CameraPlatform(Platform):
    def __init__(self, camera_model, Rci=np.eye(3), pci=np.zeros((3,1)), simulation=None, trajectory=None):
        self.timer = IdealTimer(self)
        self.camera = Camera(camera_model, Rci, pci, self)
        Platform.__init__(self, simulation, trajectory)

    @property
    def components(self):
        return [self.timer, self.camera]

def project_at_time(t, X, Rci, pci, trajectory, camera_model):
    # Spline->World (really IMU->World) transforms
    Rws = np.array(trajectory.rotation(t).toMatrix())
    pws = trajectory.position(t)

    # Landmark in IMU frame
    X_imu = np.dot(Rws.T, (X.reshape(3,1) - pws))

    # Landmark in Camera frame
    X_camera = np.dot(Rci, X_imu) + pci

    y = camera_model.project(X_camera)
    return y, X_camera

def _project_point_rs(X, t0, camera_model, Rci, pci, trajectory):
    def root_func(r):
        t = t0 + r * camera_model.readout / camera_model.rows
        (u, v), X_camera = project_at_time(t, X, Rci, pci, trajectory, camera_model)
        if X_camera[2] < 0:
            print("Behind camera", X.ravel())
        return v - r

    try:
        v = scipy.optimize.brentq(root_func, 0, camera_model.rows, xtol=0.5)
    except ValueError:
        return None, None
    vt = t0 + v * camera_model.readout / camera_model.rows
    y, _ = project_at_time(vt, X, Rci, pci, trajectory, camera_model)
    return y, vt

def projection_worker(camera_model, Rci, pci, trajectory, inq, outq):
    logger.debug("Worker process (pid=%d) started", multiprocessing.current_process().pid)
    while True:
        object = inq.get()
        if object is None:
            inq.put(object)
            break # Stop processing
        lm_id, lm_pos, t = object
        image_point, _ = _project_point_rs(lm_pos, t, camera_model, Rci, pci, trajectory)
        outq.put((lm_id, image_point))
    logger.debug("Worker process (pid=%d) quit normally", multiprocessing.current_process().pid)

class Camera(Component):
    def __init__(self, camera_model, Rci, pci, platform):
        self.current_frame = 0
        self.camera_model = camera_model
        self.Rci = Rci
        self.pci = pci

        if USE_MULTIPROC:
            self.inq = multiprocessing.Queue()
            self.outq = multiprocessing.Queue()
            self.procs = []

        super(Camera, self).__init__(platform)

    def __del__(self):
        if USE_MULTIPROC:
            self.stop_multiproc()

    def start_multiproc(self):
        if not USE_MULTIPROC:
            raise RuntimeError("Multiprocessing is turned off in code")
        args = (self.camera_model, self.Rci, self.pci, self.platform.trajectory, self.inq, self.outq)
        self.procs = [ multiprocessing.Process(target=projection_worker, args=args)
                          for _ in range(multiprocessing.cpu_count()) ]
        for proc in self.procs:
            proc.daemon = True # Kill process on parent exit
            proc.start()
        logger.info('Started %d worker processes', len(self.procs))

    def stop_multiproc(self, kill=False):
        logger.debug("Signalling worker processes to stop")
        self.inq.put(None) # Signal done
        for proc in self.procs:
            if kill:
                proc.terminate()
            else:
                proc.join()
        logger.debug("All worker processes has quit")


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
        if USE_MULTIPROC:
            if not self.procs:
                self.start_multiproc()
            image_observation_list = []
            for lm in landmarks:
                self.inq.put((lm.id, lm.position, t))

            not_seen = set([lm.id for lm in landmarks])
            while not_seen:
                lm_id, im_pt = self.outq.get()
                if im_pt is not None:
                    image_observation_list.append((lm_id, im_pt))
                not_seen.remove(lm_id)

        else:
            image_observation_list = [(lm.id, self.project_point_rs(lm.position, t)[0]) for lm in landmarks]
        image_observations = {lm_id : image_point for lm_id, image_point in image_observation_list
                              if 0 <= image_point[0] <= self.camera_model.columns \
                                and 0 <= image_point[1] < self.camera_model.rows}

        return framenum, t, image_observations

    def project_point_rs(self, X, t0):
        return _project_point_rs(X, t0, self.camera_model, self.platform.trajectory)

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


