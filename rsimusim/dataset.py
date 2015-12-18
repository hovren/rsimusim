from __future__ import print_function, division

from collections import namedtuple

import numpy as np
import crisp.rotations
import crisp.fastintegrate
from imusim.maths.quaternions import Quaternion, QuaternionArray
from imusim.utilities.time_series import TimeSeries
from imusim.trajectories.splined import \
    SplinedPositionTrajectory, SampledPositionTrajectory, \
    SampledRotationTrajectory, SplinedRotationTrajectory, \
    SampledTrajectory, SplinedTrajectory


class DatasetError(Exception):
    pass

Landmark = namedtuple('Landmark', ['position'])

class Dataset(object):
    def __init__(self):
        self._position_data = None
        self._orientation_data = None
        self.trajectory = None
        self.landmarks = []

    def position_from_nvm(self, nvm_model, frame_to_time_func=None, camera_fps=None):
        if (bool(nvm_model is None) == bool(frame_to_time_func is None)):
            raise DatasetError("Must specify frame_to_time_func OR camera_fps, not both or none of them")
        frame_time = frame_to_time_func if frame_to_time_func else lambda n: float(n) / camera_fps

        cameras = sorted(nvm_model.cameras, key=lambda c: c.framenumber)
        camera_times = np.array([frame_time(c.framenumber) for c in cameras])
        camera_pos = np.vstack([c.position for c in cameras]).T
        ts = TimeSeries(camera_times, camera_pos)
        self._position_data = ts

        self._update_trajectory()

    def orientation_from_nvm(self, nvm_model, frame_to_time_func=None, camera_fps=None):
        if (bool(nvm_model is None) == bool(frame_to_time_func is None)):
            raise DatasetError("Must specify frame_to_time_func OR camera_fps, not both or none of them")
        frame_time = frame_to_time_func if frame_to_time_func else lambda n: float(n) / camera_fps
        cameras = sorted(nvm_model.cameras, key=lambda c: c.framenumber)
        camera_times = np.array([frame_time(c.framenumber) for c in cameras])
        camera_orientations = QuaternionArray([c.orientation for c in cameras])
        camera_orientations = camera_orientations.unflipped()

        # Must resample to uniform sample time for splining to work
        camera_orientations, camera_times = resample_quaternion_array(camera_orientations, camera_times)
        ts = TimeSeries(camera_times, camera_orientations)
        self._orientation_data = ts
        self._update_trajectory()

    def landmarks_from_nvm(self, nvm_model):
        for p in nvm_model.points:
            lm = Landmark(p.position)
            self.landmarks.append(lm)

    def _update_trajectory(self):
        if self._position_data and not self._orientation_data:
            samp = SampledPositionTrajectory(self._position_data)
            self.trajectory = SplinedPositionTrajectory(samp)
        elif self._orientation_data and not self._position_data:
            samp = SampledRotationTrajectory(self._orientation_data)
            self.trajectory = SplinedRotationTrajectory(samp)
        elif self._position_data and self._orientation_data:
            samp = SampledTrajectory(self._position_data, self._orientation_data)
            self.trajectory = SplinedTrajectory(samp)

class DatasetBuilder(object):
    LANDMARK_SOURCES = ('nvm', )
    SOURCES = ('imu', ) + LANDMARK_SOURCES

    def __init__(self):
        self._nvm_model = None
        self._nvm_camera_fps = None
        self._gyro_data = None
        self._gyro_times = None

        self._orientation_source = None
        self._position_source = None
        self._landmark_source = None

    @property
    def selected_sources(self):
        return {
            'orientation' : self._orientation_source,
            'position' : self._position_source,
            'landmark' : self._landmark_source
        }

    def add_source_nvm(self, nvm, camera_fps=30.0):
        if self._nvm_model is None:
            self._nvm_model = nvm
            self._nvm_camera_fps = camera_fps
        else:
            raise DatasetError("Can only add one NVM source")

    def add_source_gyro(self, gyro_data, gyro_times):
        n, d = gyro_data.shape
        if not n == len(gyro_times):
            raise DatasetError("Gyro data and timestamps length did not match")
        if not d == 3:
            raise DatasetError("Gyro data must have shape Nx3")

        if self._gyro_data is None:
            self._gyro_data = gyro_data
            self._gyro_times = gyro_times
            dt = float(gyro_times[1] - gyro_times[0])
            if not np.allclose(np.diff(gyro_times), dt):
                raise DatasetError("Gyro samples must be uniformly sampled")
            q = crisp.fastintegrate.integrate_gyro_quaternion_uniform(gyro_data, dt)
            self._gyro_quat = QuaternionArray(q)
        else:
            raise DatasetError("Can not add multiple gyro sources")

    def set_orientation_source(self, source):
        if source in self.SOURCES:
            self._orientation_source = source
        else:
            raise DatasetError("No such source type: {}".format(source))

    def set_position_source(self, source):
        if source in self.SOURCES:
            self._position_source = source
        else:
            raise DatasetError("No such source type: {}".format(source))

    def set_landmark_source(self, source):
        if source in self.LANDMARK_SOURCES:
            self._landmark_source = source
        else:
            raise DatasetError("No such source type: {}".format(source))

    def _can_build(self):
        return self._landmark_source is not None and \
                self._orientation_source is not None and \
                self._position_source is not None

    def _imu_to_nvm_transform(self):
        def camera_time(camera):
            return camera.framenumber / self._nvm_camera_fps
        # Find common time
        tc_min, tc_max = (camera_time(self._nvm_model.cameras[i]) for i in (0, -1))
        tg_min, tg_max = (self._gyro_times[i] for i in (0, -1))
        t_min = max(tc_min, tg_min)
        t_max = min(tc_max, tg_max)
        # First valid camera
        cam_ref = min((camera for camera in self._nvm_model.cameras if camera_time(camera) >= t_min),
                      key=lambda c: np.abs(camera_time(c) - t_min))
        t_ref = camera_time(cam_ref)

        # Interpolate gyro orientation at t_ref
        qn = cam_ref.orientation
        qg = quaternion_array_interpolate(self._gyro_quat, self._gyro_times, t_ref)
        Tq = qn * qg.conjugate
        return Tq, np.array([0,0,0]).reshape(3,1)

    def build(self):
        if not self._can_build():
            raise DatasetError("Must select all sources")
        ds = Dataset()

        if all(source_value == 'nvm' for source_type, source_value in self.selected_sources.items()):
            ds.orientation_from_nvm(self._nvm_model, camera_fps=self._nvm_camera_fps)
            ds.position_from_nvm(self._nvm_model, camera_fps=self._nvm_camera_fps)
            ds.landmarks_from_nvm(self._nvm_model)
        else:
            Tq, Tp = self._imu_to_nvm_transform()
        return ds

def quaternion_slerp(q0, q1, tau):
    q0_arr = np.array([q0.w, q0.x, q0.y, q0.z])
    q1_arr = np.array([q1.w, q1.x, q1.y, q1.z])
    q_arr = crisp.rotations.slerp(q0_arr, q1_arr, tau)
    return Quaternion(*q_arr)

def quaternion_array_interpolate(qa, qtimes, t):
    i = np.flatnonzero(qtimes > t)[0]
    q0 = qa[i-1]
    q1 = qa[i]
    t0 = qtimes[i-1]
    t1 = qtimes[i]
    tau = (t - t0) / (t1 - t0)
    return quaternion_slerp(q0, q1, tau)

def resample_quaternion_array(qa, timestamps, resize=None):
    num_samples = resize if resize is not None else len(qa)
    timestamps_new = np.linspace(timestamps[0], timestamps[-1], num_samples)
    new_q = []
    unpack = lambda q: np.array([q.w, q.x, q.y, q.z])
    for t in timestamps_new:
        i = np.flatnonzero(timestamps >= t)[0]
        t1 = timestamps[i]
        if np.isclose(t1, t):
            new_q.append(qa[i])
        else:
            t0 = timestamps[i-1]
            tau = (t - t0) / (t1 - t0)
            q0 = qa[i-1]
            q1 = qa[i]
            qc = crisp.rotations.slerp(unpack(q0), unpack(q1), tau)
            q = Quaternion(qc[0], qc[1], qc[2], qc[3])
            new_q.append(q)
    return QuaternionArray(new_q), timestamps_new