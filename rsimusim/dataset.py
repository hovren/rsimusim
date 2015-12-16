from __future__ import print_function, division

import numpy as np
import crisp.rotations
from imusim.maths.quaternions import Quaternion, QuaternionArray
from imusim.utilities.time_series import TimeSeries
from imusim.trajectories.splined import \
    SplinedPositionTrajectory, SampledPositionTrajectory, \
    SampledRotationTrajectory, SplinedRotationTrajectory, \
    SampledTrajectory, SplinedTrajectory


class DatasetError(Exception):
    pass

class Dataset(object):
    def __init__(self):
        self._position_data = None
        self._orientation_data = None
        self.trajectory = None

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

        # First camera has identity rotation

        camera_orientations, camera_times = resample_quaternion_array(camera_orientations, camera_times)
        ts = TimeSeries(camera_times, camera_orientations)
        self._orientation_data = ts
        self._update_trajectory()

    def _update_trajectory(self):
        if self._position_data and not self._orientation_data:
            samp = SampledPositionTrajectory(self._position_data)
            self.trajectory = SplinedPositionTrajectory(samp)
        elif self._orientation_data and not self._position_data:
            samp = SampledRotationTrajectory(self._orientation_data)
            self.trajectory = SplinedRotationTrajectory(samp)

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