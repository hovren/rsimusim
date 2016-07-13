from __future__ import print_function, division

import bisect

import h5py
import numpy as np
from imusim.maths.quaternions import QuaternionArray
from imusim.utilities.time_series import TimeSeries
from imusim.trajectories.splined import \
    SplinedPositionTrajectory, SampledPositionTrajectory, \
    SampledRotationTrajectory, SplinedRotationTrajectory, \
    SampledTrajectory, SplinedTrajectory


class DatasetError(Exception):
    pass

class Landmark(object):
    __slots__ = ('id', '_color', 'position', 'visibility', '_observations')

    def __init__(self, _id, position, observations, color=None):
        self.id = _id
        self.position = position
        self._color = color
        self.visibility = None # Set by observation setter
        self.observations = observations

    def __repr__(self):
        return '<Landmark #{id:d} ({X[0]:.2f}, {X[1]:.2f}, {X[2]:.2f})>'.format(id=self.id, X=self.position)

    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, obs):
        if isinstance(obs, dict):
            self._observations = obs
            self.visibility = set(obs.keys())
        else:
            self._observations = {view_id : None for view_id in obs}
            self.visibility = set(obs)

    @property
    def color(self):
        if self._color is None:
            return 255 * np.ones((4,), dtype='uint8')
        else:
            r, g, b = self._color[:3]
            a = 255 if self._color.size == 3 else self._color[-1]
            return np.array([r, g, b, a], dtype='uint8')


class Dataset(object):
    def __init__(self):
        self._position_data = None
        self._orientation_data = None
        self.trajectory = None
        self.landmarks = []
        self._landmark_bounds = None
        self.name = None

    @classmethod
    def from_file(cls, filepath):
        instance = cls()

        def load_timeseries(group):
            timestamps = group['timestamps'].value
            data = group['data'].value
            if data.shape[1] == 4:
                data = QuaternionArray(data)
            return TimeSeries(timestamps, data)

        with h5py.File(filepath, 'r') as h5f:
            instance.name = h5f.attrs['name']
            instance._position_data = load_timeseries(h5f['position'])
            instance._orientation_data = load_timeseries(h5f['orientation'])
            instance._update_trajectory()

            landmarks_group = h5f['landmarks']
            instance._landmark_bounds = landmarks_group['visibility_bounds'].value
            positions = landmarks_group['positions'].value
            colors = landmarks_group['colors'].value
            landmark_keys = list(landmarks_group['visibility'].keys())
            landmark_keys.sort(key=lambda key: int(key))
            for lm_key in landmark_keys:
                lm_id = int(lm_key)
                p = positions[lm_id]
                color = colors[lm_id]
                visibility = set(list(landmarks_group['visibility'][lm_key].value))
                lm = Landmark(lm_id, p, visibility, color=color)
                instance.landmarks.append(lm)
        instance.landmarks = sorted(instance.landmarks, key=lambda lm: lm.id)

        return instance

    def visible_landmarks(self, t):
        i = bisect.bisect_left(self._landmark_bounds, t)
        interval_id = i - 1
        return [lm for lm in self.landmarks if interval_id in lm.visibility]

    def _update_trajectory(self):
        smooth_rotations = False
        if self._position_data and not self._orientation_data:
            samp = SampledPositionTrajectory(self._position_data)
            self.trajectory = SplinedPositionTrajectory(samp)
        elif self._orientation_data and not self._position_data:
            samp = SampledRotationTrajectory(self._orientation_data)
            self.trajectory = SplinedRotationTrajectory(samp, smoothRotations=smooth_rotations)
        elif self._position_data and self._orientation_data:
            samp = SampledTrajectory(self._position_data, self._orientation_data)
            self.trajectory = SplinedTrajectory(samp, smoothRotations=smooth_rotations)