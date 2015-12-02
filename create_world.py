from __future__ import division
from collections import namedtuple

import cv2
import numpy as np
from crisp import VideoStream, GyroStream, AtanCameraModel
from crisp.rotations import axis_angle_to_rotation_matrix, slerp
from crisp.l3g4200d import post_process_L3G4200D_data
from crisp.tracking import track_points

camera_model = AtanCameraModel.from_hdf('/home/hannes/Code/crisp/hero3_atan.hdf')

class GyroStreamWithHelpers(GyroStream):
    def rotation_at(self, sample, rate):
        dt = float(1 / rate) # Must be float and not numpy.float64 for integrate to work
        self.integrate(dt)
        i0 = int(np.floor(sample))
        tau = sample - i0
        q = self._GyroStream__last_q
        return slerp(q[i0], q[i0 + 1], tau)

video = VideoStream.from_file(camera_model, '/home/hannes/Datasets/gopro-gyro-dataset/walk.MP4')
gyro = GyroStreamWithHelpers.from_csv('/home/hannes/Datasets/gopro-gyro-dataset/walk_gyro.csv')

def load_ground_truth(fname):
    labels = ['Fg', 'offset', 'rot_x', 'rot_y', 'rot_z', 'gbias_x', 'gbias_y', 'gbias_z']
    X = np.loadtxt(fname, delimiter=',')
    d = {
        label : value for label, value in zip(labels, X)
    }
    return d

def correct_with_ground_truth(data, gt_dict):
    r = np.array([gt_dict['rot_{}'.format(axis)] for axis in 'xyz'])
    theta = np.linalg.norm(r)
    v = r / theta
    R = axis_angle_to_rotation_matrix(v, theta)
    bias = np.array([gt_dict['gbias_{}'.format(axis)] for axis in 'xyz']).reshape(3,1)
    data_gt = R.dot(data.T - bias).T
    return data_gt

# Remove bad frequencies in L3GD gyroscope
gyro.data = post_process_L3G4200D_data(gyro.data.T).T

# Apply ground truth rotation and bias
gt_dict = load_ground_truth('/home/hannes/Datasets/gopro-gyro-dataset/walk_reference.csv')
gyro.data = correct_with_ground_truth(gyro.data, gt_dict)

TrackItem = namedtuple('TrackItem', ['frame_num', 'point'])

class Tracker(object):
    def __init__(self):
        self.tracks = []
        self._current = None
        self._current_index = None
        self._last_frame = None

    def update(self, frame, frame_num):
        if self._current is None:
            new_points = cv2.goodFeaturesToTrack(frame, 300, 0.07, 10)
            new_points.reshape(-1, 2)
        else:
            new_points = []
            # Track current points
            [_points, status, err] = cv2.calcOpticalFlowPyrLK(self._last_frame,
                                                              frame,
                                                              self._current.reshape(-1,1,2),
                                                              np.array([]))

            if _points is not None:
                _points = _points.reshape(-1, 2)

                valid = np.flatnonzero(status == 1)
                for idx in valid:
                    track_idx = self._current_index[idx]
                    pt = _points[idx]
                    self.tracks[track_idx].append(TrackItem(frame_num, pt))

        # Append new points
        for p in new_points:
            item = TrackItem(frame_num, p)
            self.tracks.append([item])

        self._update_current(frame_num)
        self._last_frame = frame

    def _update_current(self, frame_num):
        is_visible = lambda track, k: True if [ti for ti in track if ti.frame_num == k] else False
        visible = [i for i, t in enumerate(self.tracks) if is_visible(t, frame_num)]
        current = [[ti.point for ti in self.tracks[idx] if ti.frame_num == frame_num][0] for idx in visible]
        #current = [[ti.point for ti in track if ti.frame_num == frame_num][0] for track in self.tracks if is_visible(track, frame_num)]
        self._current_index = visible
        self._current = np.vstack(current)


tracker = Tracker()

for i, frame in enumerate(video):
    t_frame = i / camera_model.frame_rate
    samp_index = gt_dict['Fg'] * (t_frame + gt_dict['offset'])
    q = gyro.rotation_at(samp_index, gt_dict['Fg'])

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.update(frame_gray, i)

    if i == 60:
        break

tracker.tracks.sort(key=lambda track: len(track))

for track in tracker.tracks[:10]:
    frames = [ti.frame_num for ti in track]
    print len(frames), frames