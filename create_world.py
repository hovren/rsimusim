from __future__ import division
from collections import namedtuple

import cv2
import numpy as np
from crisp import VideoStream, GyroStream, AtanCameraModel
from crisp.rotations import axis_angle_to_rotation_matrix, quat_to_rotation_matrix, slerp
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

class Track(object):
    __slots__ = ('id', 'frames', 'points')

    _next_id = 0

    def __init__(self, frame_num=None, point=None):
        self.id = self.__class__._next_id
        self.__class__._next_id += 1

        self.frames = []
        self.points = []

        if frame_num is not None and point is not None:
            self.add(frame_num, point)

    @classmethod
    def tracks_to_array(cls, track_list, frame_num):
        pts = [track[frame_num] for track in track_list if frame_num in track]
        return np.array(pts)

    def add(self, framenum, point):
        point = tuple(point)
        assert (len(self.frames) < 1) or (framenum == self.frames[-1] + 1)
        self.frames.append(framenum)
        self.points.append(point)

    def __contains__(self, frame_num):
        return frame_num in self.frames

    def __getitem__(self, frame_num):
        try:
            index = self.frames.index(frame_num)
            return self.points[index]
        except ValueError:
            raise KeyError("Frame not in track")

    def __repr__(self):
        return '<Track id={}, from={:d}, to={:d} #frames={:d}>'.format(self.id,
                                                                       self.frames[0],
                                                                       self.frames[-1],
                                                                       len(self.frames))

class ImageTracker(object):
    def __init__(self):
        self.tracks = []
        self._current = None
        self._last_frame = None

    def update(self, frame, frame_num):
        if self._current is None:
            new_points = cv2.goodFeaturesToTrack(frame, 300, 0.07, 10)
            new_points.reshape(-1, 2)
        else:
            new_points = []
            # Track current points
            last_frame = frame_num - 1
            current_pts = Track.tracks_to_array(self._current, last_frame).reshape(1,-1,2)
            [_points, status, err] = cv2.calcOpticalFlowPyrLK(self._last_frame,
                                                              frame,
                                                              current_pts,
                                                              np.array([]))

            if _points is not None:
                _points = _points.reshape(-1, 2)

                valid = np.flatnonzero(status == 1)
                for idx in valid:
                    track = self._current[idx]
                    pt = _points[idx]
                    track.add(frame_num, pt)

        # New points to new tracks
        for p in new_points:
            t = Track(frame_num, p)
            self.tracks.append(t)

        # Update list of currently OK tracks
        self._update_current(frame_num)
        self._last_frame = frame

    def _update_current(self, frame_num):
        self._current = [track for track in self.tracks if frame_num in track]

    def tracks_for_interval(self, a, b):
        valid = []
        for track in self.tracks:
            for frame_num in range(a, b + 1):
                if not frame_num in track:
                    break
            else:
                valid.append(track)
        return valid

frametime_to_sample = lambda t: gt_dict['Fg'] * (t + gt_dict['offset'])

tracker = ImageTracker()

keyframe_interval = 15
triangulate_back = 3

keyframes = []
Landmark = namedtuple('Landmark', ['world_point', 'visibility'])

for frame_num, frame in enumerate(video):
    t_frame = frame_num / camera_model.frame_rate
    samp_index = frametime_to_sample(t_frame)
    q = gyro.rotation_at(samp_index, gt_dict['Fg'])

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.update(frame_gray, frame_num)

    if frame_num >= triangulate_back and frame_num % keyframe_interval == 0:
        tri_frame_num = frame_num - triangulate_back
        tri_tracks = tracker.tracks_for_interval(tri_frame_num, frame_num)

        current_keyframe = len(keyframes)

        pts1 = np.vstack([track[tri_frame_num] for track in tri_tracks])
        pts1 = camera_model.invert(pts1)
        pts2 = np.vstack([track[frame_num] for track in tri_tracks])
        pts2 = camera_model.invert(pts2)
        t_tri_frame = tri_frame_num / camera_model.frame_rate
        tri_samp_index = frametime_to_sample(t_tri_frame)
        q_tri = gyro.rotation_at(tri_samp_index, gt_dict['Fg'])
        R_tri = quat_to_rotation_matrix(q_tri)
        R = quat_to_rotation_matrix(q)
        dR = np.dot(R_tri, R)

    if frame_num == 260:
        break

tracker.tracks.sort(key=lambda track: len(track.frames))

for track in tracker.tracks[:15]:
    print track