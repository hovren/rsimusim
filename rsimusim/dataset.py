from collections import namedtuple

from imusim.trajectories.sampled import SampledPositionTrajectory, SampledRotationTrajectory, SampledTrajectory
from imusim.trajectories.splined import SplinedTrajectory
from imusim.utilities.time_series import TimeSeries, QuaternionArray, Quaternion
from crisp.fastintegrate import integrate_gyro_quaternion_uniform
import crisp.rotations
import numpy as np

from world import NvmModel


class DatasetError(Exception):
    pass


Landmark = namedtuple('Landmark', ['point', 'visibility'])


class Dataset(object):
    def __init__(self):
        self._position_data = None # TimeSeries
        self._orientation_data = None # TimeSeries
        self.trajectory = None
        self.landmarks = [] # Landmark
        self.camera_times = []

    def landmarks_from_data(self, points, intervals, visibility):
        self.landmark_intervals = intervals
        self.landmarks = [Landmark(p, visible_intervals) for p, visible_intervals in zip(points.T, visibility)]

    def landmarks_from_nvm(self, nvm_model, frame_to_time_func=None, camera_fps=None):
        if bool(nvm_model is None) == bool(frame_to_time_func is None):
            raise DatasetError("Must specify frame_to_time_func OR camera_fps, not both or none of them")
        frame_time = frame_to_time_func if frame_to_time_func else lambda n: float(n) / camera_fps
        self.camera_times = np.array([frame_time(n) for n in nvm_model.camera_framenums])
        self.landmarks = [Landmark(p.position, p.visibility) for p in nvm_model.points]

    def orientation_from_gyro(self, gyro_data, timestamps):
        _, d = gyro_data.shape
        if d == 4:
            orientations = gyro_data
        elif d == 3:
            dt = timestamps[1] - timestamps[0]
            if not np.allclose(np.diff(timestamps), dt):
                raise DatasetError("Only uniformly sampled gyroscope streams can be used")
            orientations = integrate_gyro_quaternion_uniform(gyro_data, float(dt))
        else:
            raise DatasetError("Expected a Nx3 or Nx4 array of gyro data, got {}".format('x'.join(map(str, gyro_data.shape))))
        orientations = QuaternionArray(orientations)

        # Integrating gyro measurments yields R(t) such that
        # X(t_0) = R^T(t_0)R(t_1) X(t_1)
        orientations = orientations.conjugate
        #orientations = orientations.unflipped()

        # Splined trajectory seem to fail unless the samples are captured
        # uniformly over time. Resampling using SLERP fixes this.
        dtimes = np.diff(timestamps)
        if not np.allclose(dtimes, dtimes[0]):
            print 'Resampling orientation data'
            num_samples = len(timestamps)
            orientations, timestamps = resample_quatarray(orientations, timestamps, num_samples)

        ts = TimeSeries(timestamps, orientations)
        self._orientation_data = ts
        self._update_trajectory()

    def orientation_from_nvm(self, nvm_model,frame_to_time_func=None, camera_fps=None):
        if (bool(nvm_model is None) == bool(frame_to_time_func is None)):
            raise DatasetError("Must specify frame_to_time_func OR camera_fps, not both or none of them")
        frame_time = frame_to_time_func if frame_to_time_func else lambda n: float(n) / camera_fps
        camera_times = np.array([frame_time(n) for n in nvm_model.camera_framenums])
        camera_times = camera_times.flatten() # or weird error with splines
        camera_orientations = QuaternionArray([c.orientation for c in nvm_model.cameras])
        camera_orientations = camera_orientations.unflipped()
        # Splined trajectory seem to fail unless the samples are captured
        # uniformly over time. Resampling using SLERP fixes this.
        dtimes = np.diff(camera_times)
        if not np.allclose(dtimes, dtimes[0]):
            print 'Resampling orientation data'
            num_samples = len(camera_times)
            camera_orientations, camera_times = resample_quatarray(camera_orientations, camera_times, num_samples)

        ts = TimeSeries(camera_times, camera_orientations)
        self._orientation_data = ts
        self._update_trajectory()

    def position_from_accelerometer(self, data, timestamps):
        raise NotImplementedError

    def position_from_nvm(self, nvm_model, frame_to_time_func=None, camera_fps=None):
        if (bool(nvm_model is None) == bool(frame_to_time_func is None)):
            raise DatasetError("Must specify frame_to_time_func OR camera_fps, not both or none of them")
        frame_time = frame_to_time_func if frame_to_time_func else lambda n: float(n) / camera_fps

        camera_times = [frame_time(n) for n in nvm_model.camera_framenums]
        ts = TimeSeries(camera_times, nvm_model.camera_positions)
        self._position_data = ts

        self._update_trajectory()

    def _update_trajectory(self):
        if self._orientation_data is None or self._position_data is None:
            return

        samp_traj = SampledTrajectory(self._position_data, self._orientation_data)
        # Note: smoothRotations=False necessary, or loading from NVM fails for strange reasons
        self.trajectory = SplinedTrajectory(samp_traj, smoothRotations=False)
        print 'Num orient keyframes:', len(self._orientation_data)

    def visualize(self):
        from mayavi import mlab
        t_min = self.trajectory.startTime
        t_max = self.trajectory.endTime
        t_samples = (t_max - t_min) * 50
        t = np.linspace(t_min, t_max, t_samples)
        positions = self.trajectory.position(t)
        landmark_data = np.vstack([l.point for l in self.landmarks]).T
        valid_camera_times = [t for t in self.camera_times if t_min <= t <= t_max]
        camera_positions = self.trajectory.position(valid_camera_times)
        orientations = self.trajectory.rotation(valid_camera_times)

        # World to camera transform is
        # Xc = RXw - Rt where R is the camera orientation and position respectively
        # Camera to world is thus
        # Xw = RtXc + t
        zc = np.array([0, 0, 1.]).reshape(3,1)
        zw = [np.dot(np.array(q.toMatrix()).T, zc).reshape(3,1) for q in orientations]
        quiver_data = 0.5 * np.hstack(zw)
        #quiver_data = 0.5 * np.hstack([q.rotateVector(z) for q in orientations])

        mlab.points3d(landmark_data[0], landmark_data[1], landmark_data[2], scale_factor=0.001)
        mlab.plot3d(positions[0], positions[1], positions[2], color=(1, 0, 0), line_width=5.0, tube_radius=None)
        mlab.points3d(camera_positions[0], camera_positions[1], camera_positions[2], scale_factor=0.005, color=(1,0,0))
        mlab.quiver3d(camera_positions[0], camera_positions[1], camera_positions[2],
                      quiver_data[0], quiver_data[1], quiver_data[2], color=(1, 1, 0))
        mlab.show()

def resample_quatarray(qa, timestamps, num_samples):
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