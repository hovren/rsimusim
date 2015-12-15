from collections import namedtuple
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import crisp.rotations
from mayavi import mlab
from imusim.trajectories.sampled import SampledPositionTrajectory, SampledRotationTrajectory
from imusim.trajectories.splined import SplinedPositionTrajectory, SplinedRotationTrajectory
from imusim.utilities.time_series import TimeSeries, QuaternionArray, Quaternion

from rsimusim_legacy.world import NvmModel

filename = 'walk_model.nvm'
model = NvmModel.from_file(filename)
print 'Loaded {:d} cameras and {:d} points from {}'.format(len(model.cameras), len(model.points), filename)

camera_fps = 30.
camera_timestamps = np.array(model.camera_framenums) / camera_fps
camera_timeseries = TimeSeries(camera_timestamps, model.camera_positions)
samp_traj = SampledPositionTrajectory(camera_timeseries)
splined_traj = SplinedPositionTrajectory(samp_traj)

times = np.linspace(splined_traj.startTime, splined_traj.endTime, num=200)
pos = splined_traj.position(times)
vel = splined_traj.velocity(times)
speed = np.linalg.norm(vel, axis=0)
#plt.plot(speed, '-o')
#plt.show()

distance_traveled = np.sum(speed)
travel_time = times[-1] - times[0]
mean_speed = 1.4 # From WP
print 'traveled', distance_traveled, 'units in', travel_time, 'seconds'
print 'using', mean_speed, 'as guessed speed'
scale_factor = mean_speed * travel_time / distance_traveled
print 'scale factor', scale_factor
print 'travled', distance_traveled * scale_factor, 'meters in', travel_time, 'seconds'

#print 'scaling world by', scale_factor
#model.scale_world(scale_factor)


world_points = model.world_points
camera_positions = model.camera_positions
print camera_positions.shape
point_colors = np.vstack([tuple(point.color) + (1,) for point in model.points]).astype('uint8')
cam1 = model.cameras[0]
print cam1

zc = np.array([0, 0, 1.]).reshape(3,1)
def camera_to_world(Xc, camera):
    R = np.array(camera.orientation.toMatrix())
    t = camera.position
    Xw = np.dot(R.T, Xc)# + t
    return Xw

zw = [camera_to_world(zc, camera).reshape(3,1) for camera in model.cameras]
zw_arr = 0.5 * np.hstack(zw)

# Upsample camera orientations
def upsample_quatarray(qa, timestamps, new_samples):
    timestamps_new = np.linspace(timestamps[0], timestamps[-1], new_samples)
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

camera_orientations = QuaternionArray([c.orientation for c in model.cameras]).unflipped()
ori_up, ts_up = upsample_quatarray(camera_orientations, camera_timestamps, 188)
#ts = TimeSeries(camera_timestamps, camera_orientations)
#chosen_idx = sorted(np.random.choice(len(ts_up), 100, replace=False))
chosen_idx = np.arange(len(ts_up))
print chosen_idx
ts = TimeSeries(ts_up[chosen_idx], ori_up[chosen_idx])
samp_rot = SampledRotationTrajectory(ts)
splined_rot = SplinedRotationTrajectory(samp_rot, smoothRotations=False)

t_min, t_max = splined_rot.startTime, splined_rot.endTime
valid_t = camera_timestamps[(camera_timestamps >= t_min) & (camera_timestamps <= t_max)]
cam_traj_rot = splined_rot.rotation(valid_t)
for i, axis in enumerate('wxyz'):
    qi_orig = camera_orientations.array[:, i]
    qi_upsamp = ori_up.array[:, i]
    qi_traj = cam_traj_rot[:, i]
    plt.subplot(2,2,i+1)
    plt.plot(camera_timestamps, qi_orig, '-o', color='g')
    plt.plot(ts_up, qi_upsamp, '-x', color='k')
    plt.plot(valid_t, qi_traj, linewidth=4, color='r', alpha=0.5)
plt.show()

for t, q in zip(ts_up, ori_up):
    print t, q

if False:
    mlab.plot3d(camera_positions[0], camera_positions[1], camera_positions[2], color=(1, 1, 0), line_width=5.0, tube_radius=None)
    #mlab.plot3d(pos[0], pos[1], pos[2], color=(0, 0.5, 0), line_width=5.0)
    #mlab.points3d(camera_positions[0], camera_positions[1], camera_positions[2], color=(1, 0, 0), scale_factor=0.02)
    mlab.points3d(world_points[0], world_points[1], world_points[2], scale_factor=0.01)
    mlab.points3d([0], [0], [0], color=(1, 0, 1), scale_factor=0.05)
    mlab.quiver3d(camera_positions[0], camera_positions[1], camera_positions[2],
                  zw_arr[0], zw_arr[1], zw_arr[2], color=(0,1,0))
    #mlab.axes()
    #mlab.orientation_axes()
    mlab.view(focalpoint=model.cameras[0].position, distance=7.0)
    mlab.show()

#for camera in model.cameras:
#    t = camera.frame_num / 30.0
#    print '{:>6.3f} {} {}'.format(t, camera.orientation, camera.position)