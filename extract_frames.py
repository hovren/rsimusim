import argparse
import os
import json
import sys
import shutil
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import cv2

import crisp
import crisp.rotations
import stabby

argparser = argparse.ArgumentParser()
argparser.add_argument('video')
argparser.add_argument('gyro')
argparser.add_argument('camera')
argparser.add_argument('parameters')
argparser.add_argument('--mode', choices=['normal', 'smart'], default='normal')
argparser.add_argument('--outputdir', default='out/')
argparser.add_argument('--force', action='store_true')
argparser.add_argument('--start', type=int)
argparser.add_argument('--stop', type=int)
argparser.add_argument('--step', type=int)
args = argparser.parse_args()

args.outputdir = os.path.abspath(args.outputdir)

if os.path.exists(args.outputdir):
    if args.force:
        print 'Removing directory', args.outputdir
        shutil.rmtree(args.outputdir)
    else:
        print 'Error: {} already exists!'.format(args.outputdir)
        sys.exit(-1)

try:
    os.makedirs(args.outputdir)
except OSError:
    print 'Error: Failed to create {}'.format(args.outputdir)
    sys.exit(-1)


def load_params(filepath):
    _, ext = os.path.splitext(filepath)
    print ext
    if ext == '.csv':
        arr = np.loadtxt(filepath, delimiter=',')
        param_names = ('gyro_rate', 'time_offset', 'rot_x', 'rot_y', 'rot_z', 'gbias_x', 'gbias_y', 'gbias_z')
        data = {key : float(val) for key, val in zip(param_names, arr)}
    elif ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Don't know how to open {}".format(filepath))
    return data


def load_gyro(filepath):
    return crisp.GyroStream.from_csv(filepath)


def load_camera(argstring):
    camera_type, camera_path = argstring.split(':')
    load_func = {
        'opencv' : crisp.OpenCVCameraModel.from_hdf,
        'atan' : crisp.AtanCameraModel.from_hdf
    }[camera_type]

    return load_func(camera_path)

camera_model = load_camera(args.camera)
print camera_model.camera_matrix
calibration_params = load_params(args.parameters)
#print calibration_params
gyro_stream = load_gyro(args.gyro)

if os.path.splitext(args.video)[0] in ('rccar', 'walk', 'rotation'):
    import crisp.l3g4200d
    print 'gyro: Applying L3G4200D post processing'
    gyro_stream.data = crisp.l3g4200d.post_process_L3G4200D_data(gyro_stream.data.T).T

print 'gyro: Applying bias'
bias = np.array([calibration_params['gbias_{}'.format(axis)] for axis in 'xyz'])
gyro_stream.data - bias.reshape(1,3)

# Unpack R_g2c
r = np.array([calibration_params['rot_{}'.format(axis)] for axis in 'xyz'])
theta = np.linalg.norm(r)
v = r / theta
R_g2c = crisp.rotations.axis_angle_to_rotation_matrix(v, theta)



print gyro_stream
video_stream = crisp.OpenCvVideoStream(camera_model, args.video)

print 'Parameters'
print '-'*15
for key, val in calibration_params.items():
    print '{:>15s} = {:<9.4e}'.format(key, val)

print 'Starting video'

dt = 1. / calibration_params['gyro_rate']
orientations = gyro_stream.integrate(dt)
gyro_timestamps = np.arange(gyro_stream.num_samples) * dt

frametime_to_gyrosample = lambda t: calibration_params['gyro_rate'] * t
frame_to_frametime = lambda n: float(n) / camera_model.frame_rate + calibration_params['time_offset']
frame_to_gyrosample = lambda n: frametime_to_gyrosample(frame_to_frametime(n))

if False:
    rotvel = np.abs(np.array([crisp.rotations.rotation_matrix_to_axis_angle(crisp.rotations.quat_to_rotation_matrix(q))[-1] for q in orientations]))
    plt.plot(gyro_timestamps, rotvel)
    for n in range(0, 855):
        tg = frame_to_gyrosample(n) * dt
        print n, tg
        plt.axvline(tg, color='k')
    plt.show()
    sys.exit(-1)

def handle_frame(frame_num, frame):
    t_frame = frame_to_frametime(frame_num)
    rectmap = stabby.rectification_map(camera_model, gyro_stream.data, gyro_timestamps, R_g2c, t_frame)
    frame_rectified = stabby.rectify(frame, rectmap).astype('uint8')
    out_filename = os.path.join(args.outputdir, 'rectframe_{:05d}.jpg'.format(frame_num))
    cv2.imwrite(out_filename, frame_rectified)
    print 'Wrote', out_filename


if args.mode == 'normal':
    next_frame_num = args.start if args.start else 0
    step = args.step if args.step else 1

    for frame_num, frame in enumerate(video_stream):
        if frame_num == next_frame_num:
            handle_frame(frame_num, frame)
            next_frame_num += step
            if args.stop and next_frame_num > args.stop:
                break

elif args.mode == 'smart':
    ANGLE_THRESHOLD = np.deg2rad(5.0)
    STEP_THRESHOLD = 5
    last_R = None
    last_frame = None
    next_frame_num = args.start if args.start else 0
    for frame_num, frame in enumerate(video_stream):
        if frame_num == next_frame_num:
            gyro_sample = int(np.round(frame_to_gyrosample(frame_num)))
            if last_R is None:
                last_R = crisp.rotations.quat_to_rotation_matrix(orientations[gyro_sample])
                last_frame = frame_num
                handle_frame(frame_num, frame)
            else:
                this_R = crisp.rotations.quat_to_rotation_matrix(orientations[gyro_sample])
                dR = np.dot(last_R, this_R.T)
                v, theta = crisp.rotations.rotation_matrix_to_axis_angle(dR)
                if np.abs(theta) > ANGLE_THRESHOLD or frame_num - last_frame > STEP_THRESHOLD:
                    print '{:d} -> {:d} [{:d}] angle = {:.2f} degrees'.format(last_frame, frame_num, frame_num - last_frame, np.rad2deg(theta))
                    handle_frame(frame_num, frame)
                    last_frame = frame_num
                    last_R = this_R

            next_frame_num += 1
            if args.stop and next_frame_num > args.stop:
                break