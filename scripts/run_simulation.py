from __future__ import print_function, division

import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

import numpy as np
import h5py
from imusim.simulation.base import Simulation
from imusim.platforms.imus import IdealIMU
from imusim.behaviours.imu import BasicIMUBehaviour
from rsimusim.dataset import Dataset
from rsimusim.camera import CameraPlatform, PinholeModel, BasicCameraBehaviour
from rsimusim.scene import SceneEnvironment

CAMERA_MATRIX = np.array(
            [[ 862.43356025,    0.        ,  987.89341878],
       [   0.        ,  862.43356025,  525.14469927],
       [   0.        ,    0.        ,    1.        ]])
       
CAMERA_READOUT = 1. / 35
CAMERA_SIZE = (1920, 1080)
CAMERA_FPS = 30.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('output')
    parser.add_argument('--imu-rate', type=float, default=400.)
    parser.add_argument('--start', type=float)
    parser.add_argument('--length', type=float, default=-1)

    return parser.parse_args()

def save_timeseries(ts, h5anchor, groupkey):
    group = h5anchor.create_group(groupkey)
    group['data'] = ts.values
    group['timestamps'] = ts.timestamps

def pad(n):
    return int(np.ceil(np.log10(n + 0.5)))

def save_camera_observations(h5f, camera):
    camera_group = h5f.create_group('camera')
    camera_group['camera_matrix'] = CAMERA_MATRIX
    camera_group['readout'] = CAMERA_READOUT
    camera_group['framerate'] = CAMERA_FPS
    camera_group['size'] = np.array(CAMERA_SIZE, dtype='int')
    frame_super_group = h5f.create_group('frames')
    zeros = pad(len(camera.measurements.values))

    #for i, t in enumerate(camera.measurements.timestamps):
    #    fnr = (t - camera.measurements.timestamps[0]) * CAMERA_FPS
    #    print('{:.4f} {:3d} {:.4f}'.format(t, i, fnr))

    for frame_number, (frame_time, frame_observations) in enumerate(
            zip(camera.measurements.timestamps, camera.measurements.values)):
        frame_group = frame_super_group.create_group('{fnr:0{pad:d}d}'.format(
            fnr=frame_number, pad=zeros))
        frame_group['time'] = frame_time
        landmarks = sorted(frame_observations.keys())
        frame_group['landmark_ids'] = np.array(landmarks, dtype='int')
        if landmarks:
            measurements = np.vstack([np.array(frame_observations[x]).reshape(1,2)
                                  for x in landmarks])
        else:
            measurements = np.array([])
        frame_group['measurements'] = measurements
        assert len(frame_group['measurements']) == len(frame_group['landmark_ids'])

def save_results(filepath, imu, camera, start_time):
    with h5py.File(filepath, 'w') as h5f:
        save_timeseries(imu.gyroscope.rawMeasurements, h5f, 'gyroscope')
        save_timeseries(imu.accelerometer.rawMeasurements, h5f, 'accelerometer')
        save_camera_observations(h5f, camera)
        h5f['start_time'] = float(start_time)

if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.output):
        logger.error("Output file %s already exists!", args.output)
        sys.exit(-1)

    ds = Dataset.from_file(args.dataset)
    logger.info('Loaded dataset "%s" with %d landmarks from %s', ds.name, len(ds.landmarks), args.dataset)
    logger.info('Trajectory time: %.1f-%.1f (%.1f seconds)', ds.trajectory.startTime, ds.trajectory.endTime, ds.trajectory.endTime - ds.trajectory.startTime)

    if args.start is not None:
        if not ds.trajectory.startTime <= args.start < ds.trajectory.endTime:
            logger.error("Start time %.1f outside valid range", args.start)
            sys.exit(-1)
        start_time = args.start
    else:
        start_time = ds.trajectory.startTime

    environment = SceneEnvironment(ds)
    sim = Simulation(environment=environment)
    logger.info("Created simulation")

    camera_model = PinholeModel(CAMERA_MATRIX, CAMERA_SIZE, CAMERA_READOUT, CAMERA_FPS)
    logger.info("Camera model class: %s", camera_model.__class__.__name__)
    camera = CameraPlatform(camera_model, simulation=sim,
                            trajectory=ds.trajectory)
    camera.camera.start_multiproc()

    camera_behaviour = BasicCameraBehaviour(camera)
    logger.info("Created camera")

    imu = IdealIMU(simulation=sim, trajectory=ds.trajectory)
    dt = 1 / args.imu_rate
    imu_behaviour = BasicIMUBehaviour(imu, dt, initialTime=start_time)
    logger.info("Created IMU with rate %.2f", args.imu_rate)

    sim.time = start_time
    logger.info('Simulation starts at time %.1f', sim.time)
    duration = args.length if args.length > 0 else ds.trajectory.endTime - sim.time
    logger.info('Will simulate %.1f seconds of data', duration)
    end_time = sim.time + duration
    sim.run(end_time)
    camera.camera.stop_multiproc()

    print('Camera times (first 10)')
    print(camera.camera.measurements.timestamps[:10])
    print('Gyro times (first 10)')
    print(imu.gyroscope.rawMeasurements.timestamps[:10])

    save_results(args.output, imu, camera.camera, start_time)
    logging.info("Saved to %s", args.output)


