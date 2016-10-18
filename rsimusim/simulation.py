from __future__ import print_function, division

from numbers import Number
import os

import yaml
import numpy as np
import h5py
import time
import datetime
import logging

logger = logging.getLogger("rsimusim.simulation")

from crisp.camera import AtanCameraModel
from imusim.simulation.base import Simulation
from imusim.platforms.imus import IdealIMU
from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.maths.quaternions import QuaternionArray, Quaternion
from imusim.trajectories.splined import SplinedTrajectory
from imusim.trajectories.sampled import SampledTrajectory
from imusim.utilities.time_series import TimeSeries

from .camera import CameraPlatform, PinholeModel, BasicCameraBehaviour
from .inertial import DefaultIMU
from .scene import SceneEnvironment
from .dataset import Dataset

class RollingShutterImuSimulation:
    def __init__(self):
        self.config = None
        self.environment = None
        self.camera = None
        self.camera_behaviour = None
        self.imu = None
        self.imu_behaviour = None
        self.simulation = None
        self.simulation_trajectory = None


    def run(self, progress=False):
        # Camera simulator runs multiprocessed
        self.camera.camera.start_multiproc()

        # Simulate
        self.simulation.time = self.config.start_time
        t0 = datetime.datetime.now()
        self.simulation.run(self.config.end_time, printProgress=progress)
        t1 = datetime.datetime.now()

        # Stop camera worker processes
        self.camera.camera.stop_multiproc()

        # Assemble results
        results = SimulationResults()
        results.trajectory = self.simulation_trajectory
        results.config_path = self.config.path
        results.config_text = self.config.text
        results.dataset_path = self.config.dataset_path
        results.time_started = t0
        results.time_finished = t1
        results.image_measurements = self.camera.camera.measurements
        results.accelerometer_measurements = self.imu.accelerometer.rawMeasurements
        results.gyroscope_measurements = self.imu.gyroscope.rawMeasurements

        return results

    def _setup(self):
        # Trajectory used by the simulation
        # Not the same as the dataset to account for the relative pose
        # between camera and IMU
        self.simulation_trajectory = transform_trajectory(self.config.dataset.trajectory,
                                                          self.config.Rci, self.config.pci)

        from numpy.testing import assert_equal
        assert_equal(self.simulation_trajectory.startTime, self.config.dataset.trajectory.startTime)
        assert_equal(self.simulation_trajectory.endTime, self.config.dataset.trajectory.endTime)

        self.environment = SceneEnvironment(self.config.dataset)
        self.simulation = Simulation(environment=self.environment)

        # Configure camera
        self.camera = CameraPlatform(self.config.camera_model, self.config.Rci, self.config.pci,
                                     simulation=self.simulation, trajectory=self.simulation_trajectory)
        self.camera_behaviour = BasicCameraBehaviour(self.camera, self.config.end_time)

        # Configure IMU
        imu_conf = self.config.imu_config
        assert imu_conf['type'] == 'DefaultIMU'
        self.imu = DefaultIMU(imu_conf['accelerometer']['bias'], imu_conf['accelerometer']['noise'],
                              imu_conf['gyroscope']['bias'], imu_conf['gyroscope']['noise'],
                              simulation=self.simulation, trajectory=self.simulation_trajectory)
        imu_dt = 1. / self.config.imu_config['sample_rate']
        self.imu_behaviour = BasicIMUBehaviour(self.imu, imu_dt, initialTime=self.config.start_time)

    @classmethod
    def from_config(cls, path, datasetdir=None):
        instance = cls()
        instance.config = SimulationConfiguration()
        instance.config.parse_yaml(path, datasetdir)
        instance._setup()

        return instance

class SimulationResults:
    __datetime_format = '%Y-%m-%d %H:%M:%S.%f'

    def __init__(self):
        self.time_started = None
        self.time_finished = None
        self.config_text = None
        self.config_path = None
        self.dataset_path = None
        self.image_measurements = None
        self.gyroscope_measurements = None
        self.accelerometer_measurements = None

    @classmethod
    def from_file(cls, path):
        instance = cls()
        def load_timeseries(group):
            times = group['timestamps']
            values = group['data']
            return TimeSeries(timestamps=times, values=values)

        def load_observations(h5_file):
            framegroup = h5_file['camera']
            frames = sorted(framegroup.keys())
            ts = TimeSeries()
            for fkey in frames:
                group = framegroup[fkey]
                obs_dict = {lm_id: ip.reshape(2,1) for lm_id, ip in zip(group['landmarks'].value,
                                                           group['measurements'].value.T)}
                t = group['time'].value
                ts.add(t, obs_dict)
            return ts

        def load_datetime(h5ds):
            time_data = h5ds.value
            time_str = time_data.decode('utf8') if isinstance(time_data, bytes) else time_data
            return datetime.datetime.strptime(time_str, cls.__datetime_format)

        def load_trajectory(group):
            pos_ts = load_timeseries(group['position'])
            rot_array_ts = load_timeseries(group['rotation'])
            rot_ts = TimeSeries(rot_array_ts.timestamps, QuaternionArray(rot_array_ts.values.T))
            sampled = SampledTrajectory(positionKeyFrames=pos_ts, rotationKeyFrames=rot_ts)
            splined = SplinedTrajectory(sampled, smoothRotations=False)
            return splined

        with h5py.File(path, 'r') as f:
            instance.time_started = load_datetime(f['time_started'])
            instance.time_finished = load_datetime(f['time_finished'])
            instance.config_text = f['config_text'].value
            instance.config_path = f['config_path'].value
            instance.dataset_path = f['dataset_path'].value
            instance.gyroscope_measurements = load_timeseries(f['gyroscope'])
            instance.accelerometer_measurements = load_timeseries(f['accelerometer'])
            instance.image_measurements = load_observations(f)
            instance.trajectory = load_trajectory(f['trajectory'])

        return instance

    def save(self, path):
        def save_timeseries(ts, group):
            group['data'] = ts.values
            group['timestamps'] = ts.timestamps

        def save_observations(ts, h5_file):
            framegroup = h5_file.create_group('camera')
            pad = int(np.ceil(np.log10(len(ts)+0.5)))
            for framenum, (t, obs) in enumerate(zip(ts.timestamps, ts.values)):
                landmarks = np.array(sorted(list(obs.keys())), dtype='int')
                if len(landmarks) < 1:
                    measurements = []
                else:
                    measurements = np.hstack([obs[lm_id].reshape(2,1) for lm_id in landmarks])
                group = framegroup.create_group('frame_{framenum:0{pad}d}'.format(framenum=framenum, pad=pad))
                group['landmarks'] = landmarks
                group['measurements'] = measurements
                group['time'] = t

        def convert_datetime(dtime):
            return dtime.strftime(self.__datetime_format)

        def save_trajectory(trajectory, h5f):
            traj_group = h5f.create_group('trajectory')
            save_timeseries(trajectory.sampled.positionKeyFrames, traj_group.create_group('position'))
            rot_ts = TimeSeries(trajectory.sampled.rotationKeyFrames.timestamps,
                                trajectory.sampled.rotationKeyFrames.values.array.T)
            save_timeseries(rot_ts, traj_group.create_group('rotation'))

        with h5py.File(path, 'w') as f:
            f['time_started'] = convert_datetime(self.time_started)
            f['time_finished'] = convert_datetime(self.time_finished)
            f['config_text'] = self.config_text
            f['config_path'] = self.config_path
            f['dataset_path'] = self.dataset_path
            save_timeseries(self.gyroscope_measurements, f.create_group('gyroscope'))
            save_timeseries(self.accelerometer_measurements, f.create_group('accelerometer'))
            save_observations(self.image_measurements, f)
            save_trajectory(self.trajectory, f)


class SimulationConfiguration:
    def __init__(self):
        self.camera_model = None
        self.Rci = None
        self.pci = None
        self.dataset = None
        self.dataset_path = None
        self.start_time = None
        self.end_time = None
        self.imu_config = None
        self.text = None
        self.path = None

    def parse_yaml(self, path, datasetdir=None):
        with open(path, 'r') as f:
            text = f.read()
        conf = yaml.safe_load(text)
        self.text = text
        self.path = path
        self._load_camera(conf)
        self._load_relpose(conf)
        self._load_dataset(conf, datasetdir)
        self.imu_config = self._load_imu_config(conf)

    def _load_camera(self, conf):
        cinfo = conf['camera']
        ctype = cinfo['type'].lower()
        rows = cinfo['rows']
        cols = cinfo['cols']
        framerate = cinfo['framerate']
        readout = cinfo['readout']

        params = cinfo['parameters']
        camera_matrix = np.array(params['camera_matrix']).reshape(3,3)

        if ctype == 'atan':
            wc = np.array(params['dist_center'])
            lgamma = params['dist_param']
            camera = AtanCameraModel([cols, rows], framerate, readout, camera_matrix, wc, lgamma)
        elif ctype == 'pinhole':
            camera = PinholeModel(camera_matrix, [cols, rows], readout, framerate)
        else:
            raise ValueError("No such camera model: {}".format(ctype))

        self.camera_model = camera

    def _load_relpose(self, conf):
        try:
            pinfo = conf['relative_pose']
            R = np.array(pinfo['rotation']).reshape(3,3)
            p = np.array(pinfo['translation']).reshape(3,1)
            if not self._is_rotation(R):
                raise ValueError("Not a rotation matrix; {}".format(R))
        except KeyError:
            R = np.eye(3, dtype='double')
            p = np.zeros((3,1), dtype='double')
        self.Rci = R
        self.pci = p

    def _load_dataset(self, conf, datasetdir=None):
        dinfo = conf['dataset']
        search_paths = ['.'] if datasetdir is None else [datasetdir, '.']
        ds = None
        for root in search_paths:
            ds_path = os.path.join(root, dinfo['path'])
            if os.path.exists(ds_path):
                ds = Dataset.from_file(ds_path)
                break

        if not ds:
            raise ValueError("Failed to find {} in search paths {}".format(dinfo['path'], search_paths))

        # Make sure dataset has aligned spline knots in trajectory
        traj = ds.trajectory
        if not np.all(traj.sampled.rotationKeyFrames.timestamps == traj.sampled.positionKeyFrames.timestamps):
            raise ValueError("Dataset must have aligned spline knots in trajectory")

        self.dataset = ds
        self.dataset_path = ds_path
        try:
            self.start_time = dinfo['start']
            logger.info("Configuration is missing dataset>start key. Using trajectory start time.")
        except KeyError:
            self.start_time = ds.trajectory.startTime

        try:
            self.end_time = dinfo['end']
        except KeyError:
            self.end_time = ds.trajectory.endTime
            logger.info("Configuration is missing dataset>end key. Using trajectory end time.")

        if self.start_time < ds.trajectory.startTime:
            raise ValueError("Invalid start time")
        if self.end_time > ds.trajectory.endTime:
            raise ValueError("Invalid end time")

    def _load_imu_config(self, conf):
        iinfo = conf['imu']
        ainfo = iinfo['accelerometer']
        ginfo = iinfo['gyroscope']
        imu_type = iinfo['type']
        if not imu_type == 'DefaultIMU':
            return ValueError("Unsupported IMU type: {}".format(imu_type))

        def load_bias(x):
            if isinstance(x, list) and len(x) == 3:
                return np.array(x).reshape(3,1)
            else:
                raise ValueError("Bias must be vector of 3 elements")

        def load_noise(x):
            if isinstance(x, Number):
                if x >= 0:
                    return x
                else:
                    raise ValueError("Negative noise scale: {}".format(x))
            elif isinstance(x, list) and len(x) == 3:
                x = np.array(x).reshape(3,1)
                if np.any(x == 0.0):
                    raise ValueError("Noise vectors must not have zero elements")
                else:
                    return x
            else:
                raise ValueError("Unknown noise type: {}".format(x))

        # Store as either float or (3,1) numpy array
        ainfo['bias'] = load_bias(ainfo['bias'])
        ainfo['noise'] = load_noise(ainfo['noise'])
        ginfo['bias'] = load_bias(ginfo['bias'])
        ginfo['noise'] = load_noise(ginfo['noise'])

        return iinfo

    def _is_rotation(self, R):
        return np.allclose(np.dot(R, R.T), np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)

def transform_trajectory(trajectory, R, p):
    """Create new trajectory relative the given transformation

    If R1(t) and p1(t) is the rotation and translation given by inout trajectory, then
    this function returns a new trajectory that fulfills the following.

    Let X1 = R1(t).T[X - p1(t)] be the coordinate of point X in input trajectory coordinates.
    Then X2 = R X1 + p is the same point in the coordinate frame of the new trajectory
    Since X2 = R2(t).T [X - p2(t)] then we have
    R2(t) = R1(t)R and p2(t) = p1(t) + R1(t)p
    """
    ts_q1 = trajectory.sampled.rotationKeyFrames
    q1 = ts_q1.values
    ts_p1 = trajectory.sampled.positionKeyFrames
    p1 = ts_p1.values

    # Rotation
    Q = Quaternion.fromMatrix(R)
    q2 = q1 * Q
    ts_q2 = TimeSeries(ts_q1.timestamps, q2)

    # Translation
    p2 = p1 + q1.rotateVector(p)
    ts_p2 = TimeSeries(ts_p1.timestamps, p2)

    sampled = SampledTrajectory(positionKeyFrames=ts_p2, rotationKeyFrames=ts_q2)
    smoothen = False # MUST be the same as used in Dataset class
    splined = SplinedTrajectory(sampled, smoothRotations=smoothen)
    return splined
