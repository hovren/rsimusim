from __future__ import print_function, division

from numbers import Number
import os

import yaml
import numpy as np
import h5py
import time
import datetime

from crisp.camera import AtanCameraModel
from imusim.simulation.base import Simulation
from imusim.platforms.imus import IdealIMU
from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.maths.quaternions import QuaternionArray
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
        self.environment = SceneEnvironment(self.config.dataset)
        self.simulation = Simulation(environment=self.environment)

        # Configure camera
        self.camera = CameraPlatform(self.config.camera_model, self.config.Rci, self.config.pci,
                                     simulation=self.simulation, trajectory=self.config.dataset.trajectory)
        self.camera_behaviour = BasicCameraBehaviour(self.camera)

        # Configure IMU
        imu_conf = self.config.imu_config
        assert imu_conf['type'] == 'DefaultIMU'
        self.imu = DefaultIMU(imu_conf['accelerometer']['bias'], imu_conf['accelerometer']['noise'],
                              imu_conf['gyroscope']['bias'], imu_conf['gyroscope']['noise'],
                              simulation=self.simulation, trajectory=self.config.dataset.trajectory)
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
            return datetime.datetime.strptime(h5ds.value, cls.__datetime_format)

        with h5py.File(path, 'r') as f:
            instance.time_started = load_datetime(f['time_started'])
            instance.time_finished = load_datetime(f['time_finished'])
            instance.config_text = f['config_text'].value
            instance.config_path = f['config_path'].value
            instance.dataset_path = f['dataset_path'].value
            instance.gyroscope_measurements = load_timeseries(f['gyroscope'])
            instance.accelerometer_measurements = load_timeseries(f['accelerometer'])
            instance.image_measurements = load_observations(f)

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

        with h5py.File(path, 'w') as f:
            f['time_started'] = convert_datetime(self.time_started)
            f['time_finished'] = convert_datetime(self.time_finished)
            f['config_text'] = self.config_text
            f['config_path'] = self.config_path
            f['dataset_path'] = self.dataset_path
            save_timeseries(self.gyroscope_measurements, f.create_group('gyroscope'))
            save_timeseries(self.accelerometer_measurements, f.create_group('accelerometer'))
            save_observations(self.image_measurements, f)


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
        text = open(path, 'r').read()
        conf = yaml.safe_load(text)
        self.text = text
        self.path = path
        self._load_camera(conf)
        self._load_relpose(conf)
        self._load_dataset(conf, datasetdir)
        self.imu_config = self._load_imu_config(conf)

    def _load_camera(self, conf):
        cinfo = conf['camera']
        ctype = cinfo['type']
        rows = cinfo['rows']
        cols = cinfo['cols']
        framerate = cinfo['framerate']
        readout = cinfo['readout']
        if not ctype == 'AtanCameraModel':
            raise ValueError("No such camera model: {}".format(ctype))

        params = cinfo['parameters']
        camera_matrix = np.array(params['camera_matrix']).reshape(3,3)
        wc = np.array(params['dist_center'])
        lgamma = params['dist_param']
        camera = AtanCameraModel([cols, rows], framerate, readout, camera_matrix, wc, lgamma)
        self.camera_model = camera

    def _load_relpose(self, conf):
        pinfo = conf['relative_pose']
        R = np.array(pinfo['rotation']).reshape(3,3)
        if not self._is_rotation(R):
            raise ValueError("Not a rotation matrix; {}".format(R))
        self.Rci = R
        self.pci = np.array(pinfo['translation']).reshape(3,1)

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

        self.dataset = ds
        self.dataset_path = ds_path
        self.start_time = dinfo['start']
        self.end_time = dinfo['end']
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

        def load_vector_or_float(x, only_vector=False):
            if isinstance(x, Number):
                if only_vector:
                    raise ValueError("Expected vector, got float")
                else:
                    return x
            elif isinstance(x, list) and len(x) == 3:
                return np.array(x).reshape(3,1)
            else:
                raise ValueError("Failed to load vector/float: {}".format(x))

        # Store as either float or (3,1) numpy array
        ainfo['bias'] = load_vector_or_float(ainfo['bias'], only_vector=True)
        ainfo['noise'] = load_vector_or_float(ainfo['noise'])
        ginfo['bias'] = load_vector_or_float(ginfo['bias'], only_vector=True)
        ginfo['noise'] = load_vector_or_float(ginfo['noise'])

        return iinfo

    def _is_rotation(self, R):
        return np.allclose(np.dot(R, R.T), np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)