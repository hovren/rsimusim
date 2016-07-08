from __future__ import print_function, division

from numbers import Number

import yaml
import numpy as np

from crisp.camera import AtanCameraModel
from imusim.simulation.base import Simulation
from imusim.platforms.imus import IdealIMU
from imusim.behaviours.imu import BasicIMUBehaviour

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

    def run(self):
        # Camera simulator runs multiprocessed
        self.camera.camera.start_multiproc()

        # Simulate
        self.simulation.time = self.config.start_time
        self.simulation.run(self.config.end_time)

        # Stop camera worker processes
        self.camera.camera.stop_multiproc()

    @property
    def image_measurements(self):
        return self.camera.camera.measurements

    @property
    def gyroscope_measurements(self):
        return self.imu.gyroscope.rawMeasurements

    @property
    def accelerometer_measurements(self):
        return self.imu.accelerometer.rawMeasurements

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
    def from_config(cls, path):
        instance = cls()
        instance.config = SimulationConfiguration()
        instance.config.parse_yaml(path)
        instance._setup()

        return instance

class SimulationConfiguration:
    def __init__(self):
        self.camera_model = None
        self.Rci = None
        self.pci = None
        self.dataset = None
        self.start_time = None
        self.end_time = None
        self.imu_config = None

    def parse_yaml(self, path):
        conf = yaml.safe_load(open(path, 'r'))
        self._load_camera(conf)
        self._load_relpose(conf)
        self._load_dataset(conf)
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

    def _load_dataset(self, conf):
        dinfo = conf['dataset']
        ds = Dataset.from_file(dinfo['path'])
        self.dataset = ds
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