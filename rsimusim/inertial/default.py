from __future__ import print_function, division

import numpy as np

#from imusim.platforms.sensors import NoisyTransformedSensor
from imusim.platforms.gyroscopes import NoisyTransformedGyroscope
from imusim.platforms.accelerometers import NoisyTransformedAccelerometer
from imusim.platforms.magnetometers import IdealMagnetometer
from imusim.platforms.imus import IdealIMU, StandardIMU
from imusim.platforms.adcs import IdealADC
from imusim.platforms.timers import IdealTimer
from imusim.platforms.radios import IdealRadio

class DefaultIMU(StandardIMU):
    def __init__(self, acc_bias, acc_noise, gyro_bias, gyro_noise,
                 simulation=None, trajectory=None):
        identity = np.eye(3, dtype='double')
        self.accelerometer = NoisyTransformedAccelerometer(self, acc_noise, identity, acc_bias)
        self.gyroscope = NoisyTransformedGyroscope(self, gyro_noise, identity, gyro_bias)
        self.magnetometer = IdealMagnetometer(self)
        self.adc = IdealADC(self)
        self.radio = IdealRadio(self)
        self.timer = IdealTimer(self)
        StandardIMU.__init__(self, simulation, trajectory)