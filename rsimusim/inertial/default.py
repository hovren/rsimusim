from __future__ import print_function, division

import numpy as np

#from imusim.platforms.sensors import NoisyTransformedSensor
from imusim.platforms.sensors import TransformedSensor
from imusim.platforms.gyroscopes import NoisyTransformedGyroscope, IdealGyroscope
from imusim.platforms.accelerometers import NoisyTransformedAccelerometer, IdealAccelerometer
from imusim.platforms.magnetometers import IdealMagnetometer
from imusim.platforms.imus import IdealIMU, StandardIMU
from imusim.platforms.adcs import IdealADC
from imusim.platforms.timers import IdealTimer
from imusim.platforms.radios import IdealRadio

class TransformedAccelerometer(TransformedSensor, IdealAccelerometer):
    pass

class TransformedGyroscope(TransformedSensor, IdealGyroscope):
    pass

class DefaultIMU(StandardIMU):
    def __init__(self, acc_bias, acc_noise, gyro_bias, gyro_noise,
                 simulation=None, trajectory=None):
        identity = np.eye(3, dtype='double')

        # To handle the noise free case we need to choose different sensor subclasses
        # (Otherwise the call that creates random data raises an Exception due to zero scale)
        if acc_noise > 0:
            self.accelerometer = NoisyTransformedAccelerometer(self, acc_noise, identity, acc_bias)
        else:
            self.accelerometer = TransformedAccelerometer(self, identity, acc_bias)

        if gyro_noise > 0:
            self.gyroscope = NoisyTransformedGyroscope(self, gyro_noise, identity, gyro_bias)
        else:
            self.gyroscope = TransformedGyroscope(self, identity, gyro_bias)

        self.magnetometer = IdealMagnetometer(self)
        self.adc = IdealADC(self)
        self.radio = IdealRadio(self)
        self.timer = IdealTimer(self)
        StandardIMU.__init__(self, simulation, trajectory)