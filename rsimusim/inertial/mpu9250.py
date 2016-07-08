from __future__ import print_function, division

import numpy as np

from imusim.platforms.gyroscopes import IdealGyroscope
from imusim.platforms.imus import IdealIMU, StandardIMU

class MarkovProc(object):
    @classmethod
    def from_ar1(cls, Ts, phi, sigma2):
        tau = Ts / (1 - phi)
        sigma_w = np.sqrt(sigma2) / Ts
        instance = cls(tau, sigma_w)
        return instance
        
    def __init__(self, tau, sigma_w, initial_time=0):
        self._t = initial_time
        self.tau = tau
        self.sigma_w = sigma_w
        self._prev = 0

    def __call__(self, t):
        dt = t - self._t
        b = self._prev
        w = np.random.normal(scale=self.sigma_w)
        new_b = (1 - dt / self.tau) * b + dt * w
        self._prev = new_b
        self._t = t
        return new_b

class RWModel(object):
    def __init__(self, sigma2, initial_t=0):
        self.sigma = np.sqrt(sigma2)
        self._prev = 0

    def __call__(self, t):
        self._prev += np.random.normal(scale=self.sigma)
        return self._prev
        
class WNModel(object):
    def __init__(self, sigma2):
        self.sigma = np.sqrt(sigma2)
    
    def __call__(self, t):
        return np.random.normal(scale=self.sigma)

class MPU9250Gyroscope(IdealGyroscope):
    def __init__(self, platform, noiseStdDev, rng=None, **kwargs):
        if rng is None:
            rng = np.random.RandomState()
        Ts = 1. / 1000 # Sample time
        self._bias_components = [[
            MarkovProc.from_ar1(Ts, 9.999049e-01, 8.030220e-10),
            MarkovProc.from_ar1(Ts, 6.211672e-01, 4.546056e-03),
            RWModel(1.174606e-11),
            WNModel(5.876599e-06)
        ] for i in range(3)]

        # rad / s -> voltage scale factor
        #self._voltage_scale = 1. / 0.030517578125
        self._voltage_scale = 1. # Currently no rad/s -> volt scaling

        IdealGyroscope.__init__(self, platform, **kwargs)

    def noiseVoltages(self, t):
        br = np.array([sum(comp(t) for comp in axis_comp) for axis_comp in self._bias_components]).reshape(3,1)
        return self._voltage_scale * br
        
class MPU9250IMU(IdealIMU):
    def __init__(self, simulation=None, trajectory=None):
        # FIXME: Ugly loading two initalizers
        IdealIMU.__init__(self, simulation, trajectory)
        self.gyroscope = MPU9250Gyroscope(self, 0)
        StandardIMU.__init__(self, simulation, trajectory)
