#!/usr/bin/env python

import time

import matplotlib.pyplot as plt
import numpy as np
#from IPython import embed
# Import all public symbols from IMUSim
#from imusim.all import *
from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.trajectories.base import StaticTrajectory
from imusim.simulation.base import Simulation

from mpu9250 import MPU9250IMU


sim = Simulation()
trajectory = StaticTrajectory()
#wconst = np.array([0, 1, 0]).reshape(3,1)
#trajectory = StaticContinuousRotationTrajectory(wconst)

imu = MPU9250IMU(simulation=sim, trajectory=trajectory)
GYRO_SAMPLE_RATE = 1000.
dt = 1. / GYRO_SAMPLE_RATE

# Set up a behaviour that runs on the
# simulated IMU
behaviour1 = BasicIMUBehaviour(imu=imu, samplingPeriod=dt)

# Set the time inside the simulation
sim.time = trajectory.startTime
# Run the simulation till the desired
# end time
simulation_length = 3600*3
sim.run(simulation_length)

#plt.figure()
#plot(imu.gyroscope.rawMeasurements)
#plt.legend()
#plt.show()
t0 = time.time()
gdata = imu.gyroscope.rawMeasurements.values
savefilename = 'simulated_gyro.npy'
np.save(savefilename, gdata)
save_elapsed = time.time() - t0
print 'Saved {:d} samples to {}. Saving took {:.2f} seconds'.format(
    gdata.shape[1], savefilename, save_elapsed)
