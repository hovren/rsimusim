from camera import *
from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.testing.random_data import RandomTrajectory
from imusim.simulation.base import Simulation
from imusim.platforms.imus import IdealIMU

import numpy as np
import matplotlib.pyplot as plt

sim = Simulation()
trajectory = RandomTrajectory()

# Create camera platform
camera = CameraPlatform(sim, trajectory)

# IMU platform
imu = IdealIMU(sim, trajectory)

# Set up a behaviour that runs on the
# simulated Camera
behaviour_camera = DefaultCameraBehaviour(camera)

# Set up a behaviour that runs on the
# simulated IMU
dt = 1. / 200
behaviour_imu = BasicIMUBehaviour(imu, dt)

# Set the time inside the simulation
sim.time = trajectory.startTime

# Run the simulation till the desired
# end time
sim.run(trajectory.endTime)

plt.figure()
plt.subplot(2,1,1)
t_camera = behaviour_camera.timestamps
t_imu = imu.gyroscope.rawMeasurements.timestamps
plt.axhline(1, color='k')
plt.axhline(2, color='k')
plt.plot(t_camera, 1 * np.ones_like(t_camera), 'o')
plt.plot(trajectory.startTime + t_imu, 2 * np.ones_like(t_imu), 'x')
plt.ylim(0, 3)
plt.subplot(2,1,2)
ts = np.linspace(trajectory.startTime, trajectory.endTime)
plt.plot(ts, trajectory.position(ts).T)
plt.show()