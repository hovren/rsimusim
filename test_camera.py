import matplotlib.pyplot as plt
from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.platforms.imus import IdealIMU
from imusim.simulation.base import Simulation
from imusim.testing.random_data import RandomTrajectory

from rsimusim.camera import *
from rsimusim_legacy.world import *

# Create environment that contains landmarks
num_landmarks = 50
world_points = np.random.uniform(-100, 100, size=(3, num_landmarks))
world = NonBlockableWorld(world_points)
world_environment = WorldEnvironment(world)

sim = Simulation(environment=world_environment)
trajectory = RandomTrajectory()

# Create camera platform
camera_model = PinholeModel(np.eye(3), (1920, 1080), 1./35, 30.0)
camera = CameraPlatform(camera_model, simulation=sim, trajectory=trajectory)

# IMU platform
imu = IdealIMU(simulation=sim, trajectory=trajectory)

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
sim.run(trajectory.endTime - camera_model.readout)

PLOT = False
if PLOT:
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