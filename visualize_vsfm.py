from collections import namedtuple
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

import crisp.rotations

CameraPose = namedtuple('CameraPose', ['filename', 'orientation', 'position'])
WorldPoint = namedtuple('WorldPoint', ['position','color'])

class NvmModel(object):
    def __init__(self):
        self.cameras = []
        self.points = []

    def add_camera(self, filename, q, p):
        self.cameras.append(CameraPose(filename, q, p))

    def add_point(self, xyz, rgb):
        self.points.append(WorldPoint(xyz, rgb))

    def _normalize_world(self):
        # Sort cameras by filename
        self.cameras.sort(key=lambda c: c.filename)

        # Rotate cameras and world points
        R0 = crisp.rotations.quat_to_rotation_matrix(self.cameras[0].orientation)


    @classmethod
    def from_file(cls, filename):
        instance = cls()
        num_cameras = 0
        num_points = 0
        state = 'header'
        for i, line in enumerate(open(filename, 'r')):
            line = line.strip()
            if not line or line[0] == '#':
                continue

            if state == 'header':
                if line == 'NVM_V3':
                    state = 'num_cameras'
                else:
                    raise ValueError("Expected NVM_V3, got {}".format(line))

            elif state == 'num_cameras':
                try:
                    num_cameras = int(line)
                    state = 'cameras'
                    current_camera = 0
                except ValueError:
                    raise ValueError("Expected number of cameras, got {}".format(line))

            elif state == 'cameras':
                tokens = line.split()
                try:
                    #print tokens
                    params = map(float, tokens[-10:])
                except IndexError:
                    raise ValueError("Failed to parse camera")
                focal, qw, qx, qy, qz, px, py, pz, radial, _ = params
                filename = ''.join(tokens[:-10])
                filename = os.path.split(filename)[-1]
                q = np.array([qw, qx, qy, qz])
                p = np.array([px, py, pz])
                instance.add_camera(filename, q, p)

                if len(instance.cameras) >= num_cameras:
                    state = 'num_points'

            elif state == 'num_points':
                try:
                    num_points = int(line)
                    state = 'points'
                except ValueError:
                    raise ValueError("Expected number of points, got {}".format(line))

            elif state == 'points':
                tokens = line.split()
                position = np.array(map(float, tokens[:3]))
                color = np.array(map(int, tokens[3:6]), dtype='uint8')
                instance.add_point(position, color)

                if len(instance.points) >= num_points:
                    state = 'model_done'

            elif state == 'model_done':
                if not line == '0':
                    raise ValueError("Expected 0 to end model section, got {}".format(line))
                state = 'all_done'

            elif state == 'all_done':
                if not line == '0':
                    raise ValueError("Expected 0 to end file, got {}".format(line))
                state = 'finish'

            elif state == 'finish':
                if line:
                    raise ValueError("Expected nothing, got {}".format(line))

            else:
                raise ValueError("Unknown state {}".format(state))

        # Done, return normalized instance
        instance._normalize_world()
        return instance

filename = 'walk_model.nvm'
model = NvmModel.from_file(filename)
print 'Loaded {:d} cameras and {:d} points from {}'.format(len(model.cameras), len(model.points), filename)


camera_positions = np.vstack([camera.position for camera in model.cameras])
world_points = np.vstack([point.position for point in model.points])
point_colors = np.vstack([tuple(point.color) + (1,) for point in model.points]).astype('uint8')
print point_colors.shape
cam1 = model.cameras[0]
print cam1

mlab.plot3d(camera_positions[:,0], camera_positions[:,1], camera_positions[:,2], color=(1, 0, 0))
N = len(model.points)
ones = np.ones(N)
scalars = np.arange(N)
mlab.points3d(world_points[:, 0], world_points[:, 1], world_points[:, 2], scale_factor=0.1)
mlab.show()

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(camera_positions[:,0], camera_positions[:, 1], camera_positions[:, 2], '-o', linewidth=5, markersize=10, color='b')
    ax.scatter(world_points[:,0], world_points[:,1], world_points[:,2], marker='x', alpha=0.5, c=point_colors)
    plt.show()


