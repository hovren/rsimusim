import numpy as np
from imusim.maths.quaternions import Quaternion, QuaternionArray
from numpy import testing as nt

from crisp.fastintegrate import integrate_gyro_quaternion_uniform

def random_position():
    return np.random.uniform(-10, 10, size=3)


def random_orientation():
    n = np.random.uniform(-1, 1, size=3)
    n /= np.linalg.norm(n)
    phi = np.random.uniform(-np.pi, np.pi)
    qparts = (np.cos(phi/2), n * np.sin(phi/2))
    qdata = np.hstack(qparts)
    q = Quaternion(*qdata)
    return q


def random_focal():
    return np.random.uniform(100., 1000.)

def unpack_quat(q):
    return np.array([q.w, q.x, q.y, q.z])

def gyro_data_to_quaternion_array(gyro_data, gyro_times):
    dt = float(gyro_times[1] - gyro_times[0])
    nt.assert_almost_equal(np.diff(gyro_times), dt)
    q = integrate_gyro_quaternion_uniform(gyro_data, dt)
    return QuaternionArray(q)
