from imusim.trajectories.splined import SampledTrajectory, SplinedTrajectory, SampledPositionTrajectory, SplinedPositionTrajectory
from imusim.maths.quaternions import Quaternion, QuaternionArray
from imusim.utilities.time_series import TimeSeries

import numpy as np
import matplotlib.pyplot as plt

def create_path():
    fig, ax = plt.subplots()
    spline_line, = ax.plot([], [], '-', label='spline', linewidth=6, alpha=0.5)
    kp_line, = ax.plot([], [], '-x', label='user', alpha=0.5)
    knot_line, = ax.plot([],[],'-o', label='knots', alpha=0.5, linewidth=2)
    print spline_line
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    keypoints = []
    marker_x = []
    marker_y = []

    global has_spline
    has_spline = False

    def onclick(event):
        global has_spline
        if has_spline:
            return

        zconst = 0.0
        keypoints.append((event.xdata, event.ydata, zconst))
        p_kp = np.array(keypoints, dtype='float64').T

        # CLose the loop?
        PAD_KNOTS = 3
        if len(keypoints) > PAD_KNOTS and np.linalg.norm(p_kp[:, 0] - p_kp[:, -1]) < 0.25:
            p_kp = p_kp[:, :-1] # Drop last sample
            p_knots = np.empty((3, p_kp.shape[1] + PAD_KNOTS*2 + 1))
            p_knots[:, PAD_KNOTS:-(PAD_KNOTS+1)] = p_kp
            p_knots[:, :PAD_KNOTS] = p_kp[:, -PAD_KNOTS:]
            p_knots[:, -(PAD_KNOTS+1):] = p_kp[:, :(PAD_KNOTS+1)]
            knot_line.set_data(p_knots[0], p_knots[1])
            fig.canvas.draw()
            t_kp = 0.5 * np.arange(p_knots.shape[1])
            ts = TimeSeries(t_kp, p_knots)
            samp = SampledPositionTrajectory(ts)
            splined = SplinedPositionTrajectory(samp)
            t = np.linspace(splined.startTime, splined.endTime, num=len(t_kp) * 50)
            p = splined.position(t)
            spline_line.set_data(p[0], p[1])
            has_spline = True

        kp_line.set_data(p_kp[0], p_kp[1])
        fig.canvas.draw()


    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.legend()
    plt.show()

create_path()
