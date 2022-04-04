import matplotlib.pyplot as plt

import numpy as np
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader


def plot_lane_border(polygon):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    xs, ys = polygon[:, 0], polygon[:, 1]
    plt.plot(xs, ys, '--', color='grey')


def plot_traj(traj, color="#d33e4c", line_width=2):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    plt.plot(traj[:, 0], traj[:, 1], color=color, linewidth=line_width)


def show():
    plt.show()
    # plt.clf()


if __name__ == '__main__':
    data_dir = '../../../data/avgoverse/motion_forecasting/data'
    afl = ArgoverseForecastingLoader(data_dir)
    plot_lane_border(afl.agent_traj)
    plt.show()
