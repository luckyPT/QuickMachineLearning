import numpy as np

from visual import visual_tools
import matplotlib.pyplot as plt
import math


def origin_to_diff(point_seq):
    pre_seq = point_seq[:-1]
    next_seq = point_seq[1:]
    return next_seq - pre_seq


def diff_to_origin(diff_seq):
    point_seq = [diff_seq[0]]
    for i in range(1, len(diff_seq)):
        point_seq.append(diff_seq[i] + point_seq[-1])
    return np.array(point_seq)


def line_fit(point_seq):
    X, Y = point_seq[:, 0], point_seq[:, 1]
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    # Total number of values
    n = len(X)
    numer = 0
    denom = 0
    for i in range(n):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    m = numer / denom
    c = mean_y - (m * mean_x)
    # y = m * x + c
    return m, c


def pedal_point(point, k, b):
    x0 = point[0]
    y0 = point[1]
    x = (k * y0 + x0 - k * b) / (k ** 2 + 1)
    y = (k ** 2 * y0 + k * x0 + b) / (k ** 2 + 1)
    return x, y


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([2.5, 3.51, 4.45, 5.52, 6.47, 7.51])
    x = x[:, None]
    y = y[:, None]
    points = np.hstack((x, y))
    a, b = line_fit(points)
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.plot(x, y, 'ro', lw=2, markersize=6)
    plt.plot(x, a * x + b, 'r-', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel(u'X', fontsize=16)
    plt.ylabel(u'Y', fontsize=16)
    plt.show()
