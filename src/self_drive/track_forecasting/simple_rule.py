import numpy as np


def prediction_rule(traj, pre_points_count, avg_points=1):
    velocity_xs = []
    velocity_ys = []
    for i in range(len(traj) - 1, 0, -1):
        velocity_xs.append(traj[i][0] - traj[i - 1][0])
        velocity_ys.append(traj[i][1] - traj[i - 1][1])
    velocity_x = np.mean(velocity_xs)
    velocity_y = np.mean(velocity_ys)
    cur_points = list(traj[-1])
    pre_points = []
    for i in range(pre_points_count):
        cur_points[0] += velocity_x
        cur_points[1] += velocity_y
        pre_points.append(list(cur_points))
    return np.array(pre_points)


if __name__ == '__main__':
    print('---start---')

