import numpy as np

from self_drive.track_forecasting import eval_util
from self_drive.track_forecasting.eth.data_api import ETH_DATA


def prediction_rule(traj, pre_points_count, avg_points=1):
    velocity_xs = []
    velocity_ys = []
    for i in range(len(traj) - 1, 0, -1):
        velocity_xs.append(traj[i][0] - traj[i - 1][0])
        velocity_ys.append(traj[i][1] - traj[i - 1][1])
        avg_points -= 1
        if avg_points == 0:
            break
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
    test_data = ETH_DATA("../../../data/eth/eth_train.npz")
    test_feature = test_data.history_track
    test_label = test_data.future_track
    for i in range(1, 9):
        ades = []
        fdes = []
        for feature, label in zip(test_feature, test_label):
            pred = prediction_rule(feature, 12, i)
            ades.append(eval_util.ade(label, pred))
            fdes.append(eval_util.fde(label, pred))
        print(i, '\tade:\t', np.mean(ades))
        # print(i, '\tfde:\t', np.mean(fdes))
