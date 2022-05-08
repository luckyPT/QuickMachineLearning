import numpy as np

from self_drive.track_forecasting import track_util, eval_util
from self_drive.track_forecasting.eth.data_api import ETH_DATA
from visual import visual_tools


def line_fit_track(traj, pre_points_count, speed_avg_points=1):
    velocity_xs = []
    velocity_ys = []
    for i in range(len(traj) - 1, 0, -1):
        velocity_xs.append(traj[i][0] - traj[i - 1][0])
        velocity_ys.append(traj[i][1] - traj[i - 1][1])
        speed_avg_points -= 1
        if speed_avg_points == 0:
            break
    velocity_x = np.mean(velocity_xs)
    velocity_y = np.mean(velocity_ys)
    time_dis = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

    k, b = track_util.line_fit(traj)
    start_x, start_y = traj[-1]  # track_util.pedal_point(traj[-1], k, b)
    dx = time_dis / np.sqrt(k ** 2 + 1)
    dy = k * dx
    pred = []
    for i in range(pre_points_count):
        if velocity_x > 0:
            pred.append([start_x + dx, start_y + dy])
            start_x += dx
            start_y += dy
        else:
            pred.append([start_x - dx, start_y - dy])
            start_x -= dx
            start_y -= dy
    return np.array(pred)


if __name__ == '__main__':
    print('---start---')
    test_data = ETH_DATA("../../../data/eth/eth_test.npz")
    test_feature = test_data.history_track
    test_label = test_data.future_track
    for his_count in range(0, 7):
        for i in range(1, 9):
            ades = []
            fdes = []
            for feature, label in zip(test_feature, test_label):
                pred = line_fit_track(feature[his_count:], 12, i)
                ades.append(eval_util.ade(label, pred))
                fdes.append(eval_util.fde(label, pred))
                # visual_tools.points_group(feature, label, pred)
            print(his_count, i, '\tade:\t', np.nanmean(ades))
