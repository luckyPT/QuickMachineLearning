import numpy as np

from visual import visual_tools


class ETH_DATA:
    def __init__(self, npz_file_path):
        npz = np.load(npz_file_path, allow_pickle=True)
        self.history_track = npz['observations']
        self.history_speed = npz['obs_speed']
        self.future_track = npz['targets']
        self.target_speed = npz['target_speed']


if __name__ == '__main__':
    train_data = ETH_DATA("../../../../data/eth/eth_train.npz")
    for i in range(train_data.history_track.shape[0]):
        feature = train_data.history_track[i]
        target = train_data.future_track[i]
        visual_tools.points_group(feature, target)
    print('--------load complete---------')
