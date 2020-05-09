import sklearn.model_selection as ms


def split(feature, label, test_ratio=0.3):
    train_x, test_x, train_y, test_y = ms.train_test_split(feature, label, test_size=test_ratio)
    return train_x, test_x, train_y, test_y
