import numpy as np


def accuracy(label, pred):
    return np.sum(label == pred) / len(label)
