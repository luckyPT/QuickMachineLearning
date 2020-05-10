import numpy as np


def error_ratio(label, pred):
    ratios = np.abs(label - pred) / label
    return np.average(ratios)
