import numpy as np


def error_ratio(label, pred):
    ratios = np.abs(label - pred) / label
    return np.average(ratios)


def mean_absolute_error(label, pred):
    return np.average(np.abs(label - pred))
