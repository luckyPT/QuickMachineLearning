import sklearn.metrics as metrics


def accuracy(label, pred):
    return metrics.accuracy_score(label, pred)
