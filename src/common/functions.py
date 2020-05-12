import numpy as np


def softmax(X):
    exp = np.power(np.e, X)
    element_sum = np.sum(exp)
    return exp / element_sum


def sigmoid(X):
    return 1 / (1 + np.power(np.e, -1 * X))


if __name__ == '__main__':
    result = softmax(np.array([1, 1, 1]))
    print(result)
