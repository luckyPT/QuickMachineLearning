import math

import numpy as np
from sklearn import svm
import sklearn

import src.utils.data_split as ds
from src.data.sk_data import BreastCancer


class CustomSvm:
    """
    sklearn 版本：0.24.1
    """
    def __init__(self, svr_model):
        self.gamma = svr_model._gamma
        self.alpha = svr_model.dual_coef_  # .todense()
        self.support_vectors = svr_model.support_vectors_  # .todense()
        self.intercept = svr_model.intercept_[0]
        self.d = svr_model.degree
        self.r = svr_model.coef0
        self.n_support = svr_model.n_support_
        self.probA_ = svr_model.probA_
        self.probB_ = svr_model.probB_
        k_fun = svr_model.kernel
        if 'linear' == k_fun:
            self.kernel = CustomSvm.linear_kernel
        elif 'poly' == k_fun:
            self.kernel = CustomSvm.polynomial_kernel
        elif 'rbf' == k_fun:
            self.kernel = CustomSvm.rbf_kernel
        elif 'sigmoid' == k_fun:
            self.kernel = CustomSvm.sigmoid_kernel
        else:
            raise Exception("unsupported kernel function：" + k_fun)

    @staticmethod
    def rbf_kernel(in_gamma, r, d, x1, x2):
        """
        i_sum = 0
        for i in range(x2.shape[1]):
            i_sum += math.pow(x1[0, i] - x2[0, i], 2)
            print(i, i_sum)
        i_sum = -1 * i_sum * in_gamma
        print(i_sum)
        return math.exp(i_sum)
        """
        return math.pow(math.e, -1 * in_gamma * np.sum(np.square(x1 - x2)))

    @staticmethod
    def linear_kernel(in_gamma, r, d, x1, x2):
        return np.dot(x1, x2.T)

    @staticmethod
    def sigmoid_kernel(in_gamma, r, d, x1, x2):
        return math.tanh(in_gamma * np.dot(x1, x2.T) + r)

    @staticmethod
    def polynomial_kernel(in_gamma, r, d, x1, x2):
        return math.pow(in_gamma * np.dot(x1, x2.T) + r, d)

    def predict(self, features):
        pre = self.intercept
        for a, vec in zip(self.alpha.reshape(self.support_vectors.shape[0], -1),self.support_vectors):
            pre += a * self.kernel(self.gamma, self.r, self.d, features, vec)

        prob = 1 / (1 + np.exp(-pre * self.probA_ + self.probB_))
        return prob


if __name__ == '__main__':
    print(sklearn.__version__)
    features, labels = BreastCancer.features, BreastCancer.label
    train_feature, test_feature, train_label, test_label = ds.split(features, labels)
    regr = svm.SVC(probability=True)
    regr.fit(train_feature, train_label)  # 训练逻辑
    my_model = CustomSvm(regr)
    diff = []
    for i in range(len(test_feature)):
        probs = regr.predict_proba([test_feature[i]])
        my_pred = my_model.predict([test_feature[i]])
        diff.append(abs(my_pred[0] - probs[0][0]))
    diff.sort(reverse=True)
    print(np.average(diff))
    print("----------------------")
    print(diff)
