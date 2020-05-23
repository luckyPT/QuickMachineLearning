import math
import numpy as np
import src.evaluate.classify_metrics as metrics
from sklearn import svm

from src.data.sk_data import BreastCancer
import src.utils.data_split as ds


class CustomSvr:
    def __init__(self, svr_model):
        self.gamma = svr_model._gamma
        self.alpha = svr_model.dual_coef_  # .todense()
        self.support_vectors = svr_model.support_vectors_  # .todense()
        self.intercept = svr_model.intercept_[0]
        self.d = svr_model.degree
        self.r = svr_model.coef0
        k_fun = svr_model.kernel
        if 'linear' == k_fun:
            self.kernel = CustomSvr.linear_kernel
        elif 'poly' == k_fun:
            self.kernel = CustomSvr.polynomial_kernel
        elif 'rbf' == k_fun:
            self.kernel = CustomSvr.rbf_kernel
        elif 'sigmoid' == k_fun:
            self.kernel = CustomSvr.sigmoid_kernel
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
        for a, vec in zip(self.alpha.reshape(self.support_vectors.shape[0], -1), self.support_vectors):
            pre += a * self.kernel(self.gamma, self.r, self.d, features, vec)
        return np.asarray(pre)[0]


if __name__ == '__main__':
    features, labels = BreastCancer.features, BreastCancer.label
    train_feature, test_feature, train_label, test_label = ds.split(features, labels)
    regr = svm.SVR()
    regr.fit(train_feature, train_label) # 训练逻辑
    pred = regr.predict(test_feature) > 0.5 #预测逻辑
    print(metrics.accuracy(test_label, pred))
    # 手动实现向量机的前向传播计算
    my_model = CustomSvr(regr)
    my_pred = my_model.predict(test_feature) > 0.5
    print(metrics.accuracy(test_label, pred))
