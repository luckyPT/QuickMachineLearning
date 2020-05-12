import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data.iris import Iris
from src.data.sk_data import BreastCancer
import src.utils.data_split as data_split
import src.evaluate.classify_metrics as clf_metrics
import src.common.functions as functions
import sklearn.naive_bayes as bayes


class CustomGaussianBayes:
    def __init__(self, model):
        self.class_prior_ = model.class_prior_
        self.classes_ = model.classes_
        self.sigma_ = model.sigma_
        self.theta_ = model.theta_

    def predict_proba(self, x):
        jll = self._joint_log_likelihood(x)
        log_prob_x = np.log(np.sum(np.exp(jll), axis=1))
        y = jll - np.atleast_2d(log_prob_x).T
        return np.exp(y)

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


if __name__ == '__main__':
    feature = Iris.features
    label = Iris.label
    train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
    nativeBayes = bayes.GaussianNB()
    nativeBayes.fit(train_feature, train_label)
    pred = nativeBayes.predict_proba(test_feature)[0]
    print("pred", pred)
    myBayes = CustomGaussianBayes(nativeBayes)
    my_pred = myBayes.predict_proba(test_feature)[0]
    print("my_pred", my_pred)
