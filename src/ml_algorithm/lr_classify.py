import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data.iris import Iris
from src.data.sk_data import BreastCancer
import src.utils.data_split as data_split
import src.evaluate.classify_metrics as clf_metrics
import src.common.functions as functions


class CustomLrMultiClassify:
    def __init__(self, model):
        self.w = np.transpose(model.coef_)
        self.b = model.intercept_

    def predict(self, feature):
        prob = np.dot(feature, self.w) + self.b
        return np.argmax(prob, axis=1)

    def predict_probas(self, feature):
        prob = np.dot(feature, self.w) + self.b
        return functions.softmax(prob)


class CustomLrBinClassify:
    def __init__(self, model):
        self.w = np.transpose(model.coef_)
        self.b = model.intercept_

    def predict(self, feature):
        prob = np.dot(feature, self.w) + self.b
        return np.argmax(prob, axis=1)

    def predict_probas(self, feature):
        prob = np.dot(feature, self.w) + self.b
        return functions.sigmoid(prob)


if __name__ == '__main__':
    """多分类"""
    """
    feature = Iris.features
    label = Iris.label
    train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
    clf = LogisticRegression(random_state=0).fit(train_feature, train_label)
    pred = clf.predict(test_feature)
    print("sklearn result:" + str(clf_metrics.accuracy(test_label, pred)))

    myLr = CustomLrMultiClassify(model=clf)
    myPred = myLr.predict(test_feature)
    print("my result:" + str(clf_metrics.accuracy(test_label, pred)))

    print("sklearn result = ", clf.predict_proba([test_feature[0]]))
    print("myresult = ", myLr.predict_probas(test_feature[0]))
    """

    """二分类"""
    bin_feature, bin_label = BreastCancer.features, BreastCancer.label
    train_bin_feature, test_bin_feature, train_bin_label, test_bin_label = data_split.split(bin_feature, bin_label)
    bin_clf = LogisticRegression()
    bin_clf.fit(train_bin_feature, train_bin_label)
    bin_pred = bin_clf.predict_proba(test_bin_feature)

    my_bin_classify = CustomLrBinClassify(bin_clf)
    my_bin_pred = my_bin_classify.predict_probas(test_bin_feature)
    print("sklearn result:", bin_pred[0:3][:, 1])
    print("my_result:", my_bin_pred[0:3].reshape([-1, ]))
