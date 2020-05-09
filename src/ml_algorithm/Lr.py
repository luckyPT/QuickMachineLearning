import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data.iris import Iris
import src.utils.data_split as data_split
import src.evaluate.classify_metrics as clf_metrics


class CustomLr:
    def __init__(self, model):
        self.w = np.transpose(model.coef_)
        self.b = model.intercept_

    def predict(self, feature):
        prob = np.dot(feature, self.w) + self.b
        return np.argmax(prob, axis=1)


if __name__ == '__main__':
    feature = Iris.features
    label = Iris.label
    train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
    clf = LogisticRegression(random_state=0).fit(train_feature, train_label)
    pred = clf.predict(test_feature)
    print("sklearn result:" + str(clf_metrics.accuracy(test_label, pred)))

    myLr = CustomLr(model=clf)
    myPred = myLr.predict(test_feature)
    print("my result:" + str(clf_metrics.accuracy(test_label, pred)))
