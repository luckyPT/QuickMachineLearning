import numpy as np
from src.data.iris import Iris
import src.utils.data_split as data_split
import src.evaluate.classify_metrics as clf_metrics
from sklearn import tree
import matplotlib.pyplot as plt
import time


class CustomDecisionTree:
    def __init__(self, tree_model):
        print("---")


if __name__ == '__main__':
    feature, label = Iris.features, Iris.label
    train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_feature, train_label)

    pred = clf.predict(test_feature)
    print(clf_metrics.accuracy(test_label, pred))
    tree.plot_tree(clf)  # 比较旧的版本不支持这个函数
    plt.show()
    print("-----end----")
