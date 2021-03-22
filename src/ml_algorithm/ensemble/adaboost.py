from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from src.data.sk_data import Iris
from src.utils import data_split
from src.evaluate import classify_metrics

if __name__ == '__main__':
    feature, label = Iris.features, Iris.label
    train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(train_feature, train_label)
    pred = clf.predict(test_feature)
    accuracy = classify_metrics.accuracy(test_label, pred)
    print("accuracy = ", accuracy)
