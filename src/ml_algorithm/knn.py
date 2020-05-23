from sklearn.neighbors import KNeighborsClassifier

from src.data.iris import Iris
from src.utils import data_split


class CustomKnn:
    def __init__(self, model):
        print("--init--")


if __name__ == '__main__':
    feature, label = Iris.features, Iris.label
    train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(train_feature, train_label)
    pred = knn_classifier.predict_proba(test_feature)
    print(pred)
