from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.data.iris import Iris
from src.utils import data_split

if __name__ == '__main__':
    feature, label = Iris.features, Iris.label
    train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    bagging.fit(train_feature, train_label)
    pred = bagging.predict_proba(test_feature)
    print(pred)
