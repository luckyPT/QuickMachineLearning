import numpy as np

np.set_printoptions(suppress=True)
from sklearn import linear_model
from src.data.boston_house_price import BostonHousePrice
import src.utils.data_split as ds
import src.evaluate.regression_metrics as metrics


class CustomLrRegression:
    def __init__(self, model):
        self.w = np.transpose(model.coef_)
        print("w = ", self.w)
        self.b = model.intercept_
        print("b = ", self.b)

    def predict(self, feature):
        result = np.dot(feature, self.w) + self.b
        return result


if __name__ == '__main__':
    features, labels = BostonHousePrice.features, BostonHousePrice.label
    train_feature, test_feature, train_label, test_label = ds.split(features, labels)

    reg = linear_model.LinearRegression()  # 最小方差
    # reg = linear_model.Ridge(alpha=.5) #L2正则化
    # reg = linear_model.Lasso() #L1正则化
    # reg = linear_model.ElasticNet(alpha=0.3, l1_ratio=0.7)  # L1 L2融合
    reg.fit(train_feature, train_label)
    pred = reg.predict(test_feature)
    print(metrics.error_ratio(test_label, pred))

    my_model = CustomLrRegression(reg)
    my_pred = my_model.predict(test_feature)
    print(metrics.error_ratio(test_label, my_pred))
    print("mae = ", metrics.mean_absolute_error(test_label, my_pred))
