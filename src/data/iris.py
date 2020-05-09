from sklearn.datasets import load_iris


class Iris:
    data_description = """
数据集内包含 3 类共 150 条记录，每类各 50 个数据，
每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
    """
    __data = load_iris()
    features = __data.data
    features_desc = __data.feature_names
    label = __data.target
    label_desc = __data.target_names


if __name__ == '__main__':
    print(Iris.data_description)
    print(Iris.features_desc)
    print(Iris.features.shape)
    print(Iris.label_desc)
    print(Iris.label.shape)
