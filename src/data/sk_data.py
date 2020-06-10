from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris


class BreastCancer:
    data_description = """
    判断肿瘤的良性、恶性
    """
    __data = load_breast_cancer()
    features = __data.data
    features_desc = __data.feature_names
    label = __data.target
    label_desc = __data.target_names


class Minist:
    data_description = """
    手写数字数据集
    """
    __data = load_digits()
    features = __data.data.reshape([-1, 8, 8, 1])
    features_desc = "8 * 8 的像素矩阵对应的灰度值"
    label = __data.target
    label_desc = "图片对应的手写数字"


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
    print(BreastCancer.features_desc)
    print(BreastCancer.features.shape)
    print(BreastCancer.label_desc)
    print(BreastCancer.label.shape)
    print(Minist.features)
    print(Minist.label)

    print(Iris.data_description)
    print(Iris.features_desc)
    print(Iris.features.shape)
    print(Iris.label_desc)
    print(Iris.label.shape)
