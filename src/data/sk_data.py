from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits


class BreastCancer:
    data_description = """

    """
    __data = load_breast_cancer()
    features = __data.data
    features_desc = __data.feature_names
    label = __data.target
    label_desc = __data.target_names


class Minist:
    data_description = """

        """
    __data = load_digits()
    features = __data.data.reshape([-1, 8, 8, 1])
    features_desc = "8 * 8 的像素矩阵对应的灰度值"
    label = __data.target
    label_desc = "图片对应的手写数字"


if __name__ == '__main__':
    print(BreastCancer.features_desc)
    print(BreastCancer.features.shape)
    print(BreastCancer.label_desc)
    print(BreastCancer.label.shape)

    print(Minist.features)
    print(Minist.label)
