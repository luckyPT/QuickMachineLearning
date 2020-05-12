from sklearn.datasets import load_breast_cancer


class BreastCancer:
    data_description = """

    """
    __data = load_breast_cancer()
    features = __data.data
    features_desc = __data.feature_names
    label = __data.target
    label_desc = __data.target_names


if __name__ == '__main__':
    print(BreastCancer.features_desc)
    print(BreastCancer.features.shape)
    print(BreastCancer.label_desc)
    print(BreastCancer.label.shape)
