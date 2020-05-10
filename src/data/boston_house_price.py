from sklearn.datasets import load_boston


class BostonHousePrice:
    data_description = """
    该数据集是一个回归问题。共有 506 个样本，13 个输入变量和1个输出变量。
    每条数据包含房屋以及房屋周围的详细信息。
    CRIM：城镇人均犯罪率；
    ZN：住宅用地超过 25000 sq.ft. 的比例；
    INDUS：城镇非零售商用土地的比例；
    CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）；
    NOX：一氧化氮浓度；
    RM：住宅平均房间数；
    AGE：1940 年之前建成的自用房屋比例；
    DIS：到波士顿五个中心区域的加权距离；
    RAD：辐射性公路的接近指数；
    TAX：每 10000 美元的全值财产税率；
    PTRATIO：城镇师生比例；
    B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例；
    LSTAT：人口中地位低下者的比例；
    MEDV：自住房的平均房价，以千美元计；
    """

    __data = load_boston()
    features = __data.data
    features_desc = __data.feature_names
    label = __data.target
    label_desc = ["house's price"]  # __data.target_names


if __name__ == '__main__':
    print(BostonHousePrice.data_description)
    print(BostonHousePrice.features_desc)
    print(BostonHousePrice.features.shape)
    print(BostonHousePrice.label_desc)
    print(BostonHousePrice.label.shape)
