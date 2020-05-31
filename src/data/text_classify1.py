import numpy as np
from src import CONSTANT_VAR
from src.common.segmenter import CharSegmenter
from src.common.word_dic_builder import WordDic


class TextClassify1:
    def __init__(self, segmente_fun):
        self.word_dic = WordDic()
        data_description = """
       文本分类数据集
           """
        classies = {"体育.txt": 0, "女性.txt": 1, "文学出版.txt": 2, "校园.txt": 3}
        words_list = []
        for one_class in classies:
            with open(CONSTANT_VAR.ROOT_PATH + "data//text_classify1//" + one_class, encoding="UTF-8",
                      mode="r") as file:
                lines = list(map(segmente_fun, file.readlines()))
                words_list.append(lines)
                self.word_dic.build_dic(lines)
        features = []
        labels = []
        max_len = 0
        for datas, label_name in zip(words_list, classies):
            for data in datas:
                ids = self.word_dic.word_ids(data)
                features.append(ids)
                labels.append(classies[label_name])
                max_len = max_len if max_len > len(ids) else len(ids)
        for feature in features:
            for _ in range(max_len - len(feature)):
                feature.append(0)

        self.features = np.array(features)
        self.labels = np.array(labels)
        self.features_desc = ""
        self.labels_Desc = ""


if __name__ == '__main__':
    data = TextClassify1(CharSegmenter.segment)
    print(data.features.shape)
    print(data.labels.shape)
