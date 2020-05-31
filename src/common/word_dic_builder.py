class WordDic:
    def __init__(self):
        self.__word_id = {"UNKNOWN_WORDS": 1}
        self.__id_word = {1: "UNKNOWN_WORDS"}

    def build_dic(self, lists):
        for word_list in lists:
            for word in word_list:
                if word not in self.__word_id:
                    id = len(self.__word_id) + 1
                    self.__word_id[word] = id
                    self.__id_word[id] = word

    def word_ids(self, word_list):
        ids = []
        for word in word_list:
            ids.append(self.__word_id.get(word, 1))
        return ids

    def size(self):
        return len(self.__word_id)


if __name__ == '__main__':
    dic = WordDic()
    dic.build_dic([['这', '个', '粉', '底', '效', '果', '非', '常', '不', '错', '的', '说', '！', '非', '常', '喜', '欢', '！'],
                   ['【', ' ', '身', '体', '各', '个', '局', '部', '暴', '瘦', '的', '方', '法', ' ', '】']])

    print(dic.__word_id)
    print(dic.__id_word)
