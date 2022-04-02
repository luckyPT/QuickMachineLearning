import fasttext as ft
import numpy as np


# https://stackoverflow.com/questions/54181163/fasttext-embeddings-sentence-vectors
# https://github.com/facebookresearch/fastText/issues/323
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / sum(np.exp(x))


dim = 2  # 词向量的维度
lr = 0.5  # 学习率
epoch = 50  # 迭代次数
min_count = 1  # 最小词频数
word_ngrams = 2  # 词语的前后关联程度
label_prefix = '__label__'

classifier = ft.train_supervised(input="./data.txt",
                                 label_prefix=label_prefix,
                                 dim=dim,
                                 minCount=min_count,
                                 lr=lr,
                                 epoch=epoch,
                                 word_ngrams=word_ngrams,
                                 verbose=2)

weight = classifier.get_output_matrix()
print("weight:", weight)
seq_vec = classifier.get_sentence_vector("一 了 会 但 出 包 变 变")
print("seq_vec:", seq_vec)
print("predict:", classifier.predict("一 了 会 但 出 包 变 变"))
print("custom predict:", softmax(np.dot(weight, seq_vec)))


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))
    # return np.sum(x ** 2)


# print(l2_norm(np.array([3, 4])))


def div_norm(x):
    norm_value = l2_norm(x)
    if norm_value > 0:
        return x * (1.0 / norm_value)
    else:
        return x


print("-------------------")
w2v1 = classifier.get_word_vector("变")
w2v2 = classifier.get_word_vector("了")
eos = classifier.get_word_vector('</s>')
# w2v2 = classifier.get_word_vector("变</s>")

print("seq vec", classifier.get_sentence_vector("变 了"))
for i in range(1, 8):
    for j in range(1, 8):
        for k in range(1, 8):
            print("custom seq vec:", (div_norm(w2v1) * i + div_norm(w2v2) * j + div_norm(eos) * k) / (i + j + k))
