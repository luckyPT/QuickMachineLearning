from mittens import GloVe
import numpy as np

# 构造共现矩阵
cooccurrence = np.array([
    [4., 4., 2., 0.],
    [4., 61., 8., 18.],
    [2., 8., 10., 0.],
    [0., 18., 0., 5.]])
glove_model = GloVe(n=6, max_iter=100)
embeddings = glove_model.fit(cooccurrence)
print(embeddings)
