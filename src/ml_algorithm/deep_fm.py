import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.callbacks import TensorBoard

"""
DeepFM 与 FM在输入层面是有区别的，FM不会对输入使用embedding技术；
而DeepFM是对输入先进行embedding之后，再作为输入，分表使用FM 和 DNN模型；

通常DeepFM的输入是libsvm格式的数据
有个细节，有两种实现方式：
   1. 连续特征不参与二阶和高阶的特征交叉，具体来讲就是不参与FM的二阶项和DNN
   2. 连续特征参与二阶和高阶的特征交叉
这里给出的是第1种实现方式
"""
data = pd.read_csv('..\\..\\data\\criteo_data\\criteo_data.txt', nrows=10000, delimiter="\t")
data.columns = ["label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12",
                "I13", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15",
                "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]

label = ['label']
dense_features = [f for f in data.columns if f[0] == "I"]
sparse_features = [f for f in data.columns if f[0] == "C"]
# -------处理连续特征--------
# 数值型特征空值填0
data[dense_features] = data[dense_features].fillna(0)
# 数值型特征归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = scaler.fit_transform(data[dense_features])
# -------处理离散特征---------
data[sparse_features] = data[sparse_features].fillna("-1")
print(data[sparse_features].head(100))
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

# 定义输入
dense_inputs = []
for fea in dense_features:
    _input = Input([1], name=fea)
    dense_inputs.append(_input)

concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
first_order_dense_layer = Dense(1)(concat_dense_inputs)  # W*x

sparse_inputs = []
for fea in sparse_features:
    _input = Input([1], name=fea)
    sparse_inputs.append(_input)

sparse_1d_embed = []
for i, _input in enumerate(sparse_inputs):
    f = sparse_features[i]
    voc_size = data[f].nunique()
    _embed = Flatten()(Embedding(voc_size + 1, 1)(_input))
    sparse_1d_embed.append(_embed)
first_order_sparse_layer = Add()(sparse_1d_embed)  # ADD操作就是求和，这里的逻辑等价于ΣWX，因为X是1
linear_part = Add()([first_order_dense_layer, first_order_sparse_layer])

# 二阶特征交叉
k = 8
sparse_kd_embed = []
for i, _input in enumerate(sparse_inputs):
    f = sparse_features[i]
    voc_size = data[f].nunique()
    _embed = Embedding(voc_size + 1, k)(_input)  # 这里不要Flatten打平
    sparse_kd_embed.append(_embed)
concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)

# 求和的平方，这里没有和x相乘，是因为X对应的值是1，没有相乘的必要
sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_sparse_kd_embed)
square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])
# 求平方和
square_kd_embed = Multiply()([concat_sparse_kd_embed, concat_sparse_kd_embed])
sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed)
# 汇总得到二阶交叉FM层
sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
sub = Lambda(lambda x: x * 0.5)(sub)
second_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)

# DNN层
flatten_sparse_embed = Flatten()(concat_sparse_kd_embed)
fc_layer = Dropout(0.5)(Dense(128, activation='relu')(flatten_sparse_embed))
fc_layer = Dropout(0.3)(Dense(64, activation='relu')(fc_layer))
fc_layer = Dropout(0.1)(Dense(32, activation='relu')(fc_layer))
fc_layer_output = Dense(1)(fc_layer)
output_layer = Add()([linear_part, second_order_sparse_layer, fc_layer_output])
output_layer = Activation("sigmoid")(output_layer)

# 模型构建
model = Model(dense_inputs + sparse_inputs, output_layer)
print(dense_inputs + sparse_inputs)
# plot_model(model, "deepfm.png")
model.summary()
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                         histogram_freq=0,
                         write_graph=True,
                         # write_grads=True,
                         write_images=True,
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)
train_data = data.loc[:8000 - 1]
valid_data = data.loc[8000:]
train_dense_x = [train_data[f].values for f in dense_features]
train_sparse_x = [train_data[f].values for f in sparse_features]
train_label = [train_data['label'].values]

val_dense_x = [valid_data[f].values for f in dense_features]
val_sparse_x = [valid_data[f].values for f in sparse_features]
val_label = [valid_data['label'].values]
print(train_dense_x + train_sparse_x)
model.fit(train_dense_x + train_sparse_x,
          train_label, epochs=5, batch_size=256,
          validation_data=(val_dense_x + val_sparse_x, val_label),
          callbacks=[tbCallBack]
          )
