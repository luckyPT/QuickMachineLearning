import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

"""
LR+FM+DEEP_FM
数据处理可参考：https://github.com/luckyPT/jvm-ml/tree/master/src/main/java/com/pt/ml/kaggle
"""
tf.config.experimental_run_functions_eagerly(True)
data_set = tf.data.experimental.make_csv_dataset(
    file_pattern="../../data/avazu_ctr/train_data.csv",
    batch_size=128,
    label_name="click",
    na_value="0",
    num_epochs=1,
    ignore_errors=True
)
features_info = {"C1Index": 6, "banner_posIndex": 6, "site_idIndex": 4737, "site_domainIndex": 7745,
                 "site_categoryIndex": 25, "app_idIndex": 8552, "app_domainIndex": 559, "app_categoryIndex": 35,
                 # "device_idIndex": 2686408,
                 "device_modelIndex": 8251, "device_typeIndex": 4,
                 "device_conn_typeIndex": 3,
                 "C14Index": 2626, "C15Index": 7, "C16Index": 8, "C17Index": 435, "C18Index": 3, "C19Index": 68,
                 "C20Index": 171, "C21Index": 60, "hourIndex": 23, "day_of_week": 7}
inputs = {k: tf.keras.layers.Input(name=k, shape=(), dtype='int32') for k in features_info}
feature_cols = [
    tf.feature_column.categorical_column_with_identity(key=col_name, num_buckets=features_info[col_name] + 1)
    for col_name in features_info]
embeddings = [tf.keras.layers.DenseFeatures([
    tf.feature_column.embedding_column(feature_col, 64)
])(inputs) for feature_col in feature_cols]

# one-hot input for LR
movie_ind_cols = [tf.feature_column.indicator_column(feature_col) for feature_col in
                  feature_cols]  # cols id indicator columns
lr_input = tf.keras.layers.DenseFeatures(movie_ind_cols)(inputs)
lr_out = tf.keras.layers.Dense(units=1, activation="sigmoid")(lr_input)

# FM
fm = tf.keras.layers.Concatenate(axis=1)([tf.expand_dims(embed, axis=1) for embed in embeddings])
sum_suqare = tf.pow(tf.reduce_sum(fm, axis=1), 2)
square_sum = tf.reduce_sum(tf.pow(fm, 2), axis=1)
fm_out = 0.5 * tf.reduce_sum(tf.subtract(sum_suqare, square_sum), axis=1)

# deep
deepInput = tf.keras.layers.Concatenate(axis=1)(embeddings)
deep1 = tf.keras.layers.Dense(units=256, activation="relu")(deepInput)
deep2 = tf.keras.layers.Dense(units=128, activation="relu")(deep1)
deep3 = tf.keras.layers.Dense(units=64, activation="relu")(deep2)
deep4 = tf.keras.layers.Dense(units=32, activation="relu")(deep3)
deep_out = tf.keras.layers.Dense(units=1, activation='sigmoid')(deep4)

# 求和
add = tf.keras.layers.Add()([lr_out, fm_out, deep_out])
out = tf.keras.layers.Activation(activation="sigmoid")(add)
model = tf.keras.Model(inputs, out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                         histogram_freq=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./checkPoint",
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(data_set, epochs=5,
          batch_size=256,
          validation_data=data_set,

          callbacks=[tbCallBack, cp_callback])
