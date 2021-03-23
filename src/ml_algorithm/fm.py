import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from data.boston_house_price import BostonHousePrice
import src.utils.data_split as ds

np.set_printoptions(precision=3, suppress=True)


class FmModel(Model):
    def __init__(self):
        super(FmModel, self).__init__()
        self.feature_count = 13

        self.b = tf.Variable(initial_value=0.01, dtype=tf.float32)
        self.w = tf.Variable(initial_value=0.01 * tf.random.normal(shape=[self.feature_count, 1], dtype=tf.float32))
        self.factor = tf.Variable(
            initial_value=0.01 * tf.random.normal(shape=[self.feature_count, 3], dtype=tf.float32))

    def call(self, inputs):
        linear_result = tf.squeeze(tf.matmul(inputs, self.w))
        # 和的平方
        factor_sum1 = tf.pow(tf.matmul(inputs, self.factor), 2)
        # 平方和
        factor_sum2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.factor, 2))

        return self.b + linear_result + 0.5 * tf.reduce_sum(factor_sum1 - factor_sum2, -1)


if __name__ == '__main__':
    features, labels = BostonHousePrice.features, BostonHousePrice.label
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean) / std

    train_feature, test_feature, train_label, test_label = ds.split(features, labels)
    print(train_feature.shape)
    model = FmModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    for epoch in range(100000):
        with tf.GradientTape() as tape:
            pred = model(train_feature)
            loss = mse_loss_fn(train_label, pred)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if epoch % 1000 == 0:
                pred = model(test_feature)
                print(mse_loss_fn(test_label, pred))
                print(np.average((np.abs(pred - test_label)) / test_label))
