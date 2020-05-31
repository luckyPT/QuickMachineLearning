import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from src.data.boston_house_price import BostonHousePrice
import src.utils.data_split as ds
from src.evaluate import regression_metrics


class MyDnnModel(Model):
    def __init__(self):
        super(MyDnnModel, self).__init__()
        self.dense1 = Dense(10, activation=tf.keras.activations.relu)
        self.dense2 = Dense(5, activation=tf.keras.activations.elu)
        self.dense3 = Dense(1, )
        self.complete_custom = False

    def call(self, inputs):
        d1 = self.dense1(inputs)
        d2 = self.dense2(d1)
        pred = self.dense3(d2)
        return pred

    def init_custom_params(self):
        self.d1_w = self.dense1.kernel.numpy()
        self.d1_b = self.dense1.bias.numpy()
        self.d1_activation = self.dense1.activation

        self.d2_w = self.dense2.kernel.numpy()
        self.d2_b = self.dense2.bias.numpy()
        self.d2_activation = self.dense2.activation

        self.d3_w = self.dense3.kernel.numpy()
        self.d3_b = self.dense3.bias.numpy()
        self.d3_activation = self.dense3.activation

        self.complete_custom = True

    def custom_predict(self, inputs):
        if not self.complete_custom:
            self.init_custom_params()

        d1 = self.d1_activation(np.dot(inputs, self.d1_w) + self.d1_b)
        d2 = self.d2_activation(np.dot(d1, self.d2_w) + self.d2_b)
        d3 = self.d3_activation(np.dot(d2, self.d3_w) + self.d3_b)
        return d3


model = MyDnnModel()
loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adamax()


# @tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# @tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    print("loss = ", loss_object(labels, predictions).numpy())
    print("my_metrics = ", regression_metrics.error_ratio(labels, predictions))


if __name__ == '__main__':
    features, labels = BostonHousePrice.features, BostonHousePrice.label
    train_feature, test_feature, train_label, test_label = ds.split(features, labels)

    EPOCHS = 1000
    for epoch in range(EPOCHS):
        train_step(train_feature, train_label)
    test_step(test_feature, test_label)
    pred = model.call(test_feature)
    my_pred = model.custom_predict(test_feature)
    print(pred[0].numpy(), my_pred[0])
