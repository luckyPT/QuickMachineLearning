import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from src.data.sk_data import Minist
import src.evaluate.classify_metrics as classify_metrics


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
        self.complete_custom = False

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def init_custom_params(self):
        self.conv1_kernel = self.conv1.kernel.numpy()
        self.conv1_bias = self.conv1.bias.numpy()
        self.conv1_activation = self.conv1.activation
        self.conv1_stride = self.conv1.strides
        self.conv1_deepth = self.conv1.filters

        self.d1_w = self.d1.kernel.numpy()
        self.d1_b = self.d1.bias.numpy()
        self.d1_activation = self.d1.activation

        self.d2_w = self.d2.kernel.numpy()
        self.d2_b = self.d2.bias.numpy()
        self.d2_activation = self.d2.activation

        self.complete_custom = True

    def custom_predict(self, x):
        if not self.complete_custom:
            self.init_custom_params()
        conv1_result = []
        for row in range(0, x.shape[-3] - self.conv1_kernel.shape[0] + 1, self.conv1_stride[1]):
            row_result = []
            for col in range(0, x.shape[-2] - self.conv1_kernel.shape[1] + 1, self.conv1_stride[0]):
                cur_area = x[row:row + self.conv1_kernel.shape[0], col:col + self.conv1_kernel.shape[1], :]
                one_pixel = []
                for deep in range(0, self.conv1_deepth):
                    deep_value = np.sum(np.multiply(cur_area, self.conv1_kernel[:, :, :, deep]))
                    one_pixel.append(deep_value)
                one_pixel = np.array(one_pixel).T
                one_pixel += self.conv1_bias
                row_result.append(one_pixel)
            row_result = np.array(row_result)
            conv1_result.append(row_result)
        conv1_result = np.array(conv1_result)
        conv1_result = self.conv1_activation(conv1_result)
        conv1_result = np.reshape(conv1_result, [-1, ])

        d1 = self.d1_activation(np.dot(conv1_result, self.d1_w) + self.d1_b)
        d2 = self.d2_activation(np.dot(d1, self.d2_w) + self.d2_b)
        print(d2)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# @tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    print(predictions[0])
    predictions = np.argmax(predictions, axis=1)
    print("accuracy = ", classify_metrics.accuracy(labels, predictions))


if __name__ == '__main__':
    features, labels = Minist.features, Minist.label
    # Create an instance of the model
    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS = 50
    for epoch in range(EPOCHS):
        train_step(features, labels)
    test_step(features, labels)
    model.custom_predict(features[0])
