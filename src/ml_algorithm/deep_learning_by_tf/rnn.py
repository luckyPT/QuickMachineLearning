from tensorflow.keras import Model
import tensorflow as tf
from src.common.segmenter import CharSegmenter
from src.data.text_classify1 import TextClassify1
import src.utils.data_split as data_split
import numpy as np
import src.common.activations as activations
import src.evaluate.classify_metrics as classify_metrics


class MyRnnModel(Model):
    def __init__(self, word_dic_size):
        super(MyRnnModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(word_dic_size, 64)
        self.rnn1 = tf.keras.layers.GRU(units=128)
        self.dense1 = tf.keras.layers.Dense(units=4, activation=tf.keras.activations.softmax)
        self.complete_custom = False

    def call(self, x):
        embeded = self.embedding(x)
        rnn1_result = self.rnn1(embeded)
        out = self.dense1(rnn1_result)
        return out

    def init_custom_params(self):
        self.word_vec = self.embedding.embeddings.numpy()
        self.rnn1_kernel = self.rnn1.weights[0].numpy()
        self.rnn1_recurent_kernel = self.rnn1.weights[1].numpy()
        self.rnn1_bias = self.rnn1.weights[2].numpy()
        self.rnn1_recurent_activation = self.rnn1.recurrent_activation
        self.rnn1_activation = self.rnn1.activation

        self.d1_w = self.dense1.kernel.numpy()
        self.d1_b = self.dense1.bias.numpy()
        self.d1_activation = self.dense1.activation
        print("--")

    def custom_predict(self, input):
        if not self.complete_custom:
            self.init_custom_params()

        embeded = []
        for id in input:
            embeded.append(self.word_vec[id])
        embeded = np.array(embeded)
        last_state = np.zeros([128, ])
        for i in range(len(input)):
            step_input = np.array(embeded[i])
            matrix_x = np.dot(step_input, self.rnn1_kernel)
            matrix_x = np.add(matrix_x, self.rnn1_bias[0])
            x_z, x_r, x_h = np.split(matrix_x, 3, axis=0)
            matrix_inner = np.dot(last_state, self.rnn1_recurent_kernel)
            matrix_inner = np.add(matrix_inner, self.rnn1_bias[1])
            recurrent_z, recurrent_r, recurrent_h = np.split(matrix_inner, 3, axis=0)
            z = self.rnn1_recurent_activation(x_z + recurrent_z)
            r = self.rnn1_recurent_activation(x_r + recurrent_r)
            hh = self.rnn1_activation(x_h + r * recurrent_h)
            last_state = z * last_state + (1 - z) * hh
        d1 = activations.softmax(np.dot(last_state.numpy(), self.d1_w) + self.d1_b)
        return d1


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # print(loss.numpy())


def test_step(images, labels):
    predictions = model(images, training=False)
    print(predictions)
    predictions = np.argmax(predictions, axis=1)
    print("accuracy = ", classify_metrics.accuracy(labels, predictions))


if __name__ == '__main__':
    data = TextClassify1(CharSegmenter.segment)
    features, labels = data.features, data.labels
    train_bin_feature, test_bin_feature, train_bin_label, test_bin_label = data_split.split(features, labels)
    model = MyRnnModel(data.word_dic.size() + 1)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS = 50
    BATCH_SIZE = 32
    for i in range(EPOCHS):
        for batch in range(0, len(train_bin_feature) - BATCH_SIZE, BATCH_SIZE):
            print(i, batch)
            train_step(train_bin_feature[batch:batch + BATCH_SIZE][:, :100], train_bin_label[batch:batch + BATCH_SIZE])
        print("-------------------------")
        test_step(test_bin_feature[0:BATCH_SIZE][:, :100], test_bin_label[0:BATCH_SIZE])
        for i in range(10):
            result = model.custom_predict(test_bin_feature[0:BATCH_SIZE][:, :100][i])
            print(result)
