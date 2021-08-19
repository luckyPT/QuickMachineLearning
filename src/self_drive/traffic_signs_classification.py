import os
# 数据源：http://www.nlpr.ia.ac.cn/pal/trafficdata/detection.html
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import numpy as np

np.set_printoptions(threshold=np.inf)

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print('GPU', tf.test.is_gpu_available())
print("---end---")


def image_paths_from_dir(dir):
    image_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            image_paths.append(root + "\\" + file)
    return image_paths


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192], method=ResizeMethod.BILINEAR)
    # image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def analyze_label(path):
    return tf.strings.to_number(tf.strings.split(tf.strings.split(path, "\\")[-1], "_")[0], out_type=tf.int32)


def image_paths2data_set(image_paths):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = path_ds.map(analyze_label, num_parallel_calls=AUTOTUNE)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_label_ds = image_label_ds.shuffle(buffer_size=4096)
    return image_label_ds


train_image_paths = image_paths_from_dir("../../data/traffic_sign/train")
test_image_paths = image_paths_from_dir("../../data/traffic_sign/test")
train_set = image_paths2data_set(train_image_paths).batch(batch_size=16)
test_set = image_paths2data_set(test_image_paths).batch(batch_size=100)

# model = tf.keras.applications.DenseNet121(weights=None)
model = tf.keras.models.load_model("./traffic_signs_classify1")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./training/model-{epoch:04d}-{val_accuracy:.3f}.ckpt",
    save_weights_only=True)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
model.fit(train_set, validation_data=test_set, epochs=15, callbacks=[cp_callback])
model.save("./traffic_signs_classify1")
"""
# 模型加载预测
load_model = tf.keras.applications.DenseNet121(weights=None)
load_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
latest = tf.train.latest_checkpoint("./training")
load_model.load_weights(latest)
load_model.evaluate(test_set)
"""

"""
# 自建模型
image = tf.keras.layers.Input(name="image", shape=(192, 192, 3))
conv1 = tf.keras.layers.Conv2D(2, 5, strides=3, padding='same', activation="relu")(image)
max_pool1 = tf.keras.layers.MaxPooling2D(3, strides=3)(conv1)
flatten = tf.keras.layers.Flatten()(max_pool1)
# dropout = tf.keras.layers.Dropout(rate=0.1)(flatten)
dense = tf.keras.layers.Dense(units=58, activation="softmax")(flatten)

model = tf.keras.Model(image, dense)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
model.fit(train_set, validation_data=test_set, epochs=5)
model.save("./traffic_signs_classify_custom")
model.evaluate(test_set)

load_model = tf.keras.models.load_model("./traffic_signs_classify_custom")
load_model.compile(metrics=['accuracy'])
load_model.evaluate(test_set)
model.evaluate(test_set)
"""
