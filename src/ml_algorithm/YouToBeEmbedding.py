# 论文地址：https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf
# 非官方工程示例：https://blog.csdn.net/wdh315172/article/details/106581377
"""
模拟数据生成：
    val schema = StructType(List(
      StructField("his_ids", ArrayType(IntegerType, containsNull = true)),
      StructField("age", FloatType),
      StructField("city", IntegerType),
      StructField("pre_ids", ArrayType(IntegerType, containsNull = true)),
      StructField("label", ArrayType(IntegerType, containsNull = true))))
    val dataBuffer = ArrayBuffer[Row]()
    val itemCount = 100000
    for (_ <- 0 until 100000) {
      val his_ids = List(new Random().nextInt(itemCount), new Random().nextInt(itemCount),
        new Random().nextInt(itemCount), new Random().nextInt(itemCount), new Random().nextInt(itemCount))
      val age = new Random().nextInt(100) / 100.0F
      val city = new Random().nextInt(4)
      val pre_ids = List(new Random().nextInt(itemCount), new Random().nextInt(itemCount), new Random().nextInt(itemCount))
      val label = List(1, 0, 0)
      dataBuffer.append(new GenericRow(Array[Any](his_ids, age, city, pre_ids, label)))
    }
    val dataArray = dataBuffer.toArray
    val rdd = spark.sparkContext.parallelize(dataArray)
    val df2: DataFrame = spark.createDataFrame(rdd, schema)
    df2.repartition(1).write.format("tfrecords").option("recordType", "Example").save("./tfrecord")
"""
import numpy as np
import tensorflow as tf

# 输入：5个最近访问的itemId，年龄，城市，预测的id（第一个是正样本，其余是负样本），label（1，0，0...）
item_count = 100000
item_vec_size = 128
raw_dataset = tf.data.TFRecordDataset("../../data/youtube/YouTuBeRecall.tfrecord")
feature_description = {
    'his_ids': tf.io.FixedLenFeature([5], tf.int64),
    'age': tf.io.FixedLenFeature([], tf.float32),
    'city': tf.io.FixedLenFeature([], tf.int64),
    'pre_ids': tf.io.FixedLenFeature([3], tf.int64),
    'label': tf.io.FixedLenFeature([3], tf.int64)
}


def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example, example["label"]


dataset = raw_dataset.map(_parse_function).batch(16)

item_vec = np.random.randint(0, 1000, [item_count, item_vec_size]) / 1000
item_ids = tf.keras.layers.Input(name="his_ids", shape=5)
user_city = tf.keras.layers.Input(name="city", shape=())
user_age = tf.keras.layers.Input(name="age", shape=1)
label_item_ids = tf.keras.layers.Input(name="pre_ids", shape=3)

item_vec_layer = tf.keras.layers.Embedding(item_count, item_vec_size, input_length=5, weights=[item_vec],
                                           trainable=False)(item_ids)
item_vec_layer = tf.reduce_mean(item_vec_layer, axis=-2)
city_Embedding = tf.keras.layers.Embedding(4, 64, input_length=1)(user_city)

user_info = tf.keras.layers.Concatenate(axis=-1)([item_vec_layer, city_Embedding, user_age])

dense1 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(user_info)
dense2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(dense1)

new_item_embedding = tf.keras.layers.Embedding(item_count, item_vec_size, input_length=3)(label_item_ids)
out = tf.matmul(new_item_embedding, tf.expand_dims(dense2, -1))
out = tf.keras.layers.Softmax(name="label", axis=-1)(tf.squeeze(out, axis=-1))

model = tf.keras.Model([item_ids, user_city, user_age, label_item_ids], out)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(dataset, epochs=10, validation_data=dataset)
