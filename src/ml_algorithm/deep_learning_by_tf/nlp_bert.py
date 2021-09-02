"""
模型结构：
Bert:Bidirectional Enoceder Representations from Transformers双向的Transformers的Encoder
以Transfoermer结构为基础，只取它的Encoder结构
Bert论文：https://arxiv.org/pdf/1810.04805.pdf
输入：token sequence
如果是单个句子：为句子开头增加“[CLS]” token；最终这个token对应的输出用于分类任务
如果是句子对：则第一个句子开头加“[CLS]”，句子之间用“[SEP]”分开，同时会额外增加一个标识，标识这个token是句子A还是句子B
所以汇总下来，一共是三部分：token id序列，标识AB的序列，位置序列；（其中token序列和位置序列是transformer固有的，标识AB的序列是Bert特有的）
输出：
每一个token的编码
双向self-attention的构造：

训练逻辑：
    Masked LM + Next Sentence Prediction
    前者可以认为是基于上下文预测中心词；后者是一个分类问题，预测后一个句子是否是前一个句子的下一句
应用Demo如下：
"""
import tensorflow_text as text
import tensorflow_hub as hub
import tensorflow as tf


def bert_single_seqs_demo(seqs):
    sentences = tf.constant([seqs])
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_zh_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4", trainable=True)
    outputs = encoder(encoder_inputs)
    # sequence_output、encoder_outputs、default、pooled_output
    pooled_output = outputs["pooled_output"]  # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

    embedding_model = tf.keras.Model(text_input, sequence_output)
    embedding_model.summary()
    out = embedding_model(sentences)
    print(out)


def bert_pairs_seqs_demo(seq1, seq2):
    """
    会将seq1、seq2处理成：[CLS]seq1[SEP]seq2[SEP]
    """
    preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_zh_preprocess/3")
    # Step 1: tokenize batches of text inputs.
    text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string),
                   tf.keras.layers.Input(shape=(), dtype=tf.string)]  # This SavedModel accepts up to 2 text inputs.
    tokenize = hub.KerasLayer(preprocessor.tokenize)
    tokenized_inputs = [tokenize(segment) for segment in text_inputs]
    # Step 3: pack input sequences for the Transformer encoder.
    seq_length = 128  # Your choice here.
    bert_pack_inputs = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length))  # Optional argument.
    encoder_inputs = bert_pack_inputs(tokenized_inputs)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4", trainable=True)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]  # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    embedding_model = tf.keras.Model(text_inputs, sequence_output)
    input1 = tf.constant([seq1])
    input2 = tf.constant([seq2])
    out = embedding_model([input1, input2])
    print(out)


if __name__ == '__main__':
    bert_single_seqs_demo("早上好")
    bert_pairs_seqs_demo("早上好", "吃饭了吗")
