import tensorflow as tf
from tensorflow.contrib import learn
tf.set_random_seed(0)

text_data = ["이 영화는 별로다",
             "이 영화는 좋다",
             "이 영화는 글쎄",
             "진짜 이 영화는 좋다",
             "싫다",
             "나는 좋다",
             "진짜 별로다",
             "나는 영화는 좋다",
             "정말 싫다",
             "나는 진짜 좋다"]

target_data = [[0.], [1.], [0.], [1.], [0.], [1.], [0.], [1.], [0.], [1.]]
sentence_size = 4
min_word_freq = 1

vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
vocab_processor.fit_transform(text_data)
embedding_size = len(vocab_processor.vocabulary_)

identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

X = tf.placeholder(dtype=tf.int32, shape=[sentence_size])

x_embed = tf.nn.embedding_lookup(identity_mat, X)
x_col_sums = tf.reduce_sum(x_embed, 0)
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)

XX = tf.placeholder(dtype=tf.float32, shape=[None, embedding_size])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
W = tf.get_variable('weight', shape=[embedding_size, 1], initializer=tf.random_normal_initializer())
b = tf.get_variable('bias',  shape=[1])
model_output = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(XX, W), b))

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=Y))
prediction = tf.cast(tf.reshape(model_output, [-1])>0.5, tf.float32)
optim = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

vocab_dict = vocab_processor.vocabulary_._mapping
print(vocab_dict.items())

import numpy as np

print("\n> Start conversion")
for ix, t in enumerate(vocab_processor.fit_transform(text_data)):
    a = sess.run(x_col_sums_2D, feed_dict={X: t})
    if ix == 0:
        transformed_data = a
    else:
        transformed_data = np.append(transformed_data, a, axis=0)
print("> Done")

print("\n> Start train")
for i in range(1000):
    _, l = sess.run([optim, loss], feed_dict={XX: transformed_data, Y: target_data})
    if (i+1)%100 == 0:
        print("Epoch : {:3d}, Loss : {:.6f}".format(i+1, l))
print("> Done")

print("\n> Predict")
for ix in range(10):
    print(text_data[ix], end="")
    print(sess.run(prediction, feed_dict={XX: [transformed_data[ix]]}))

'''
dict_items([('영화는', 1), ('좋다', 2), ('싫다', 7), ('나는', 4), ('<UNK>', 0), ('진짜', 5), ('별로다', 6), ('이', 3)])

> Start conversion
> Done

> Start train
Epoch : 100, Loss : 0.564246
Epoch : 200, Loss : 0.532625
Epoch : 300, Loss : 0.521445
Epoch : 400, Loss : 0.515860
Epoch : 500, Loss : 0.512586
Epoch : 600, Loss : 0.510477
Epoch : 700, Loss : 0.509029
Epoch : 800, Loss : 0.507985
Epoch : 900, Loss : 0.507205
Epoch : 1000, Loss : 0.506605
> Done

> Predict
이 영화는 별로다[0.]
이 영화는 좋다[1.]
이 영화는 글쎄[0.]
진짜 이 영화는 좋다[1.]
싫다[0.]
나는 좋다[1.]
진짜 별로다[0.]
나는 영화는 좋다[1.]
정말 싫다[0.]
나는 진짜 좋다[1.]
'''
