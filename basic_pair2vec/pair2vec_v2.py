'''
pair2vec_v2
2018-10-30
Objective: Bivariate Negative Sampling
minibatch SGD
'''

from konlpy.tag import Kkma
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib
import random

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus']=False


def word_tokenizer(text_data, stopword_list):
    kkma = Kkma()
    texts = []
    for t in text_data:
        tmp = [n for n in kkma.nouns(t) if n not in stopword_list]
        texts.append(tmp)
    return texts


def build_vocab(some_texts):
    keys = []
    for t in some_texts:
        keys.extend(t)
    key_set = set(keys)
    vocab_dict = {}
    vocab_dict['RARE'] = 0
    for ix, key in enumerate(key_set):
        vocab_dict[key] = ix+1
    return vocab_dict, keys


def one_hot_encoding(word_id, vocab_size):
    tmp = np.zeros([vocab_size])
    tmp[word_id] = 1.0
    return tmp


def build_word_dataset(texts, pos_iter, neg_iter):
    pos_dataset = []
    neg_dataset= []
    for c, t in enumerate(texts):
        for ix in range(len(t)):
            for _ in range(pos_iter):
                cnd = True
                while cnd:
                    y = random.choice(t)
                    if not y == t[ix]:
                        cnd = False
                pos_dataset.append([word_vocab[t[ix]], word_vocab[y], c, 1.0])

        neg_c = [i for i in range(len(texts)) if not i==c]
        for ix in range(len(t)):
            for _ in range(neg_iter):
                cnd = True
                while cnd:
                    y = random.choice(t)
                    if not y == t[ix]:
                        cnd = False
                nc = random.choice(neg_c)
                neg_dataset.append([word_vocab[t[ix]], word_vocab[y], nc, 0.0])

        dataset = np.append(pos_dataset, neg_dataset, axis=0)
        mask = np.random.permutation(len(dataset))
        dataset = dataset[mask]
    return dataset


def build_sentence_dataset(texts, word_vocab):
    len_max = max([len(texts[i]) for i in range(len(texts))])
    len_vocab = len(word_vocab)
    nn = 0
    for ix, t in enumerate(texts):
        n = 0
        for w in t:
            id = word_vocab[w]
            one_hot = np.zeros([1, 1, len_vocab])
            one_hot[0, 0, id] = 1.0
            if n == 0:
                tmp_data = one_hot
                n += 1
            else:
                tmp_data = np.append(tmp_data, one_hot, axis=1)
        while np.size(tmp_data, 1) < len_max:
            one_hot = np.zeros([1, 1, len_vocab])
            one_hot[0, 0, 0] = 1.0
            tmp_data = np.append(tmp_data, one_hot, axis=1)
        if nn == 0:
            dataset= tmp_data
            nn+=1
        else:
            dataset = np.append(dataset, tmp_data, axis=0)

    return dataset, len_max


tf.set_random_seed(0)
text_data = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다",
             "별을 노래하는 마음으로 모든 죽어가는 것을 사랑해야지",
             "그리고 나한테 주어진 길을 걸어가야겠다",
             "오늘 밤에도 별이 바람에 스치운다"]

stopword_list = ['우', '러']

texts = word_tokenizer(text_data, stopword_list)
word_vocab, words = build_vocab(texts)
word_vocab_rev = dict(zip(word_vocab.values(), word_vocab.keys()))
word_vocab_size = len(word_vocab)

total_epochs = 1000
embed_size = 2
batch_size = 16
pos_iter = 100
neg_iter = 50

word_dataset = build_word_dataset(texts, pos_iter, neg_iter)
sentence_dataset, len_max = build_sentence_dataset(texts, word_vocab)

X = tf.placeholder(dtype=tf.float32, shape=[16, word_vocab_size], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[16, word_vocab_size], name='Y')
T = tf.placeholder(dtype=tf.float32, shape=[None], name='T')

W1 = tf.get_variable('W1', [word_vocab_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
b1 = tf.get_variable('b1', [embed_size], initializer=tf.zeros_initializer())

o1 = tf.nn.l2_normalize(tf.nn.bias_add(tf.matmul(X, W1), b1), axis=1, name='X_embedd')
o2 = tf.nn.l2_normalize(tf.nn.bias_add(tf.matmul(Y, W1), b1), axis=1, name='Y_embedd')
o3 = tf.multiply(o1, o2, name='XY_product')

word_input = tf.concat([o1, o2, o3], axis=1)

W2 = tf.get_variable('W2', [embed_size*3, embed_size*2], initializer=tf.truncated_normal_initializer(stddev=0.01))
b2 = tf.get_variable('b2', [embed_size*2], initializer=tf.zeros_initializer())
h2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(word_input, W2), b2, name='h2'))

W3 = tf.get_variable('W3', [embed_size*2, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
b3 = tf.get_variable('b3', [embed_size], initializer=tf.zeros_initializer())
h3 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h2, W3), b3, name='h3'))

C = tf.placeholder(dtype=tf.float32, shape=[None, len_max, word_vocab_size])

cell = tf.contrib.rnn.LSTMCell(num_units=embed_size, initializer=tf.glorot_uniform_initializer())
outputs, states = tf.nn.dynamic_rnn(cell, C, dtype=tf.float32)

k = tf.get_variable('k', [1, 1, embed_size], initializer=tf.ones_initializer(), trainable=False)
W = tf.get_variable('W', [embed_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.01))

Cc = []
for ix in range(batch_size):
    h = outputs[ix, :, :]
    w = tf.reshape(tf.nn.softmax(tf.reduce_sum(h*k, axis=2)), [len_max, 1])
    tmp_c = tf.matmul(h, W)
    tmp_c = tf.reduce_sum(w*tmp_c, axis=0, keepdims=True)
    Cc.append(tmp_c)

Cc = tf.concat(Cc, axis=0)
logits = tf.reduce_sum(h3*Cc, axis=1)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=T, logits=logits))

X_one = tf.placeholder(dtype=tf.float32, shape=[1, word_vocab_size], name='X_one')
prediction = tf.nn.bias_add(tf.matmul(X_one, W1), b1)

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

print("=== The number of data samples : {:d} ===".format(len(word_dataset)))
print("> Train Start")
total_steps = int(len(word_dataset)/batch_size)
for epoch in range(total_epochs):
    losses = 0
    for step in range(total_steps):
        st = 0
        for ix in range(step*batch_size, (step+1)*batch_size):
            xx = np.expand_dims(one_hot_encoding(int(word_dataset[ix][0]), word_vocab_size), axis=0)
            yy = np.expand_dims(one_hot_encoding(int(word_dataset[ix][1]), word_vocab_size), axis=0)
            cc = np.expand_dims(sentence_dataset[int(word_dataset[ix][2])], axis=0)
            tt = np.array([word_dataset[ix][3]])
            if st == 0:
                x_batch = xx
                y_batch = yy
                c_batch = cc
                t_batch = tt
                st+=1
            else:
                x_batch = np.append(x_batch, xx, axis=0)
                y_batch = np.append(y_batch, yy, axis=0)
                c_batch = np.append(c_batch, cc, axis=0)
                t_batch = np.append(t_batch, tt, axis=0)

        _, c = sess.run([train, cost], feed_dict={X: x_batch, Y: y_batch, C: c_batch, T: t_batch})
        losses += c/total_steps

    if (epoch+1) % 100 == 0:
        print("Epoch : {:4d}, Cost : {:.6f}".format(epoch+1, losses))

for ix, sentence in enumerate(texts):
    for word in sentence:
        id = word_vocab[word]
        word_one_hot = np.expand_dims(one_hot_encoding(id, word_vocab_size), axis=0)
        word_vec = np.reshape(sess.run(prediction, feed_dict={X_one: word_one_hot}),-1)
        if ix == 0: color = 'b'
        elif ix == 1: color = 'r'
        elif ix == 2: color = 'k'
        elif ix == 3: color = 'g'

        plt.scatter(word_vec[0], word_vec[1], c=color)
        plt.annotate(word, (word_vec[0], word_vec[1]))

print(texts)

plt.grid()
plt.show()

'''
=== The number of data samples : 3150 ===
> Train Start
Epoch :  100, Cost : 0.344267
Epoch :  200, Cost : 0.343320
Epoch :  300, Cost : 0.342278
Epoch :  400, Cost : 0.286317
Epoch :  500, Cost : 0.115812
Epoch :  600, Cost : 0.101747
Epoch :  700, Cost : 0.089372
Epoch :  800, Cost : 0.082475
Epoch :  900, Cost : 0.078406
Epoch : 1000, Cost : 0.075668
[['날', '하늘', '우러러', '점', '부끄럼', '잎새', '바람', '나'], ['별', '노래', '마음', '모든', '사랑'], ['나', '길', '가야'], ['오늘', '밤', '별', '바람', '스치운']]
'''
