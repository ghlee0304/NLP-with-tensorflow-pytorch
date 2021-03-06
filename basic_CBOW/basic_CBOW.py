from konlpy.tag import Kkma
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib
#import matplotlib.font_manager as fm

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus']=False
#font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
#print(font_list)

def tokenizer(text_data, stopword_list):
    kkma = Kkma()
    texts = []
    for t in text_data:
        tmp = [n for n in kkma.nouns(t) if n not in stopword_list]
        texts.append(tmp)
    return texts

def build_vocab(some_texts):
    words = []
    for t in some_texts:
        words.extend(t)
    words_set = set(words)
    vocab_dict = {}
    vocab_dict['RARE'] = 0
    for ix, word in enumerate(words_set):
        vocab_dict[word] = ix+1
    return vocab_dict, words

def one_hot_encoding(word_id, vocab_size):
    tmp = np.zeros([vocab_size])
    tmp[word_id] = 1.0
    return tmp

def generate_batch(some_texts, words, vocab, n_pos, n_neg, window_size, iter, seed=0):

    positive_pairs = []
    negative_pairs = []

    for text in some_texts:
        len_text = len(text)
        for ix, word in enumerate(text):
            context_words = [text[k] for k in range(ix - window_size, ix + window_size + 1) if not k < 0 and not k >= len_text and not k == ix]
            for i in range(iter):
                negative_words = np.random.choice([w for w in words if not w in context_words] , window_size*2, replace=False)
                if len(context_words)>window_size*2:
                    positive_words = np.random.choice([w for w in words if w in context_words] , window_size*2, replace=False)
                else:
                    positive_words = np.random.choice([w for w in words if w in context_words], window_size * 2, replace=True)

                positive_pairs.append([[vocab[positive_word] for positive_word in positive_words], vocab[word]])
                negative_pairs.append([[vocab[negative_word] for negative_word in negative_words], vocab[word]])

    positive_pairs = np.array(positive_pairs)
    negative_pairs = np.array(negative_pairs)

    pos_mask = np.random.choice(len(positive_pairs), n_pos, replace=True)
    neg_mask = np.random.choice(len(negative_pairs), n_neg, replace=True)

    pos_data = positive_pairs[pos_mask]
    neg_data = negative_pairs[neg_mask]

    input_data = pos_data[:, 0]
    target_data = pos_data[:, 1]
    labels = np.ones(len(pos_data))

    input_data = np.append(input_data, neg_data[:, 0])
    target_data = np.append(target_data, neg_data[:, 1])
    labels = np.append(labels, np.zeros(len(neg_data)))

    input_batch = np.zeros([len(input_data), window_size*2, len(vocab)])
    target_batch = np.zeros([len(target_data), len(vocab)])

    for ix, ids in enumerate(input_data):
        for iix, id in enumerate(ids):
            input_batch[ix, iix, id] = 1.0

    for ix, id in enumerate(target_data):
        target_batch[ix, id] = 1.0

    return input_batch, target_batch, labels


tf.set_random_seed(0)

text_data = ["하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 부끄러워 했다",
             "별을 노래하는 마음으로 모든 죽어가는 것을 사랑해야지"]

stopword_list = ['우', '러', '점']

texts = tokenizer(text_data, stopword_list)
vocab, words = build_vocab(texts)
vocab_rev = dict(zip(vocab.values(), vocab.keys()))
vocab_size = len(vocab)

n_pos = 10
n_neg = 5
total_epochs = 10000
embed_size = 2
window_size = 1
iter = 3
batch_size = n_pos+n_neg
early_stopping = 0.01

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, window_size*2, vocab_size], name='X') #문맥 내의 여러 단어
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size], name='Y') #0 또는 1
T = tf.placeholder(dtype=tf.float32, shape=[batch_size, vocab_size], name='T') #하나의 단어

W1 = tf.get_variable('W1', [vocab_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
b1 = tf.get_variable('b1', [embed_size], initializer=tf.zeros_initializer())

o1 = []
for b in range(batch_size):
    tmp = tf.zeros([1, embed_size])
    for w in range(2*window_size):
        tmp += tf.nn.bias_add(tf.matmul(tf.reshape(X[b, w, :], [1, vocab_size]), W1), b1) / (window_size*2)
    o1.append(tmp)

o1 = tf.nn.l2_normalize(tf.concat(o1, axis=0), axis=1)
o2 = tf.nn.l2_normalize(tf.nn.bias_add(tf.matmul(T, W1), b1), axis=1)
logits = tf.reduce_sum(o1*o2, axis=1)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))

X_one = tf.placeholder(dtype=tf.float32, shape=[1, vocab_size], name='X_one')
prediction = tf.nn.bias_add(tf.matmul(X_one, W1), b1)

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
for epoch in range(total_epochs):
    input_batch, target_batch, labels = generate_batch(texts, words, vocab, n_pos, n_neg, window_size, iter)
    _, c= sess.run([train, cost], feed_dict={X: input_batch, Y: labels, T: target_batch})
    if c < early_stopping:
        break
    if epoch % 1000 == 0:
        print(epoch, c)

cnt = 0
for sentence in texts:
    for word in sentence:
        id = vocab[word]
        word_one_hot = np.expand_dims(one_hot_encoding(id, vocab_size), axis=0)
        word_vec = np.reshape(sess.run(prediction, feed_dict={X_one: word_one_hot}),-1)
        if cnt == 0:
            plt.scatter(word_vec[0], word_vec[1], c='r')
            plt.annotate(word, (word_vec[0], word_vec[1]))
        else:
            plt.scatter(word_vec[0], word_vec[1], c='b')
            plt.annotate(word, (word_vec[0], word_vec[1]))
    cnt += 1

plt.grid()
plt.show()