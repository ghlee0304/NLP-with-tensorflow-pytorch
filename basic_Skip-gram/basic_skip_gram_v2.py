from konlpy.tag import Kkma
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus']=False

tf.set_random_seed(0)

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
    tmp = np.expand_dims(tmp, axis=0)
    return tmp


def build_dataset(some_texts, words, vocab, window_size, pk, nk, seed):

    central_sample = []
    positive_sample = []
    negative_sample = []
    _seed = seed

    for text in some_texts:
        len_text = len(text)
        for ix, word in enumerate(text):
            if ix - window_size >= 0 and ix + window_size <= len_text:
                indices = [i for i in range(ix-window_size,ix+window_size) if not i is ix]
                context_words = [text[i] for i in indices]
                for _ in range(pk):
                    np.random.seed(_seed)
                    _seed += 1
                    context_word = np.random.choice(context_words)
                    central_sample.append([vocab[word]])
                    positive_sample.append([vocab[context_word]])
                    negative_words = [w for w in words if not w in context_words]
                    np.random.seed(_seed)
                    _seed += 1
                    negative_words = np.random.choice(negative_words, nk, replace=True)
                    negative_sample.append([vocab[word] for word in negative_words])

    central_sample = np.array(central_sample)
    positive_sample = np.array(positive_sample)
    negative_sample = np.array(negative_sample)

    return central_sample, positive_sample, negative_sample


text_data = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다",
             "별을 노래하는 마음으로 모든 죽어가는 것을 사랑해야지",
             "그리고 나한테 주어진 길을 걸어가야겠다",
             "오늘 밤에도 별이 바람에 스치운다"]

stopword_list = ['우', '러']

texts = tokenizer(text_data, stopword_list)
vocab, words = build_vocab(texts)
vocab_rev = dict(zip(vocab.values(), vocab.keys()))
vocab_size = len(vocab)

total_epochs = 100
embed_size = 2
window_size = 1
pk = 50
nk = 5
seed = 0

central_sample, positive_sample, negative_sample = build_dataset(texts, words, vocab, window_size, pk, nk, seed)

central_input = tf.placeholder(dtype=tf.float32, shape=[1, vocab_size], name='central_input')
pos_input = tf.placeholder(dtype=tf.float32, shape=[1, vocab_size], name='pos_input')
neg_input = tf.placeholder(dtype=tf.float32, shape=[None, vocab_size], name='neg_input')

W1 = tf.get_variable('W1', [vocab_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
b1 = tf.get_variable('b1', [embed_size], initializer=tf.zeros_initializer())

central_embed = tf.nn.bias_add(tf.matmul(central_input, W1), b1)
pos_embed = tf.nn.bias_add(tf.matmul(pos_input, W1), b1)
neg_embed = tf.nn.bias_add(tf.matmul(neg_input, W1), b1)

pos_objective = tf.log(tf.nn.sigmoid(tf.reduce_sum(central_embed*pos_embed)))
neg_objective = tf.reduce_mean(tf.log(tf.nn.sigmoid(-tf.reduce_sum(central_embed*neg_embed, axis=1))))

cost = -(pos_objective+neg_objective)

X_one = tf.placeholder(dtype=tf.float32, shape=[1, vocab_size], name='X_one')
prediction = tf.nn.bias_add(tf.matmul(X_one, W1), b1)

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

print("> The number of data samples : {}".format(len(central_sample)))
print("> Train Start!!!")

for epoch in range(total_epochs):
    for i in range(len(central_sample)):
        _, c = sess.run([train, cost], feed_dict={central_input: one_hot_encoding(central_sample[i], vocab_size),
                                                  pos_input: one_hot_encoding(positive_sample[i], vocab_size),
                                                  neg_input: np.concatenate([one_hot_encoding(w, vocab_size) for w in negative_sample[i]], axis=0)})
    if epoch % 10 == 0:
        print("Epoch : {:4d}, Cost : {:.6f}".format(epoch, c))

for ix, sentence in enumerate(texts):
    for word in sentence:
        id = vocab[word]
        word_one_hot = one_hot_encoding(id, vocab_size)
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
> The number of data samples : 850
> Train Start!!!
Epoch :    0, Cost : 1.311364
Epoch :   10, Cost : 1.132766
Epoch :   20, Cost : 1.050370
Epoch :   30, Cost : 1.020001
Epoch :   40, Cost : 1.017627
Epoch :   50, Cost : 1.016848
Epoch :   60, Cost : 1.016014
Epoch :   70, Cost : 1.015304
Epoch :   80, Cost : 1.014766
Epoch :   90, Cost : 1.014364
'''
