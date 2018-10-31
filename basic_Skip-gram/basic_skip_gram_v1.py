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

def build_dataset(some_texts, words, vocab, window_size, iter, seed):

    positive_pairs = []
    negative_pairs = []

    for text in some_texts:
        len_text = len(text)
        for ix, word in enumerate(text):
            if ix - window_size >= 0 and ix + window_size <= len_text:
                indices = [i for i in range(ix-window_size,ix+window_size) if not i is ix]
                for i in range(iter):
                    ixx = np.random.choice(indices)
                    context_word = text[ixx]
                    positive_pairs.append([vocab[word],vocab[context_word]])
                    negative_word = np.random.choice([w for w in words if not w in text])
                    negative_pairs.append([vocab[word], vocab[negative_word]])

    negative_pairs = [w for w in negative_pairs if not w in positive_pairs]
    positive_pairs = np.array(positive_pairs)
    negative_pairs = np.array(negative_pairs)

    input_data = positive_pairs[:, 0]
    target_data = positive_pairs[:, 1]
    labels = np.ones(len(positive_pairs))

    input_data = np.append(input_data, negative_pairs[:, 0])
    target_data = np.append(target_data, negative_pairs[:, 1])
    labels = np.append(labels, np.zeros(len(negative_pairs)))

    input_batch = np.zeros([len(input_data), len(vocab)])
    target_batch = np.zeros([len(target_data), len(vocab)])

    for ix, id in enumerate(input_data):
        input_batch[ix, id] = 1.0

    for ix, id in enumerate(target_data):
        target_batch[ix, id] = 1.0

    return input_data, target_data, labels


tf.set_random_seed(0)

text_data = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다",
             "별을 노래하는 마음으로 모든 죽어가는 것을 사랑해야지",
             "그리고 나한테 주어진 길을 걸어가야겠다",
             "오늘 밤에도 별이 바람에 스치운다"]

stopword_list = ['우', '러']

texts = tokenizer(text_data, stopword_list)
vocab, words = build_vocab(texts)
vocab_rev = dict(zip(vocab.values(), vocab.keys()))
vocab_size = len(vocab)

total_epochs = 1000
embed_size = 2
window_size = 1
iter = 30
batch_size = 256
seed = 0

input_data, target_data, labels = build_dataset(texts, words, vocab, window_size, iter, seed)

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, vocab_size], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size], name='Y')
T = tf.placeholder(dtype=tf.float32, shape=[batch_size, vocab_size], name='T')

W1 = tf.get_variable('W1', [vocab_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
b1 = tf.get_variable('b1', [embed_size], initializer=tf.zeros_initializer())

o1 = tf.nn.l2_normalize(tf.nn.bias_add(tf.matmul(X, W1), b1), axis=1)
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

total_step = int(len(input_data)/batch_size)
for epoch in range(total_epochs):
    np.random.seed(epoch)
    mask = np.random.permutation(len(input_data))
    loss_per_epoch = 0
    for step in range(total_step):
        s = step*batch_size
        t = (step+1)*batch_size
        _, c = sess.run([train, cost], feed_dict={X: np.concatenate([np.expand_dims(one_hot_encoding(w, vocab_size), axis=0) for w in input_data[mask[s:t]]], axis=0),
                                                 Y: labels[mask[s:t]],
                                                 T: np.concatenate([np.expand_dims(one_hot_encoding(w, vocab_size), axis=0) for w in target_data[mask[s:t]]], axis=0)})
        loss_per_epoch += c/total_step

    if epoch % 100 == 0:
        print("Epoch : {:4d}, Cost : {:.6f}".format(epoch, loss_per_epoch))

for ix, sentence in enumerate(texts):
    for word in sentence:
        id = vocab[word]
        word_one_hot = np.expand_dims(one_hot_encoding(id, vocab_size), axis=0)
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
Epoch :    0, Cost : 0.755819
Epoch :  100, Cost : 0.434483
Epoch :  200, Cost : 0.438684
Epoch :  300, Cost : 0.437103
Epoch :  400, Cost : 0.433184
Epoch :  500, Cost : 0.429189
Epoch :  600, Cost : 0.433840
Epoch :  700, Cost : 0.431945
Epoch :  800, Cost : 0.435796
Epoch :  900, Cost : 0.429726
[['날', '하늘', '우러러', '점', '부끄럼', '잎새', '바람', '나'], 
 ['별', '노래', '마음', '모든', '사랑'], 
 ['나', '길', '가야'], 
 ['오늘', '밤', '별', '바람', '스치운']]
'''