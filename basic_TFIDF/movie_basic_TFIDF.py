import tensorflow as tf
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
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

def tokenizer(text):
    okt = Okt()
    words = okt.morphs(text)
    return words

stopword_list = ['나','는','다','이']
max_features = 7
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopword_list, max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(text_data)
dense_tfidf_texts = sparse_tfidf_texts.todense()

X = tf.placeholder(dtype=tf.float32, shape=[None, max_features])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
W = tf.get_variable('weight', shape=[max_features, 1], initializer=tf.random_normal_initializer())
b = tf.get_variable('bias',  shape=[1])
model_output = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(X, W), b))

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=Y))
prediction = tf.cast(tf.reshape(model_output, [-1]) > 0.5, tf.float32)
optim = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(tfidf.get_feature_names())

print("\n> Start train")
for i in range(1000):
    _, l = sess.run([optim, loss], feed_dict={X: dense_tfidf_texts, Y: target_data})
    if (i+1)%100 == 0:
        print("Epoch : {:3d}, Loss : {:.6f}".format(i+1, l))
print("> Done")

print("\n> Predict")
for ix in range(10):
    print(text_data[ix], end="")
    print(sess.run(prediction, feed_dict={X: dense_tfidf_texts[ix]}))

'''
['글쎄', '별로', '싫다', '영화', '정말', '좋다', '진짜']

> Start train
Epoch : 100, Loss : 0.617542
Epoch : 200, Loss : 0.564370
Epoch : 300, Loss : 0.540849
Epoch : 400, Loss : 0.528785
Epoch : 500, Loss : 0.521805
Epoch : 600, Loss : 0.517394
Epoch : 700, Loss : 0.514419
Epoch : 800, Loss : 0.512309
Epoch : 900, Loss : 0.510754
Epoch : 1000, Loss : 0.509571
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
