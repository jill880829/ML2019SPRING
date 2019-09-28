import numpy as np
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization,Dropout,Dense,Activation,Flatten, Conv2D, MaxPooling2D, Activation, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.models import load_model
import csv
from sys import argv
import jieba
jieba.load_userdict(argv[3])
from gensim.models.word2vec import Word2Vec
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
def process_jieba(data_df):
    data_x = []
    for s in data_df:
        ss = jieba.lcut(str(s))
        # ss = list(map(emoji.demojize, ss))
        data_x.append(ss)
    return data_x

pad = np.load('pad.npy')
unknown = np.load('unknown.npy')
w2v_model = Word2Vec.load("word2vec.model")
model1 = load_model('best1.h5')
model5 = load_model('best5.h5')
model6 = load_model('best6.h5')
test_x = np.genfromtxt(argv[1],dtype=str,delimiter=',',encoding='utf-8')
test_x = np.delete(test_x,0,0)
test_x = np.delete(test_x,0,1)
test_x = process_jieba(test_x)
word2index = dict()
index2word = w2v_model.wv.index2word
index2vec = []
for i, w in enumerate(index2word):
    word2index[w] = i
    index2vec.append(w2v_model.wv[w])
pad_idx = len(index2vec)
index2vec.append(pad)
unknown_idx = len(index2vec)
index2vec.append(unknown)
max_sentence_len = 80
for i in range(len(test_x)):
    s = []
    for j,w in enumerate(test_x[i]):
        if (j==0 or test_x[i][j]!=test_x[i][j-1]) and w != ' ':
            s.append(test_x[i][j])
    s = s[:max_sentence_len]
    s = list(map(lambda w: word2index[w] if w in w2v_model.wv else unknown_idx, s))
    s += [pad_idx] * (max_sentence_len - len(s))
    test_x[i] = s
test_x = np.asarray(test_x)
test_y1 = model1.predict(test_x)
test_y5 = model5.predict(test_x)
test_y6 = model6.predict(test_x)
test_y = test_y1+test_y5+test_y6 > 0.5*3
test_y = np.squeeze(test_y)
idnum = []
for i in range(test_x.shape[0]):
    idnum.append(i)
ans = []
ans.append(idnum)
ans.append(test_y)
ttle = np.asarray([["id","label"]])
ans = np.asarray(ans)
ans = np.concatenate((ttle,ans.transpose()), axis = 0)
np.savetxt(argv[2],ans,delimiter=',',fmt="%s")
