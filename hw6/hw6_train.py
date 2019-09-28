from sys import argv
import csv
import numpy as np
import jieba
jieba.load_userdict(argv[4])
from gensim.models.word2vec import Word2Vec
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Conv2D, Flatten, Reshape, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras import backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import hw6_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
pad = np.random.uniform(0,1,128)
np.save('pad.npy',pad)
unknown = np.random.uniform(0,1,128)
np.save('unknown.npy',unknown)

def process_jieba(data_df):
    data_x = []
    for s in data_df:
        ss = jieba.lcut(str(s))
        # ss = list(map(emoji.demojize, ss))
        data_x.append(ss)
    return data_x


if __name__ == "__main__":
    train_x = np.genfromtxt(argv[1],dtype=str,delimiter=',',encoding='utf-8')
    train_x = np.delete(train_x,0,0)
    train_x = np.delete(train_x,0,1)
    train_x = train_x[:119018]
    train_y = np.genfromtxt(argv[2],dtype=int,delimiter=',')
    train_y = np.delete(train_y,0,0)
    train_y = np.delete(train_y,0,1)
    train_y = train_y[:119018]
    np.random.seed(0)
    p = np.random.permutation(train_x.shape[0])
    train_x = train_x[p]
    train_y = train_y[p]
    test_x = np.genfromtxt(argv[3],dtype=str,delimiter=',',encoding='utf-8')
    test_x = np.delete(test_x,0,0)
    test_x = np.delete(test_x,0,1)
    train_x = process_jieba(train_x)
    test_x = process_jieba(test_x)
    corpus = train_x + test_x
    w2v_model = Word2Vec(corpus, size=128,sg=1, window = 5, min_count=3,compute_loss = True, iter = 27, batch_words = 64)
    w2v_model.save('word2vec.model')

    # process sentence
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
    for i in range(len(train_x)):
        s = []
        for j,w in enumerate(train_x[i]):
            if (j==0 or train_x[i][j]!=train_x[i][j-1]) and w != ' ':
                s.append(train_x[i][j])
        s = s[:max_sentence_len]
        s = list(map(lambda w: word2index[w] if w in w2v_model.wv else unknown_idx, s))
        s += [pad_idx] * (max_sentence_len - len(s))
        train_x[i] = s
    train_x = np.asarray(train_x)
    model1 = hw6_model.build_model1(train_x,train_y,index2vec)
    model5 = hw6_model.build_model5(train_x,train_y,index2vec)
    model6 = hw6_model.build_model6(train_x,train_y,index2vec)