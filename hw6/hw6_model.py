# from keras.models import Model, Sequential
# from keras.layers.normalization import BatchNormalization
# from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Conv2D, Flatten, Reshape, MaxPooling1D
# from keras.layers.embeddings import Embedding
# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import *
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import layers
# from keras import backend as K
# from keras.layers import Layer
# from keras import initializers, regularizers, constraints
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Conv2D, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from keras import layers
import numpy as np
def build_model1(train_x, train_y, index2vec):
    model = Sequential()
    model.add(Embedding(len(index2vec), len(index2vec[0]), weights=[np.array(index2vec)], trainable=False))
    model.add(Bidirectional(LSTM(256, activation="tanh", dropout=0.6,
                                 return_sequences=False, kernel_initializer='Orthogonal')))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint('best1.h5', monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max')
    es = EarlyStopping(monitor = 'val_acc',verbose = 2,patience = 80, mode = 'max')
    model.fit(train_x, train_y, batch_size=512, epochs=100, validation_split=0.1, callbacks=[cp,es])
    return model


# def dot_product(x, kernel):
#     if K.backend() == 'tensorflow':
#         return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
#     else:
#         return K.dot(x, kernel)
 
# class AttentionWithContext(Layer):
#     def __init__(self,
#                  W_regularizer=None, u_regularizer=None, b_regularizer=None,
#                  W_constraint=None, u_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
 
#         self.supports_masking = True
#         self.init = initializers.get('glorot_uniform')
 
#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.u_regularizer = regularizers.get(u_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
 
#         self.W_constraint = constraints.get(W_constraint)
#         self.u_constraint = constraints.get(u_constraint)
#         self.b_constraint = constraints.get(b_constraint)
 
#         self.bias = bias
#         super(AttentionWithContext, self).__init__(**kwargs)
 
#     def build(self, input_shape):
#         assert len(input_shape) == 3
 
#         self.W = self.add_weight((input_shape[-1], input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         if self.bias:
#             self.b = self.add_weight((input_shape[-1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)
 
#         self.u = self.add_weight((input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_u'.format(self.name),
#                                  regularizer=self.u_regularizer,
#                                  constraint=self.u_constraint)
 
#         super(AttentionWithContext, self).build(input_shape)
 
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None
 
#     def call(self, x, mask=None):
#         uit = dot_product(x, self.W)
 
#         if self.bias:
#             uit += self.b
 
#         uit = K.tanh(uit)
#         ait = dot_product(uit, self.u)
 
#         a = K.exp(ait)
 
#         # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())
 
#         # in some cases especially in the early stages of training the sum may be almost zero
#         # and this results in NaN's. A workaround is to add a very small positive number  to the sum.
#         # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
#         a = K.expand_dims(a)
#         weighted_input = x * a
#         return K.sum(weighted_input, axis=1)
 
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[-1]

# def build_model2(input_sp):
#     model = Sequential()
#     model.add(keras.layers.core.Masking(mask_value=0., input_shape=input_sp))
#     model.add(Bidirectional(GRU(units=128,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
#                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
#                   bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
#                   bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
#                   return_state=False, go_backwards=False, stateful=False, unroll=False),merge_mode='concat'))   #input_shape=(max_lenth, max_features),
#     model.add(Bidirectional(GRU(units=128,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
#                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
#                   bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
#                   bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
#                   return_state=False, go_backwards=False, stateful=False, unroll=False),merge_mode='concat'))   #input_shape=(max_lenth, max_features),
#     model.add(Dropout(0.5))
#     model.add(AttentionWithContext())
#     model.add(Dense(1))
#     model.add(BatchNormalization())
#     model.add(keras.layers.core.Activation('sigmoid'))
     
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=[metrics.binary_crossentropy])
#     model.summary()
#     cp = ModelCheckpoint('best2.h5', monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max')
#     es = EarlyStopping(monitor = 'val_acc',verbose = 2,patience = 5, mode = 'max')
#     return model,cp,es

def build_model3(train_x, train_y, index2vec):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    model = Sequential()
    model.add(GRU(128,activation="tanh",dropout=0.3,return_sequences = True,
            kernel_initializer='Orthogonal', input_shape=input_sp))
    model.add(GRU(128,activation="tanh",dropout=0.3,return_sequences = True,
            kernel_initializer='Orthogonal'))
    model.add(Reshape([30, 128, 1]))
    model.add(Conv2D(filters=8, kernel_size=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint('best3.h5', monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max')
    es = EarlyStopping(monitor = 'val_acc',verbose = 2,patience = 80, mode = 'max')
    return model,cp,es

# class NonMasking(Layer):
#     def __init__(self, **kwargs):
#         self.supports_masking = True
#         super(NonMasking, self).__init__(**kwargs)
 
#     def build(self, input_shape):
#         input_shape = input_shape
 
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None
 
#     def call(self, x, mask=None):
#         return x
 
#     def get_output_shape_for(self, input_shape):
#         return input_shape
# def build_model4(input_sp):
#     model_left = Sequential()
#     model_left.add(keras.layers.core.Masking(mask_value=0., input_shape=input_sp))  
#     model_left.add(GRU(units=128,activation='relu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
#                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
#                   bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
#                   bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
#                   return_state=False, go_backwards=False, stateful=False, unroll=False))
#     model_left.add(GRU(units=128,activation='relu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
#                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
#                   bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
#                   bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
#                   return_state=False, go_backwards=False, stateful=False, unroll=False))
#     model_left.add(NonMasking())  
#     model_left.add(Flatten())
     
#     ## FCN
#     model_right = Sequential()
#     model_right.add(Conv1D(128, 3, padding='same', input_shape=input_sp))
#     model_right.add(BatchNormalization())
#     model_right.add(Activation('relu'))
#     model_right.add(Conv1D(256, 3))
#     model_right.add(BatchNormalization())
#     model_right.add(Activation('relu'))
#     model_right.add(Conv1D(128, 3))
#     model_right.add(BatchNormalization())
#     model_right.add(Activation('relu'))
#     model_right.add(GlobalAveragePooling1D())
#     model_right.add(Reshape((1,1,-1)))
#     model_right.add(Flatten())
     
#     model = Sequential()
#     model.add(Merge([model_left,model_right], mode='concat'))
     
#     model.add(Dense(128))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     model.add(BatchNormalization())
#     model.add(Activation('sigmoid'))
#     model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
#     model.summary()
#     cp = ModelCheckpoint('best4.h5', monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max')
#     es = EarlyStopping(monitor = 'val_acc',verbose = 2,patience = 10, mode = 'max')
#     return model,cp,es

def build_model5(train_x, train_y, index2vec):
    model = Sequential()
    model.add(Embedding(len(index2vec), len(index2vec[0]), weights=[np.array(index2vec)], trainable=False))
    model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True,
            kernel_initializer='Orthogonal'))
    model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False,
            kernel_initializer='Orthogonal')))
    # model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False,
    #       kernel_initializer='Orthogonal'))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint('best5.h5', monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max')
    es = EarlyStopping(monitor = 'val_acc',verbose = 2,patience = 20, mode = 'max')
    model.fit(train_x, train_y, batch_size=512, epochs=100, validation_split=0.1, callbacks=[cp,es])
    return model

def build_model6(train_x, train_y, index2vec):
    model = Sequential()
    model.add(Embedding(len(index2vec), len(index2vec[0]), weights=[np.array(index2vec)], trainable=False))
    model.add(Bidirectional(LSTM(256, activation="tanh", dropout=0.6,
                                 return_sequences=True, kernel_initializer='Orthogonal')))
    model.add(Bidirectional(LSTM(128, activation="tanh", dropout=0.6,
                                 return_sequences=False, kernel_initializer='Orthogonal')))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint('best6.h5', monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max')
    es = EarlyStopping(monitor = 'val_acc',verbose = 2,patience = 20, mode = 'max')
    model.fit(train_x, train_y, batch_size=512, epochs=100, validation_split=0.1, callbacks=[cp,es])
    return model
