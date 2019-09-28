import numpy as np
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization,Dropout,Dense,Activation,Flatten, Conv2D, SeparableConv2D, MaxPooling2D, Activation, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import csv
import sys

def read_test_data():
	test_dat = np.genfromtxt(sys.argv[1],dtype = str, delimiter=',')
	test_dat = test_dat[1:,]
	x_test_string = test_dat[:,1]
	x_test = np.zeros((len(x_test_string),48*48))
	for i in range(len(x_test)):
		x_test[i] = np.fromstring(x_test_string[i], dtype=int, sep=' ')
	x_test = x_test.reshape(-1,48,48,1)
	x_test = (x_test - x_test.mean()) / x_test.std()
	return x_test

def save_res(y_test):
	print(y_test.shape)
	y_test = y_test.tolist()
	idnum = []
	for i in range(len(y_test)):
		idnum.append(i)
	ans = []
	ans.append(idnum)
	ans.append(y_test)
	ttle = np.asarray([["id","label"]])
	ans = np.asarray(ans)
	ans = np.concatenate((ttle,ans.transpose()), axis = 0)
	np.savetxt(sys.argv[2],ans,delimiter=',',fmt="%s")

def build_model1():
	model = Sequential()
	model.add(Conv2D(filters = 16, kernel_size = (3,3),input_shape=(48,48,1),activation = 'relu'))
	model.add(SeparableConv2D(filters = 32, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 48, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(SeparableConv2D(filters = 64, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(filters = 96, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(SeparableConv2D(filters = 128, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(SeparableConv2D(filters = 128, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())

	model.add(Dense(units=7,activation='softmax'))
	model.summary()
	return model

def predict(x_test):
	model1 = build_model1()
	weight_sz = model1.get_weights()
	w_new = []
	weight = np.load('weight2.npz')['arr_0']
	print(weight.shape)
	cur = 0
	for x in weight_sz:
		sz = len(x.ravel())
		ly_w = weight[cur:cur+sz]
		ly_w = ly_w.reshape(x.shape)
		w_new.append(ly_w)
		cur += sz
	assert cur == len(weight)
	model1.set_weights(w_new)
	y_test = model1.predict(x_test)
	y_test = np.argmax(y_test,axis = 1)
	return y_test

x_test = read_test_data()
y_test = predict(x_test)
save_res(y_test)
