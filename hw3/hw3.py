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

def predict(x_test):
	model1 = load_model('keras_model1.h5')
	model2 = load_model('keras_model2.h5')
	model3 = load_model('keras_model3.h5')
	y_test1 = model1.predict(x_test)
	y_test2 = model2.predict(x_test)
	y_test3 = model3.predict(x_test)
	y_test = y_test1+y_test2+y_test3
	y_test = np.argmax(y_test,axis = 1)
	return y_test

x_test = read_test_data()
y_test = predict(x_test)
save_res(y_test)
