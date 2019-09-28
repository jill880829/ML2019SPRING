import csv
import numpy as np
import sys

x_train = np.genfromtxt(sys.argv[1],delimiter=',')
x_train = np.delete(x_train,0,0)
x_train_n = x_train / x_train.max(axis=0)

y_train = np.genfromtxt(sys.argv[2],delimiter=',')
y_train = np.delete(y_train,0,0)

b = np.ones((x_train_n.shape[0], 1))
x_train_addb = np.concatenate((x_train_n,b), axis = 1)

w = np.full(x_train_addb.shape[1],0.3)
sigma = 0
epoch = 10000
x_train_t = x_train_addb.transpose()
lumbda = 10

for i in range(epoch):
	z = np.dot(x_train_addb,w)
	ep = np.exp(-z)
	sigmoid = 1/(1+ep)
	loss = sigmoid-y_train
	gra = np.dot(x_train_t,loss)+lumbda*np.sum(w)
	sigma = sigma+gra**2
	lr = 5/(sigma**0.5)
	w = w-lr*gra

ACC=0
for i in range(x_train_addb.shape[0]):
	if sigmoid[i]<0.5:
		hypo = 0
	else:
		hypo = 1
	if hypo == y_train[i]:
		ACC = ACC+1
ACC/=x_train_addb.shape[0]
print(ACC)

np.save('max_X.npy',x_train.max(axis=0))
np.save('model.npy',w)