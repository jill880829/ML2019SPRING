import csv 
import numpy as np
import sys

fl = np.genfromtxt(sys.argv[1],dtype=str,delimiter=',',encoding='big5')
fl = np.delete(fl,2,1)
fl = np.delete(fl,1,1)
fl = np.delete(fl,0,1)
fl = np.delete(fl,0,0)
fl_shape = fl.shape
dat = np.zeros(fl_shape)

for x in range(fl_shape[0]):
	for y in range(fl_shape[1]):
		if fl[x,y] == 'NR':
			dat[x,y] = float(0)
		else:
			dat[x,y] = float(fl[x,y])

train_x = []
train_y = []
for i in range(12):
	tem_dat = dat[i*20*18:(i*20+1)*18]
	for j in range(1,20):
		tem_dat = np.concatenate((tem_dat,dat[(i*20+j)*18:(i*20+j+1)*18]), axis = 1)
	for j in range(1,480):
		for k in range(18):
			tem_dat[k,j] = tem_dat[k,j-1] if tem_dat[k,j]<0 else tem_dat[k,j]
	for j in range(471):
		tem_dat_T = tem_dat.transpose()
		new_x = np.append(tem_dat_T[j:j+9].flatten(),[1])
		train_x.append(new_x)
		train_y.append(tem_dat[9][j+9])

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
print(train_x.shape)

w = np.zeros(len(train_x[0]))
S = 0
epoch = 100000
train_x_t = train_x.transpose()
lumbda = 0.5

for i in range(epoch):
	hypo = np.dot(train_x,w)
	loss = hypo-train_y
	gra = 2*np.dot(train_x_t,loss)+2*lumbda*np.sum(w)
	S = S+gra**2
	lr = 0.1/(S**0.5)
	w = w-lr*gra

RMSE = 0
for i in range(len(loss)):
	RMSE += loss[i]**2
print((RMSE/len(loss))**0.5)

np.save('model.npy',w)