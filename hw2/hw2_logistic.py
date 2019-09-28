import csv
import numpy as np
import sys

max_X = np.load('max_X.npy')
w = np.load('model.npy')

x_test = np.genfromtxt(sys.argv[1],delimiter=',')
x_test = np.delete(x_test,0,0)
x_test = x_test / max_X
b = np.ones((x_test.shape[0], 1))
x_test = np.concatenate((x_test,b), axis = 1)

z_hypo = []
z_test = np.dot(x_test,w)
z_ep = np.exp(-z_test)
z_sigmoid = 1/(1+z_ep)
for i in range(x_test.shape[0]):
	if z_sigmoid[i]<0.5:
		z_hypo.append(0)
	else:
		z_hypo.append(1)

idnum = []
for i in range(x_test.shape[0]):
	idnum.append(i+1)
ans = []
ans.append(idnum)
ans.append(z_hypo)
ttle = np.asarray([["id","label"]])
ans = np.asarray(ans)
ans = np.concatenate((ttle,ans.transpose()), axis = 0)
np.savetxt(sys.argv[2],ans,delimiter=',',fmt="%s")
